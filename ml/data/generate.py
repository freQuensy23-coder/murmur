"""ONE command for the PLAN_V2 stage-C synthetic set: plan -> teacher (resumable) -> build -> filter -> dedup -> split.

  tools/with-teacher .cache/ml/data/venv/bin/python -m ml.data.generate --out .cache/ml/data/synth_v2 \\
      --n 3000 --dev 300 --test 300 [--mined ml/roundtrip/out/mined] [--max-requests 700]

  python -m ml.data.generate --teacher stub --out /tmp/x --n 300 --dev 30 --test 30     # offline plumbing test

Writes OUT/{train,dev,test}.jsonl (shared contract), OUT/scenarios.jsonl, OUT/teacher/*.jsonl (resumable
shards: re-running the same command continues), OUT/rejects.jsonl (every rejected example with reasons),
OUT/review.jsonl (kept records whose target has a content word found in none of spoken / transcript / screen:
for a human pass), OUT/stats.json and OUT/REPORT.md. Records that near-duplicate a held-out eval item
(prompts/eval, ml/data/contam.py) are dropped. Each (split, category) keeps exactly the target count of
sampler.CATEGORY_TARGETS when enough examples survive (shortfalls are reported, never padded).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from ml.common import text as T
from ml.common.render_bridge import REPO
from ml.common.schema import write_jsonl
from ml.common.util import stable_id
from ml.data.contam import contaminated
from ml.data.synth import build as B
from ml.data.synth import disfluency as D
from ml.data.synth import filters as F
from ml.data.synth import sampler as S
from ml.data.synth import screen as SCR
from ml.data.synth import teacher as TC
from ml.data.synth.channel import Channel
from ml.data.synth.teacher_prompts import BATCH_SCHEMA, GEN_VERSION, gen_request

TONES = S.TONES


def make_jobs(scs: list[dict], batch: int, prefix: str = "gen") -> list[dict]:
    """Batches of scenarios with the same language (one shared prompt per request)."""
    jobs = []
    by_lang = defaultdict(list)
    for sc in scs:
        by_lang[sc["lang"]].append(sc)
    for lang in sorted(by_lang):
        lst = by_lang[lang]
        for i in range(0, len(lst), batch):
            chunk = lst[i:i + batch]
            ins, inp = gen_request(chunk, TONES)
            # the id hashes the exact request text: a changed scenario or prompt is never answered from an old shard
            jobs.append({"id": stable_id(prefix, GEN_VERSION, ins, inp), "instructions": ins, "input": inp,
                         "schema": BATCH_SCHEMA, "name": "examples", "sids": [s["sid"] for s in chunk], "gen": GEN_VERSION})
    return jobs


def interleave(jobs: list[dict], by_sid: dict) -> list[dict]:
    """Request order only (ids and batches unchanged): round-robin over (category, language) of each batch's first
    scenario, so the first answers of a long run cover every category and can be reviewed early."""
    groups = defaultdict(list)
    for j in jobs:
        sc = by_sid[j["sids"][0]]
        groups[(sc["category"], sc["lang"])].append(j)
    keys = sorted(groups)
    out, i = [], 0
    while len(out) < len(jobs):
        for k in keys:
            if i < len(groups[k]):
                out.append(groups[k][i])
        i += 1
    return out


RESERVE_SEED_OFFSET = 1000
RESERVE_INDEX_OFFSET = 100000


def parse_reserve(spec: str) -> list[dict]:
    """'train/context_resolution=120,dev/injection=6;train/address=6' -> one dict per ';'-separated layer,
    {("train", "context_resolution"): 120, ...}. A later layer adds scenarios after those of the earlier ones and
    is batched apart, so extending a run never changes a batch the teacher already answered."""
    layers = []
    for layer in filter(None, (x.strip() for x in (spec or "").split(";"))):
        out = {}
        for part in filter(None, (x.strip() for x in layer.split(","))):
            cell, k = part.split("=")
            split, cat = cell.split("/")
            if cat in S.CODE_ONLY:
                raise SystemExit(f"--reserve: {cat} is code-made (counterparts come from context_resolution reserves)")
            out[(split, cat)] = int(k)
        layers.append(out)
    return layers


def reserve_plan(a, layers: list[dict]) -> list[dict]:
    """Reserve scenarios for cells that fell short: the plan of another seed (same shares, its own groups), the next
    k per (split, category) for each layer. They sort after every primary scenario, so they only fill a shortfall;
    the primary plan, its batches and their job ids stay unchanged. sc["reserve"] = layer number (1, 2, ...)."""
    if isinstance(layers, dict):
        layers = [layers]
    if not any(layers):
        return []
    alt = S.plan(a.n, a.dev, a.test, a.seed + RESERVE_SEED_OFFSET, overgen=1.0,
                 lang_profile=getattr(a, "lang_profile", "ru_en"))
    by_cell = defaultdict(list)
    for sc in alt:
        by_cell[(sc["split"], sc["category"])].append(sc)
    out, used = [], defaultdict(int)
    for li, cells in enumerate(layers, 1):
        for c, k in cells.items():
            for sc in by_cell[c][used[c]:used[c] + k]:
                out.append(sc | {"reserve": li, "index": sc["index"] + RESERVE_INDEX_OFFSET})
            used[c] += k
    order = {sc["sid"]: i for i, sc in enumerate(alt)}
    return sorted(out, key=lambda sc: (sc["reserve"], order[sc["sid"]]))


def stratified_limit(scs: list[dict], k: int) -> list[dict]:
    """First k scenarios taken round-robin over (category, split), so a small test run covers every category."""
    groups = defaultdict(list)
    for sc in scs:
        groups[(sc["category"], sc["split"])].append(sc)
    keys = sorted(groups, key=lambda g: (g[1] != "train", g[0]))
    out, i = [], 0
    while len(out) < k and any(groups.values()):
        g = keys[i % len(keys)]
        if groups[g]:
            out.append(groups[g].pop(0))
        i += 1
        if i > 10 * (k + len(keys)):
            break
    return out


def examples_from(results: dict, jobs: list[dict]) -> tuple[dict, Counter]:
    ex, bad = {}, Counter()
    for j in jobs:
        r = results.get(j["id"])
        if not r:
            bad["job_failed"] += len(j["sids"])
            continue
        got = {e["sid"]: e for e in r["result"]["examples"] if e.get("sid") in j["sids"]}
        for sid in j["sids"]:
            if sid in got:
                ex[sid] = got[sid] | {"_teacher": r.get("model", ""), "_gen": r.get("gen") or "gen.v2"}
            else:
                bad["sid_missing_in_batch"] += 1
    return ex, bad


def run(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    plan = S.plan(a.n, a.dev, a.test, a.seed, overgen=a.overgen, lang_profile=getattr(a, "lang_profile", "ru_en"))
    reserves = reserve_plan(a, parse_reserve(getattr(a, "reserve", "")))
    plan += reserves
    tgt_counts = S.targets(a.n, a.dev, a.test)
    with open(out / "scenarios.jsonl", "w", encoding="utf-8") as f:
        for sc in plan:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")
    teach = [sc for sc in plan if sc["category"] not in S.CODE_ONLY]
    if a.limit:
        teach = stratified_limit(teach, a.limit)
        keep_sids = {sc["sid"] for sc in teach}
    channel = Channel(a.mined)  # fail fast on a bad --mined dir (workers build their own)
    teacher = TC.make_teacher(a.teacher, max_requests=a.max_requests)
    # reserves are batched apart, so the primary batches (and their job ids) never change
    layers = sorted({sc.get("reserve") or 0 for sc in teach})
    jobs = interleave([j for li in layers for j in make_jobs([sc for sc in teach if (sc.get("reserve") or 0) == li], a.batch)],
                      {sc["sid"]: sc for sc in teach})
    offline = getattr(a, "offline", False)
    if offline:  # rebuild from the answered shards only: no request is ever sent, unanswered jobs are reported
        answered = TC.load_results(out / "teacher", "gen")
        st_t = {"jobs": len(jobs), "ok": sum(j["id"] in answered for j in jobs), "offline": True,
                "unanswered": sum(j["id"] not in answered for j in jobs), "http_requests": 0, "wall_s": 0.0}
    else:
        st_t = TC.run_sharded(jobs, teacher, out / "teacher", "gen", shard_size=a.shard_size,
                              concurrency=a.concurrency, max_retries=a.max_retries)
    ex, bad = examples_from(TC.load_results(out / "teacher", "gen"), jobs)
    # one top-up pass for scenarios a batch dropped or a failed job lost
    missing = [sc for sc in teach if sc["sid"] not in ex]
    st_top = None
    if missing and offline:
        ex.update(examples_from(TC.load_results(out / "teacher", "topup"), make_jobs(missing, a.batch, prefix="topup"))[0])
    elif missing and not st_t.get("quota_stop") and a.topup:
        tjobs = make_jobs(missing, a.batch, prefix="topup")
        st_top = TC.run_sharded(tjobs, teacher, out / "teacher", "topup", shard_size=a.shard_size,
                                concurrency=a.concurrency, max_retries=a.max_retries)
        ex2, _ = examples_from(TC.load_results(out / "teacher", "topup"), tjobs)
        ex.update(ex2)
    if getattr(a, "stage", "all") == "teacher":  # shards only; the same command without --stage builds from them (0 requests)
        st = {"stage": "teacher", "teacher_jobs": len(jobs), "teacher_run": st_t, "topup_run": st_top,
              "teacher_examples": len(ex), "batch_problems": dict(bad), "wall_s": round(time.time() - t0, 1)}
        (out / "teacher_stage.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        return st
    # build: render + OCR + noise + filters (parallel workers; each owns its OCR session)
    recs, rejects, reasons = [], [], Counter()
    ocr_drops: dict = {"by_category": Counter(), "by_term": Counter(), "by_rule": Counter()}
    by_sid = {sc["sid"]: sc for sc in plan}
    todo = []
    for sc in plan:
        if a.limit and sc["category"] not in S.CODE_ONLY and sc["sid"] not in keep_sids:
            continue
        if sc["category"] == "context_counterpart":
            continue
        if sc["category"] != "noise_only" and sc["sid"] not in ex:
            reasons["no_teacher_example"] += 1
            continue
        todo.append((sc, ex.get(sc["sid"])))
    if getattr(a, "sample", 0):  # scratch build of the answered shards: K scenarios round-robin over (category, split)
        keep = {sc["sid"] for sc in stratified_limit([t[0] for t in todo], a.sample)}
        todo = [t for t in todo if t[0]["sid"] in keep]
    backend_info, timing = _backend_probe(a.screens)
    if backend_info["backend"] == "fixture" and not a.limit and a.n >= 1000 and not a.allow_fixture:
        raise SystemExit("refusing a full-size run with the fixture screen backend (no OCR): wait for "
                         "ml/screens/READY + ml/ocr/READY or pass --allow-fixture")
    tb = time.time()
    # every word the teacher had users say: with the channel's real words, the vocabulary that tells a real-word
    # substitution ('Would' for 'Could') from a misspelling (build.real_word_guess)
    vocab = sorted({w for e in ex.values() for w in T.norm_words(e.get("spoken") or "")})
    results = _build_all(todo, a, teacher.name, vocab)
    build_wall = time.time() - tb
    for sc, (built, err, tm) in zip([t[0] for t in todo], results):
        for k, v in tm.items():
            timing[k] = timing.get(k, 0.0) + v
        if err:
            reasons["build:" + err["reason"]] += 1
            if err["reason"].startswith("ocr:"):
                ocr_drops["by_category"][sc["category"]] += 1
                ocr_drops["by_rule"][err["reason"][4:]] += 1
                if err.get("term"):
                    ocr_drops["by_term"][err["term"]] += 1
            rejects.append({"sid": sc["sid"], "category": sc["category"], "reasons": ["build:" + err["reason"]],
                            "term": err.get("term"), "example": ex.get(sc["sid"])})
            continue
        for rec in built:
            if "_reject" in rec:
                reasons["build:" + rec["_reject"]] += 1
                if rec["_reject"].startswith("ocr:"):
                    ocr_drops["by_category"][rec["category"]] += 1
                    ocr_drops["by_rule"][rec["_reject"][4:]] += 1
                    if rec.get("term"):
                        ocr_drops["by_term"][rec["term"]] += 1
                rejects.append({"sid": sc["sid"], "category": rec["category"], "reasons": ["build:" + rec["_reject"]],
                                "term": rec.get("term")})
                continue
            why = F.check(rec)
            if why:
                for w in why:
                    reasons["filter:" + w] += 1
                rejects.append({"sid": sc["sid"], "category": rec["category"], "reasons": why, "record": rec})
            else:
                recs.append(rec)
    if a.limit:  # noise_only is code-made: keep it proportional in a limited run
        n_no = max(1, round(len(recs) * S.CATEGORY_TARGETS["noise_only"] / 3000))
        no = [r for r in recs if r["category"] == "noise_only"][:n_no]
        recs = [r for r in recs if r["category"] != "noise_only"] + no
    recs, dups = F.dedup(recs)
    for rid, why, other in dups:
        reasons["dedup:" + why] += 1
    # held-out eval sets (prompts/eval) must not leak into the synthetic data, in any split
    clean = []
    for r in recs:
        hit = contaminated(r)
        if hit:
            reasons["dedup:eval_contamination"] += 1
            rejects.append({"sid": r["meta"]["scenario"], "category": r["category"],
                            "reasons": ["eval_contamination:" + hit], "record": r})
        else:
            clean.append(r)
    recs = clean
    # the prompt's false-friend illustration ('точка сбора у входа') still echoes in some formatting examples:
    # at most ECHO_CAP of them per split, so one phrase does not become the false friend the model learns
    recs, echo = cap_prompt_echo(recs, by_sid)
    for _ in echo:
        reasons["dedup:prompt_echo_cap"] += 1
    # every survivor before the exact-count cut (selection can be redone without a rebuild)
    write_jsonl(out / "survivors.jsonl", recs)
    final, short = select_exact(recs, tgt_counts, by_sid, keep_all=bool(a.limit or getattr(a, "sample", 0)))
    leak = F.leakage(final)
    for split in ("train", "dev", "test"):
        write_jsonl(out / f"{split}.jsonl", [r for r in final if r["split"] == split])
    write_jsonl(out / "rejects.jsonl", rejects, key_order=False)
    review = [{"id": r["id"], "split": r["split"], "category": r["category"], "words": fl,
               "transcript": r["transcript"], "target": r["target"], "spoken": r["meta"].get("spoken", "")}
              for r in final if (fl := F.review_flags(r))]
    write_jsonl(out / "review.jsonl", review, key_order=False)
    stats = {
        "args": {k: v for k, v in vars(a).items()}, "teacher": getattr(getattr(teacher, "cfg", None), "public", lambda: {"name": "stub"})(),
        "provenance": provenance(teacher),
        "noise_source": channel.source, "scenarios_planned": len(plan), "reserve_scenarios": len(reserves),
        "final_from_reserve": sum(1 for r in final if by_sid[r["meta"]["scenario"]].get("reserve")),
        "teacher_scenarios": len(teach),
        "teacher_jobs": len(jobs), "teacher_run": st_t, "topup_run": st_top, "teacher_examples": len(ex),
        "batch_problems": dict(bad), "built_and_filtered_kept": len(recs) + len(dups), "after_dedup": len(recs),
        "final": len(final), "reasons": dict(reasons.most_common()), "shortfall": short, "leakage": leak,
        "final_by_split": dict(Counter(r["split"] for r in final)),
        "final_by_category": dict(Counter(r["category"] for r in final).most_common()),
        "final_by_lang": dict(Counter(r["lang"] for r in final)),
        "final_by_asr": dict(Counter(r["meta"].get("asr_model", "-") for r in final).most_common()),
        "review_flagged": len(review),
        "asr_disfluency": asr_disfluency(final, channel),
        "final_by_app": dict(Counter(r["ctx"]["app"]["category"] for r in final).most_common()),
        "final_with_screen": sum(1 for r in final if r["ctx"].get("screen")),
        "final_without_screen": dict(Counter(r["ctx"].get("privacy", {}).get("reason") or "-" for r in final
                                             if not r["ctx"].get("screen"))),
        "final_without_field": sum(1 for r in final if "field" not in r["ctx"]),
        "final_by_device": dict(Counter(r["meta"].get("screen", {}).get("spec", {}).get("device", "-") for r in final)),
        "final_rerendered": sum(1 for r in final if r["meta"].get("screen", {}).get("variant") == 1),
        "screens": backend_info,
        "ocr_drops": {k: dict(v.most_common()) for k, v in ocr_drops.items()},
        "timing": {"teacher_wall_s": st_t.get("wall_s", 0) + ((st_top or {}).get("wall_s") or 0),
                   "build_wall_s": round(build_wall, 1), "workers": a.workers,
                   **{k: round(v, 1) for k, v in timing.items()}},
        "wall_s": round(time.time() - t0, 1),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "REPORT.md").write_text(report(stats), encoding="utf-8")
    return stats


ECHO_PHRASES = ("точка сбора", "period of time")
ECHO_CAP = 3


def cap_prompt_echo(recs: list[dict], by_sid: dict) -> tuple[list[dict], list[dict]]:
    seen, keep, drop = defaultdict(int), [], []
    for r in sorted(recs, key=lambda r: by_sid[r["meta"]["scenario"]]["index"]):
        low = (r["meta"].get("spoken") or "").lower()
        hit = next((p for p in ECHO_PHRASES if p in low), None)
        if hit and r["category"] == "formatting_commands":
            seen[(r["split"], hit)] += 1
            if seen[(r["split"], hit)] > ECHO_CAP:
                drop.append(r)
                continue
        keep.append(r)
    ids = {r["id"] for r in keep}
    return [r for r in recs if r["id"] in ids], drop


def select_exact(recs: list[dict], tgt_counts: dict, by_sid: dict, keep_all: bool = False) -> tuple[list[dict], dict]:
    """Exact counts per (split, category): first survivors in plan order (primary scenarios before reserves).
    A counterpart is useless without its context-resolution parent, so context_resolution first takes as many
    parents with a surviving counterpart as the counterpart cell needs, then unpaired scenarios; counterparts are
    drawn only from the selected parents (a parent stays useful alone)."""
    idx = lambda r: by_sid[r["meta"]["scenario"]]["index"]  # noqa: E731
    with_cp = {r["meta"]["counterpart_of"] for r in recs if r["category"] == "context_counterpart"}
    by_cell = defaultdict(list)
    for r in recs:
        by_cell[(r["split"], r["category"])].append(r)
    final, short = [], {}
    for split, cc in tgt_counts.items():
        cats = sorted(cc, key=lambda c: c == "context_counterpart")  # parents first
        chosen = set()
        for cat in cats:
            c = cc[cat]
            cell = by_cell.get((split, cat), [])
            if cat == "context_resolution":
                # as many parents with a surviving counterpart as the counterpart cell needs, then the scenarios
                # planned without a counterpart (a surplus paired parent is one more Cyrillic garble; the unpaired
                # ones carry the recognisers' Latin misspellings), plan order inside each group
                paired = sorted((r for r in cell if r["id"] in with_cp), key=idx)[:cc.get("context_counterpart", 0)]
                ids = {r["id"] for r in paired}
                cell = paired + sorted((r for r in cell if r["id"] not in ids),
                                       key=lambda r: (bool(by_sid[r["meta"]["scenario"]].get("counterpart")), idx(r)))
            elif cat == "context_counterpart":
                cell = sorted((r for r in cell if r["meta"]["counterpart_of"] in chosen), key=idx)
            else:
                cell = sorted(cell, key=idx)
            if not keep_all and len(cell) < c:
                short[f"{split}/{cat}"] = {"target": c, "got": len(cell)}
            take = cell if keep_all else cell[:c]
            chosen |= {r["id"] for r in take}
            final += take
    return final, short


def provenance(teacher) -> dict:
    """Who wrote the targets, and the open licence question (docs/ml/DATA_CARD.md section 2a)."""
    name = getattr(getattr(teacher, "cfg", None), "model", None) or getattr(teacher, "name", "stub")
    return {"teacher_model": name, "teacher_api": "OpenAI-compatible Responses API (local proxy, owner's account)"
            if name != "stub" else "stub", "generated": time.strftime("%Y-%m-%d"), "prompt": GEN_VERSION,
            "terms_status": "UNCONFIRMED: the teacher provider's terms on training models with its outputs and on "
                            "publishing such models need the owner's confirmation before any public upload",
            "third_party_text": "none copied; screen content and targets are teacher-written, noise tables from "
                                "ml/roundtrip, fonts per ml/screens"}


def asr_disfluency(final: list[dict], channel: Channel) -> dict:
    """Per ASR model: hesitation tokens per 100 transcript words in the final set, next to the model's render
    profile (probe-measured survival of hesitation sounds and repeats) and the mined profile's WER."""
    out: dict = {}
    for r in final:
        m = r["meta"].get("asr_model")
        if not m:
            continue
        o = out.setdefault(m, {"records": 0, "words": 0, "hesitation_tokens": 0, "hesitations_said": 0,
                               "hesitations_written": 0})
        o["records"] += 1
        o["words"] += len(r["transcript"].split())
        o["hesitation_tokens"] += D.hesitation_tokens(r["transcript"])
        for op in r["meta"].get("disfluency") or []:
            if op["op"] in ("filler", "hesitation") and T.norm_word(op["text"]) in T.HESITATION:
                o["hesitations_said"] += 1
                o["hesitations_written"] += 0 if op.get("suppressed") else 1
    for m, o in out.items():
        o["hesitation_per_100w"] = round(100 * o["hesitation_tokens"] / max(1, o["words"]), 2)
        o["render"] = channel.disfluency_render(m)
        if channel.nm is not None and m in channel.nm.profiles:
            o["mined_wer"] = channel.nm.profiles[m].get("wer")
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ parallel build
_W: dict = {}


def _winit(mined, screens, vocab=()):
    import os
    B.set_vocab(vocab)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MURMUR_OCR_THREADS"):
        os.environ.setdefault(v, "1")
    _W["channel"] = Channel(mined)
    _W["backend"] = SCR.Backend(screens)


def _build_one(job):
    sc, e, teacher_name = job
    ch, be = _W["channel"], _W["backend"]
    t0, c0, r0, o0 = be.timing["build_ctx_s"], be.timing["calls"], be.timing["render_s"], be.timing["ocr_s"]
    h0 = be.timing.get("cache_hits", 0)
    t = time.time()
    try:
        if sc["category"] == "noise_only":
            built = B.build_noise_only(sc, ch, be)
        else:
            built = B.build(sc, e, ch, e.get("_teacher", teacher_name), be, gen=e.get("_gen"))
        err = None
    except B.Reject as rj:
        built, err = [], {"reason": str(rj), "term": getattr(rj, "term", None)}
    tot = time.time() - t
    ocr = be.timing["build_ctx_s"] - t0
    return built, err, {"screen_build_ctx_s": ocr, "screen_calls": be.timing["calls"] - c0, "code_s": tot - ocr,
                        "render_s": be.timing["render_s"] - r0, "ocr_s": be.timing["ocr_s"] - o0,
                        "screen_cache_hits": be.timing.get("cache_hits", 0) - h0}


def _build_all(todo, a, teacher_name, vocab=()):
    jobs = [(sc, e, teacher_name) for sc, e in todo]
    t0, out = time.time(), []

    def tick():
        if len(out) % 100 == 0 or len(out) == len(jobs):
            print(f"[build] {len(out)}/{len(jobs)} scenarios {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    if a.workers <= 1:
        _winit(a.mined, a.screens, vocab)
        for j in jobs:
            out.append(_build_one(j))
            tick()
        return out
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers, initializer=_winit, initargs=(a.mined, a.screens, vocab)) as pool:
        for res in pool.imap(_build_one, jobs, chunksize=2):  # ordered, streamed for progress
            out.append(res)
            tick()
    return out


def _backend_probe(kind):
    be = SCR.Backend(kind)
    info = dict(be.version) | {"renderer_v2": SCR.renderer_shows_screen(),
                               "ocr_hard_terms": len(SCR.ocr_hard_terms())}
    return info, {}


def report(st: dict) -> str:
    t = st["teacher_run"]
    lines = ["# Synthetic set (PLAN_V2 stage C) — run report", "",
             f"Teacher: {st['teacher']}; noise: `{st['noise_source']}`.", "",
             f"Scenarios planned {st['scenarios_planned']}, sent to the teacher {st['teacher_scenarios']} in "
             f"{st['teacher_jobs']} batches; teacher examples received {st['teacher_examples']}; "
             f"kept after build+filters {st['built_and_filtered_kept']}, after dedup {st['after_dedup']}, final {st['final']}.",
             "", f"Teacher run: {json.dumps(t, ensure_ascii=False)}", "", "## Final counts", "",
             "| split | n |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in st["final_by_split"].items()] + [
             "", "| category | n |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in st["final_by_category"].items()] + [
             "", "| lang | n |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in st["final_by_lang"].items()] + [
             "", "## Why examples were dropped", "", "| reason | n |", "|---|---:|"] + [
             f"| {k} | {v} |" for k, v in st["reasons"].items()] + [
             "", f"Shortfall vs targets: {json.dumps(st['shortfall'], ensure_ascii=False) or 'none'}",
             "", f"Leakage check: {json.dumps(st['leakage'], ensure_ascii=False)}", "",
             "## Screens (CONTEXT_V2)", "", f"Backend: `{json.dumps(st['screens'], ensure_ascii=False)}`", "",
             f"With screen {st['final_with_screen']}, without screen {json.dumps(st['final_without_screen'])}, "
             f"without field {st['final_without_field']}, re-rendered once {st['final_rerendered']}, "
             f"devices {json.dumps(st['final_by_device'])}.", "",
             f"OCR-rule drops: {json.dumps(st['ocr_drops'], ensure_ascii=False)}", "",
             f"Timing: {json.dumps(st['timing'])}", "",
             f"Review list (target words found in none of spoken/transcript/screen): {st['review_flagged']} records "
             "in review.jsonl.", "",
             "## ASR disfluency rendering per model", "",
             "| model | records | hesitation tokens / 100 words | hesitations said -> written | render (hes/rep keep, source) |",
             "|---|---:|---:|---|---|"] + [
             f"| {m} | {o['records']} | {o['hesitation_per_100w']} | {o['hesitations_said']} -> {o['hesitations_written']} | "
             f"{o['render']['hesitation_keep']}/{o['render']['repeat_keep']} {o['render']['source']} |"
             for m, o in st["asr_disfluency"].items()] + [""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000, help="training examples")
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lang-profile", default="ru_en", choices=["ru_en", "hebrew"],
                    help="language mix profile: ru_en (ru/en/mixed) or hebrew (he/he_en/he_ru)")
    ap.add_argument("--teacher", default="responses", choices=["responses", "chat", "stub"])
    ap.add_argument("--mined", default=None, help="mined error tables dir (ml/roundtrip); default builtin simulator")
    ap.add_argument("--batch", type=int, default=6, help="scenarios per teacher request")
    ap.add_argument("--overgen", type=float, default=1.15, help="scenario over-generation to absorb filter losses")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--max-requests", type=int, default=0, help="hard cap on HTTP requests (0 = none)")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--shard-size", type=int, default=25, help="teacher jobs per resumable shard")
    ap.add_argument("--limit", type=int, default=0, help="only K teacher scenarios, stratified (test runs)")
    ap.add_argument("--no-topup", dest="topup", action="store_false")
    ap.add_argument("--screens", default="auto", choices=["auto", "real", "fixture"],
                    help="ctx source: real = ml.screens.build_ctx (render + OCR); auto = real when READY files exist")
    ap.add_argument("--allow-fixture", action="store_true", help="allow a full-size run on the fixture backend")
    ap.add_argument("--workers", type=int, default=4, help="build processes (render + OCR); <= 6 on shared CPUs")
    ap.add_argument("--reserve", default="", help="extra scenarios for short cells: 'train/near_miss=40,dev/list=3' "
                    "(another seed's plan; used only after every primary survivor)")
    ap.add_argument("--stage", default="all", choices=["all", "teacher"],
                    help="teacher = stop after the (resumable) teacher shards; all = build the set as well")
    ap.add_argument("--offline", action="store_true", help="build from the answered teacher shards only (never sends "
                    "a request; unanswered jobs are reported)")
    ap.add_argument("--sample", type=int, default=0, help="scratch build: only K answered scenarios, round-robin over "
                    "(category, split); job ids unchanged")
    ap.add_argument("--ctx-cache", default=None, help="directory caching render + OCR results per render spec "
                    "(rebuilds that change only the channel or the filters skip render + OCR)")
    a = ap.parse_args(argv)
    if a.ctx_cache:
        import os
        os.environ["MURMUR_CTX_CACHE"] = str(Path(a.ctx_cache).resolve())
    if a.concurrency is None:
        a.concurrency = TC.TeacherConfig().concurrency if a.teacher != "stub" else 4
    st = run(a)
    if a.stage == "teacher":
        print(json.dumps(st, ensure_ascii=False, indent=1))
        return 0
    print(json.dumps({k: st[k] for k in ("final", "final_by_split", "teacher_run", "shortfall", "leakage", "wall_s")},
                     ensure_ascii=False, indent=1))
    print(f"report: {Path(a.out) / 'REPORT.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
