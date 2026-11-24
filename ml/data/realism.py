"""Synthetic transcripts vs the real round trip: measured tables for the channel and an audit of a built set.

  python -m ml.data.realism measure  [--rows ROWS]            -> ml/data/probe/asr_style.v1.json
  python -m ml.data.realism calibrate --data DIR [--rows ROWS] -> ml/data/probe/word_noise_scale.v1.json
  python -m ml.data.realism audit --data DIR [--rows ROWS] [--json OUT]

ROWS = the real text -> TTS -> ASR round trip (ml/roundtrip runner output, `clean` channel only), default
.cache/ml/roundtrip/out/full/roundtrip/full.jsonl.

measure    per (ASR model, language): share of hypotheses with any punctuation, with a sentence-final mark given
           any punctuation, curly apostrophes, the share of reference words the hypothesis misses (non-term,
           non-number words) and the first-word miss rate (streaming models drop the first word).
calibrate  per (model, language) factor on the channel's non-term word error rates: previous factor x (real miss /
           synthetic miss of the built set DIR) ** 0.7, clipped to [0.6, 2.0] (the recoverability filter removes part of the
           extra noise, so the factor is refit on a rebuilt set).
audit      the numbers behind the 2026-11-20 data review findings (terms restored without screen support, real-word
           substitutions the target restores, term-garble scripts, word miss ratio, final punctuation, digit/word
           numbers, artifact strings), for before/after comparisons.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from ml.common.render_bridge import REPO

ROWS = REPO / ".cache" / "ml" / "roundtrip" / "out" / "full" / "roundtrip" / "full.jsonl"
PROBE = Path(__file__).resolve().parent / "probe"
STYLE_FILE = PROBE / "asr_style.v1.json"
SCALE_FILE = PROBE / "word_noise_scale.v1.json"
END_MARKS = ".!?…"
# above about 2x the extra noise mostly makes the examples unrecoverable: the recogniser then drops out of the set
# (Qwen3-ASR Russian 75 -> 20 records at 3.0) instead of its transcripts getting noisier
MAX_FACTOR = 2.0
NUMW = set("one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty hundred thousand "
           "один два три четыре пять шесть семь восемь девять десять двадцать тридцать сорок пятьдесят сто тысяч "
           "тысячи "
           "אחד אחת שתיים שניים שלוש שלושה ארבע ארבעה חמש חמישה שש שישה שבע שבעה שמונה תשע תשעה עשר עשרה "
           "עשרים שלושים ארבעים חמישים מאה מאתיים אלף אלפיים".split())


def toks(s: str) -> list[str]:
    s = re.sub(r"[\u0591-\u05c7]", "", s or "")
    return re.findall(r"[\w'\-׳״]+", s.lower().replace("ё", "е"))


def _matched(a: list[str], b: list[str]) -> set[int]:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out: set[int] = set()
    for bl in sm.get_matching_blocks():
        out.update(range(bl.a, bl.a + bl.size))
    return out


def miss(ref: str, hyp: str, excl: set[str]) -> tuple[int, int]:
    """Reference words (not in excl, no digits) the hypothesis does not contain in order: (missed, counted)."""
    a, b = toks(ref), toks(hyp)
    m = _matched(a, b)
    idx = [i for i, t in enumerate(a) if t not in excl and not re.search(r"\d", t)]
    return sum(i not in m for i in idx), len(idx)


def has_punct(h: str) -> bool:
    return bool(re.search(r"[,.!?…;:]", h))


def ends_mark(h: str) -> bool:
    return h.rstrip()[-1:] in END_MARKS


def load_rows(path: Path = ROWS) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        x = json.loads(line)
        if x.get("channel") == "clean" and x.get("hypothesis"):
            out.append(x)
    return out


# ------------------------------------------------------------------ real tables
def style_table(rows: list[dict]) -> dict:
    acc: dict = defaultdict(lambda: Counter())
    for x in rows:
        c = acc[(x["asr_model"], x["lang"])]
        h = x["hypothesis"]
        c["n"] += 1
        c["any"] += has_punct(h)
        c["end_any"] += has_punct(h) and ends_mark(h)
        c["end"] += ends_mark(h)
        c["curly"] += "’" in h
        ex: set[str] = set()
        for t in x.get("terms") or []:
            ex.update(toks(t["term"]))
        m, n = miss(x["text"], h, ex | NUMW)
        c["miss"] += m
        c["words"] += n
        a = toks(x["text"])
        if len(a) >= 3:
            mt = _matched(a, toks(h))
            c["n3"] += 1
            c["first_miss"] += 0 not in mt
            c["mid_miss"] += sum(i not in mt for i in range(1, len(a) - 1))
            c["mid_words"] += len(a) - 2
    out: dict = {}
    for (m, lg), c in sorted(acc.items()):
        first = c["first_miss"] / max(1, c["n3"])
        mid = c["mid_miss"] / max(1, c["mid_words"])
        out.setdefault(m, {})[lg] = {
            "n": c["n"], "any_punct": round(c["any"] / c["n"], 4),
            "end_given_punct": round(c["end_any"] / max(1, c["any"]), 4), "end_rate": round(c["end"] / c["n"], 4),
            "curly_rate": round(c["curly"] / c["n"], 4), "word_miss": round(c["miss"] / max(1, c["words"]), 4),
            "first_word_miss": round(first, 4), "mid_word_miss": round(mid, 4),
            # a streaming model loses the first word far more often than any other word
            "first_word_drop": round(max(0.0, first - mid), 4) if first - mid >= 0.1 else 0.0}
    return out


def synth_miss(recs: list[dict]) -> dict:
    acc: dict = defaultdict(lambda: [0, 0])
    for x in recs:
        md = x["meta"]
        if not md.get("asr_model") or not md.get("spoken"):
            continue
        ex: set[str] = set()
        for e in md.get("entities") or []:
            ex.update(toks(e))
        g = md.get("garble") or {}
        for k in ("term", "person", "from"):
            if isinstance(g.get(k), str):
                ex.update(toks(g[k]))
        m, n = miss(md["spoken"], x["transcript"], ex | NUMW)
        a = acc[(md["asr_model"], x["lang"])]
        a[0] += m
        a[1] += n
    return {k: v[0] / max(1, v[1]) for k, v in acc.items()}


def load_split(d: Path, splits=("train",)) -> list[dict]:
    out = []
    for s in splits:
        p = Path(d) / f"{s}.jsonl"
        if p.exists():
            out += [json.loads(line) for line in open(p, encoding="utf-8")]
    return out


# ------------------------------------------------------------------ audit (review findings)
def _norm(s: str) -> str:
    return s.casefold().replace("ё", "е")


def _screen(x: dict) -> str:
    return " ".join(ln["text"] for ln in (x["ctx"].get("screen") or {}).get("lines", []))


def audit(d: Path, rows: list[dict] | None) -> dict:
    from ml.data.synth.build import is_well_known, real_word_guess
    from ml.data.synth import build as B
    tr = load_split(d, ("train",))
    allr = load_split(d, ("train", "dev", "test"))
    rep: dict = {"data": str(d), "train": len(tr)}
    # F1 terms restored without screen support
    fixed, kept = defaultdict(list), defaultdict(list)
    for x in tr:
        m = x["meta"]
        scr = _norm(_screen(x))
        if x["category"] == "context_counterpart":
            kept[m["garble"]["term"]].append(m["garble"]["form"])
            continue
        for e in m.get("noise", {}).get("events") or []:
            mm = re.match(r"term:(.+?)->(.+?):(\w+)$", e)
            if not mm:
                continue
            t, f, _ = mm.groups()
            if _norm(t) == _norm(f):
                continue
            if _norm(t) in _norm(x["target"]) and _norm(t) not in _norm(x["transcript"]) and _norm(t) not in scr:
                fixed[t].append((x["id"], f))
    cyr = lambda s: bool(re.search("[а-яё]", s, re.I))  # noqa: E731
    heb = lambda s: bool(re.search("[\u05d0-\u05ea]", s))  # noqa: E731
    nwk = [(t, i, f) for t, v in fixed.items() for i, f in v if not is_well_known(t)]
    conflict = sorted(t for t in set(fixed) & set(kept) if any(cyr(f) or heb(f) for _, f in fixed[t]))
    rep["F1_term_fix_no_screen_records"] = len({i for v in fixed.values() for i, _ in v})
    rep["F1_non_well_known_fix_records"] = len({i for _, i, _ in nwk})
    rep["F1_non_well_known_cyrillic_fix_records"] = len({i for _, i, f in nwk if cyr(f)})
    rep["F1_cyrillic_fix_vs_counterpart_conflicts"] = conflict
    rep["F1_latin_fix_vs_counterpart_terms"] = sorted(t for t in set(fixed) & set(kept) if t not in conflict)
    # F2 real-word substitutions restored by the target (reviewer metric + the generator's guess rule)
    vocab = Counter()
    for x in allr:
        vocab.update(toks(x["meta"].get("spoken") or ""))
    rw, guess = set(), set()
    for x in tr:
        scr = toks(_screen(x))
        tt = toks(x["target"])
        for e in x["meta"].get("noise", {}).get("events") or []:
            if not e.startswith("sub:"):
                continue
            a, b = e[4:].split("->", 1)
            if a.lower() == b.lower() or len(a) < 4:
                continue
            if not (a.lower() in tt and b.lower() not in tt):
                continue
            if vocab[b.lower()] >= 2 and a.lower() not in scr:
                rw.add(x["id"])
            if real_word_guess(a, b, x["target"], _screen(x), vocab_extra=set(vocab)):
                guess.add(x["id"])
    rep["F2_real_word_restored_records"] = len(rw)
    rep["F2_guess_rule_records"] = len(guess)
    # F3 garble script / source of context garbles (ru + mixed speech)
    gs, src = defaultdict(Counter), Counter()
    for x in tr:
        g = x["meta"].get("garble") or {}
        if "term" in g and x["lang"] != "en":
            gs[x["meta"]["asr_model"]]["heb" if heb(g["form"]) else "cyr" if cyr(g["form"]) else "lat"] += 1
        if "term" in g and x["category"] == "context_resolution":
            src[g.get("source")] += 1
    tot = Counter()
    for c in gs.values():
        tot.update(c)
    rep["F3_garble_script_ru_mixed"] = {m: dict(c) for m, c in sorted(gs.items())}
    rep["F3_garble_latin_share"] = round(tot["lat"] / max(1, sum(tot.values())), 3)
    rep["F3_cr_no_counterpart_latin"] = _cr_latin(tr)
    rep["F3_garble_sources_cr"] = dict(src)
    # F4 / F5 vs the real round trip
    if rows:
        real = style_table(rows)
        sm = synth_miss(tr)
        ratios, endp = {}, {}
        syn_end = defaultdict(lambda: [0, 0])
        for x in tr:
            m = x["meta"].get("asr_model")
            if m and x["meta"].get("spoken"):
                k = (m, x["lang"])
                syn_end[k][0] += ends_mark(x["transcript"])
                syn_end[k][1] += 1
        for (m, lg), v in sorted(sm.items()):
            r = (real.get(m) or {}).get(lg)
            if r and r["word_miss"]:
                ratios[f"{m.split('@')[0]}/{lg}"] = round(v / r["word_miss"], 2)
            if r and syn_end[(m, lg)][1]:
                endp[f"{m.split('@')[0]}/{lg}"] = f"{r['end_rate']:.2f}/{syn_end[(m, lg)][0] / syn_end[(m, lg)][1]:.2f}"
        rep["F4_word_miss_ratio_synth_over_real"] = ratios
        rep["F5_end_mark_real_over_synth"] = endp
    rep["F5_curly_transcripts"] = sum("’" in x["transcript"] for x in allr)
    # F6 numbers
    vals = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30}
    rev, style = [], Counter()
    for x in tr:
        t, h = x["target"], x["transcript"]
        for w, v in vals.items():
            if re.search(r"\b%d\b" % v, h) and re.search(r"\b%s\b" % w, t, re.I) and not re.search(r"\b%d\b" % v, t):
                rev.append(x["id"])
        if x["lang"] == "en":
            for m in re.finditer(r"\b(\w+) (minutes|hours|days|people|dollars|weeks)\b", t):
                style["word" if m.group(1).lower() in vals else "digit" if m.group(1).isdigit() else "other"] += 1
    rep["F6_digit_to_word_reversions"] = rev
    rep["F6_en_N_units"] = dict(style)
    rep["F6_dropped_named"] = sorted(B.BAD_TEACHER_EXAMPLES)
    # F7 artifacts
    art = defaultdict(lambda: defaultdict(Counter))
    models = Counter()
    for x in allr:
        if x["category"] not in ("asr_artifact", "noise_only"):
            continue
        if x["category"] == "asr_artifact":
            models[x["meta"].get("asr_model")] += 1
        for e in x["meta"].get("noise", {}).get("events") or []:
            if e.startswith(("tail:", "loop:")):
                art[x["category"]][x["split"]][e] += 1
        if x["category"] == "noise_only":
            art["noise_only"][x["split"]]["text:" + x["transcript"]] += 1
    rep["F7_asr_artifact_models"] = dict(models)
    for cat, by in art.items():
        tr_s = set(by["train"])
        held = set(by["dev"]) | set(by["test"])
        rep[f"F7_{cat}_distinct_strings"] = {s: len(by[s]) for s in ("train", "dev", "test")}
        rep[f"F7_{cat}_heldout_strings_in_train"] = len(held & tr_s)
    return rep


def _cr_latin(tr: list[dict]) -> dict:
    cp = {x["meta"].get("counterpart_of") for x in tr if x["category"] == "context_counterpart"}
    c = Counter()
    for x in tr:
        g = x["meta"].get("garble") or {}
        if x["category"] == "context_resolution" and x["lang"] != "en" and "form" in g:
            k = "with_counterpart" if x["id"] in cp else "no_counterpart"
            c[k + ("_lat" if re.search(r"[A-Za-z]", g["form"]) else "_cyr")] += 1
    return dict(c)


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["measure", "calibrate", "audit"])
    ap.add_argument("--rows", default=str(ROWS))
    ap.add_argument("--data", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--prev", default=str(SCALE_FILE))
    a = ap.parse_args(argv)
    rows = load_rows(Path(a.rows)) if Path(a.rows).exists() else None
    if a.cmd == "measure":
        if rows is None:
            raise SystemExit(f"no round-trip rows at {a.rows}")
        out = {"source": str(Path(a.rows).relative_to(REPO)) if str(a.rows).startswith(str(REPO)) else a.rows,
               "channel": "clean", "rows": len(rows), "measured": time.strftime("%Y-%m-%d"),
               "models": style_table(rows)}
        STYLE_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {STYLE_FILE} ({len(rows)} rows)")
        return 0
    if a.cmd == "calibrate":
        real = style_table(rows)
        sm = synth_miss(load_split(Path(a.data)))
        prev = json.loads(Path(a.prev).read_text(encoding="utf-8")).get("factors", {}) if Path(a.prev).exists() else {}
        fac: dict = {}
        for (m, lg), v in sorted(sm.items()):
            r = (real.get(m) or {}).get(lg)
            if not r or not r["word_miss"] or not v:
                continue
            old = (prev.get(m) or {}).get(lg, 1.0)
            # damped (exponent 0.7): one cell holds a few dozen missed words, so a single fit overshoots
            fac.setdefault(m, {})[lg] = round(min(MAX_FACTOR, max(0.6, old * (r["word_miss"] / v) ** 0.7)), 3)
        out = {"note": "multiplier on the mined non-term word substitution/deletion rates per (model, language); "
                       "previous factor x real/synthetic word-miss rate of the built set", "fitted_on": a.data,
               "fitted": time.strftime("%Y-%m-%d"), "factors": fac}
        SCALE_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(fac, indent=1))
        return 0
    rep = audit(Path(a.data), rows)
    txt = json.dumps(rep, ensure_ascii=False, indent=1)
    if a.json:
        Path(a.json).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
