"""Turnkey Girgur Hebrew+mixed shard generator and RU+EN+HE dataset merger.

Generates ~1,200 train / ~150 dev / ~150 test Hebrew (`he`, `he_en`, `he_ru`) records via
`ml.data.generate` (`lang_profile="hebrew"`, seed 101), then merges them with the existing
RU+EN dataset (default `.cache/ml/data/final/`) into `.cache/ml/data/girgur/` with full
schema validation, `recheck.py` CONTEXT_V2 verification, eval contamination check, and
split-leakage verification.

Usage (on openclaw-dp):
  tools/with-teacher .cache/ml/ocr/venv/bin/python -m ml.data.generate_hebrew \\
      --hebrew-out .cache/ml/data/hebrew_v1 \\
      --base-dir .cache/ml/data/final \\
      --out .cache/ml/data/girgur

Merge-only (after Hebrew shard is already generated):
  python -m ml.data.generate_hebrew --merge-only \\
      --hebrew-out .cache/ml/data/hebrew_v1 \\
      --base-dir .cache/ml/data/final \\
      --out .cache/ml/data/girgur

Offline smoke test (no network, no OCR):
  python -m ml.data.generate_hebrew --teacher stub --screens fixture \\
      --n 60 --dev 15 --test 15 --hebrew-out /tmp/he_stub --out /tmp/girgur_stub --hebrew-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from ml.common.render_bridge import REPO
from ml.common.schema import read_jsonl, validate_record, write_jsonl
from ml.data import contam, generate as G, recheck
from ml.data.synth import filters as F, teacher as TC

DEFAULT_BASE = REPO / ".cache" / "ml" / "data" / "final"
DEFAULT_HEBREW = REPO / ".cache" / "ml" / "data" / "hebrew_v1"
DEFAULT_OUT = REPO / ".cache" / "ml" / "data" / "girgur"
SPLITS = ("train", "dev", "test")


def merge_datasets(
    base_dir: Path | None,
    hebrew_dir: Path | list[Path],
    out_dir: Path,
    *,
    seed: int = 42,
    check_render: bool = True,
    run_recheck: bool = True,
    strict_schema: bool = True,
    fail_on_recheck: bool = False,
) -> dict:
    """Merge RU+EN (`base_dir`) and one or more Hebrew shard dirs (`hebrew_dir`) into `out_dir`.

    Returns a summary dict (also written to `out_dir/merge_stats.json`).
    Raises SystemExit on duplicate IDs, schema errors, or split leakage.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sources: list[tuple[str, Path]] = []
    if base_dir is not None:
        if not (base_dir / "train.jsonl").exists():
            raise SystemExit(f"base dataset missing {base_dir / 'train.jsonl'}")
        sources.append(("base", base_dir))
    he_dirs = [hebrew_dir] if isinstance(hebrew_dir, Path) else list(hebrew_dir)
    for idx, hd in enumerate(he_dirs):
        if not (hd / "train.jsonl").exists():
            raise SystemExit(f"hebrew dataset missing {hd / 'train.jsonl'}")
        sources.append((f"hebrew{'' if idx == 0 else idx + 1}", hd))

    seen_ids: set[str] = set()
    all_recs: list[dict] = []
    dropped: Counter = Counter()
    recheck_failures: Counter = Counter()
    schema_errors: list[tuple[str, list[str]]] = []
    per_source: Counter = Counter()

    for tag, d in sources:
        for split in SPLITS:
            p = d / f"{split}.jsonl"
            if not p.exists():
                continue
            for rec in read_jsonl(p):
                if rec.get("split") != split:
                    schema_errors.append((str(rec.get("id")), [f"split mismatch: {rec.get('split')!r} != {split!r}"]))
                    continue
                rid = rec.get("id")
                if rid in seen_ids:
                    raise SystemExit(f"duplicate record id {rid!r} when merging {tag}:{p}")
                seen_ids.add(rid)
                errs = validate_record(rec, check_render=check_render, strict=strict_schema)
                if errs:
                    schema_errors.append((str(rid), errs))
                    dropped["schema_invalid"] += 1
                    continue
                if run_recheck:
                    rerrs = recheck.rule_errors(rec)
                    if rerrs:
                        for e in rerrs:
                            recheck_failures[":".join(e.split(":")[:2])] += 1
                        dropped["recheck_failed"] += 1
                        continue
                hit = contam.contaminated(rec)
                if hit:
                    dropped["eval_contamination"] += 1
                    continue
                all_recs.append(rec)
                per_source[f"{tag}:{split}"] += 1

    if schema_errors:
        sample = schema_errors[:5]
        raise SystemExit(f"schema validation failed on {len(schema_errors)} record(s), first: {sample}")
    if fail_on_recheck and recheck_failures:
        raise SystemExit(f"recheck.py rule failures: {dict(recheck_failures)}")

    # Cross-shard near-duplicate removal (held-out splits win over train)
    kept, dups = F.dedup(all_recs)
    for _, why, _ in dups:
        dropped[f"dedup:{why}"] += 1

    # Strict split-leakage check (no entity or scenario group may cross train vs dev/test)
    leak = F.leakage(kept)
    if leak["entity_overlap_train_dev"] or leak["entity_overlap_train_test"] or leak["group_overlap"]:
        raise SystemExit(f"split leakage detected in merged dataset: {json.dumps(leak, ensure_ascii=False)}")

    rng = random.Random(seed)
    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for r in kept:
        by_split[r["split"]].append(r)
    rng.shuffle(by_split["train"])

    for split in SPLITS:
        write_jsonl(out_dir / f"{split}.jsonl", by_split[split])

    stats = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_dir": str(base_dir) if base_dir else None,
        "hebrew_dir": [str(p) for p in he_dirs] if len(he_dirs) > 1 else str(he_dirs[0]),
        "out_dir": str(out_dir),
        "loaded_by_source": dict(sorted(per_source.items())),
        "dropped": dict(dropped.most_common()),
        "recheck_failures": dict(recheck_failures.most_common()),
        "leakage": leak,
        "final_total": len(kept),
        "final_by_split": {s: len(by_split[s]) for s in SPLITS},
        "final_by_lang": dict(Counter(r["lang"] for r in kept).most_common()),
        "final_by_split_lang": {
            s: dict(Counter(r["lang"] for r in by_split[s]).most_common()) for s in SPLITS
        },
        "final_by_category": dict(Counter(r["category"] for r in kept).most_common()),
    }
    (out_dir / "merge_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hebrew-out", default=str(DEFAULT_HEBREW), help="output dir for the Hebrew+mixed shard")
    ap.add_argument("--extra-hebrew", action="append", default=[], help="additional Hebrew shard dirs to include in merge")
    ap.add_argument("--base-dir", default=str(DEFAULT_BASE), help="existing RU+EN dataset dir (.cache/ml/data/final)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="merged Girgur dataset dir (.cache/ml/data/girgur)")
    ap.add_argument("--merge-only", action="store_true", help="skip generation; merge --base-dir + --hebrew-out -> --out")
    ap.add_argument("--hebrew-only", action="store_true", help="skip merging with --base-dir (validate Hebrew shard into --out)")
    ap.add_argument("--n", type=int, default=1200, help="Hebrew+mixed training examples (default 1200)")
    ap.add_argument("--dev", type=int, default=150, help="Hebrew+mixed dev examples (default 150)")
    ap.add_argument("--test", type=int, default=150, help="Hebrew+mixed test examples (default 150)")
    ap.add_argument("--seed", type=int, default=101, help="planner seed (distinct from RU+EN seed=1)")
    ap.add_argument("--teacher", default="responses", choices=["responses", "chat", "stub"])
    ap.add_argument("--mined", default=None, help="optional mined tables dir (Hebrew uses builtin zipformer2_he/whisper_he)")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--overgen", type=float, default=1.20)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--shard-size", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-topup", dest="topup", action="store_false")
    ap.add_argument("--screens", default="auto", choices=["auto", "real", "fixture"])
    ap.add_argument("--allow-fixture", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reserve", default="")
    ap.add_argument("--stage", default="all", choices=["all", "teacher"])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--ctx-cache", default=None)
    ap.add_argument("--skip-recheck", action="store_true", help="skip recheck.py rules (fixture tests only)")
    ap.add_argument("--fail-on-recheck", action="store_true", help="fail instead of dropping records that fail recheck.py")
    a = ap.parse_args(argv)

    if a.ctx_cache:
        os.environ["MURMUR_CTX_CACHE"] = str(Path(a.ctx_cache).resolve())

    hebrew_dir = Path(a.hebrew_out)
    if not a.merge_only:
        gen_args = argparse.Namespace(
            out=str(hebrew_dir), n=a.n, dev=a.dev, test=a.test, seed=a.seed, lang_profile="hebrew",
            teacher=a.teacher, mined=a.mined, batch=a.batch, overgen=a.overgen,
            concurrency=a.concurrency if a.concurrency is not None else (TC.TeacherConfig().concurrency if a.teacher != "stub" else 4),
            max_requests=a.max_requests, max_retries=a.max_retries, shard_size=a.shard_size,
            limit=a.limit, topup=a.topup, screens=a.screens, allow_fixture=a.allow_fixture,
            workers=a.workers, reserve=a.reserve, stage=a.stage, offline=a.offline,
            sample=a.sample, ctx_cache=a.ctx_cache,
        )
        st = G.run(gen_args)
        if a.stage == "teacher":
            print(json.dumps(st, ensure_ascii=False, indent=1))
            return 0

    base_dir = None if a.hebrew_only else Path(a.base_dir)
    he_dirs = [hebrew_dir] + [Path(p) for p in a.extra_hebrew]
    merged = merge_datasets(
        base_dir=base_dir,
        hebrew_dir=he_dirs if len(he_dirs) > 1 else hebrew_dir,
        out_dir=Path(a.out),
        check_render=True,
        run_recheck=not a.skip_recheck,
        fail_on_recheck=a.fail_on_recheck,
    )
    print(json.dumps({k: merged[k] for k in ("final_total", "final_by_split", "final_by_lang", "dropped", "leakage")},
                     ensure_ascii=False, indent=1))
    print(f"merged dataset written to: {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
