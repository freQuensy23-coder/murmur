"""Merge girgur_v2 (sim) + audio_v3 (real Multi-TTS -> Audio Degradation -> Multi-ASR) into girgur_v3.

Enforces zero train/dev/test pair leakage and deterministically shuffles each split.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def _read(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="girgur_v2 dir")
    ap.add_argument("--extra", required=True, help="audio_v3 dir")
    ap.add_argument("--out", required=True, help="girgur_v3 output dir")
    ap.add_argument("--seed", type=int, default=20261125)
    args = ap.parse_args()

    base_dir, extra_dir, out_dir = Path(args.base), Path(args.extra), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load held-out splits first so train can never contain any held-out (transcript, target) or target!
    held_pairs: set[tuple[str, str]] = set()
    held_targets: set[str] = set()
    out_rows: dict[str, list[dict]] = {"dev": [], "test": [], "train": []}

    for sp in ("dev", "test"):
        seen: set[tuple[str, str]] = set()
        for src_dir in (base_dir, extra_dir):
            for r in _read(src_dir / f"{sp}.jsonl"):
                k = (_norm(r["transcript"]), _norm(r["target"]))
                if k in seen or k in held_pairs:
                    continue
                seen.add(k)
                held_pairs.add(k)
                held_targets.add(_norm(r["target"]))
                r["split"] = sp
                out_rows[sp].append(r)

    seen_train: set[tuple[str, str]] = set()
    dropped_leak = 0
    for src_dir in (base_dir, extra_dir):
        for r in _read(src_dir / "train.jsonl"):
            k = (_norm(r["transcript"]), _norm(r["target"]))
            if k in seen_train:
                continue
            if k in held_pairs or (len(r["target"]) >= 25 and _norm(r["target"]) in held_targets):
                dropped_leak += 1
                continue
            seen_train.add(k)
            r["split"] = "train"
            out_rows["train"].append(r)

    rng = random.Random(args.seed)
    stats = {"dropped_leak": dropped_leak, "splits": {}}
    for sp in ("train", "dev", "test"):
        rows = out_rows[sp]
        rng.shuffle(rows)
        (out_dir / f"{sp}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
        )
        stats["splits"][sp] = {
            "count": len(rows),
            "by_lang": dict(Counter(r["lang"] for r in rows)),
            "by_source": dict(Counter(r.get("source", "sim") for r in rows)),
            "by_cat": dict(Counter(r.get("category", "plain") for r in rows)),
            "unique_apps": len({(r.get("ctx") or {}).get("app", {}).get("package", "") for r in rows}),
        }

    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
