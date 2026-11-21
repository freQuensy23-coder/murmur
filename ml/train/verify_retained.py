from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def verify_gguf(root: Path, run: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", run):
        raise ValueError(f"invalid finalist: {run!r}")
    gguf = root / "runs" / run / "gguf"
    manifest = json.loads((gguf / "manifest.json").read_text())
    entries = [f for f in manifest["files"] if f["quant"] == "Q4_0" and not f.get("deleted")]
    if len(entries) != 1:
        raise ValueError(f"{run}: expected one Q4_0 entry in manifest")
    entry = entries[0]
    name = entry["path"]
    if Path(name).name != name or not name.endswith("-Q4_0.gguf"):
        raise ValueError(f"{run}: invalid Q4_0 manifest filename")
    path = gguf / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular finalist GGUF: {path}")
    with path.open("rb") as f:
        if path.stat().st_size != entry["bytes"] or hashlib.file_digest(f, "sha256").hexdigest() != entry["sha256"]:
            raise ValueError(f"finalist GGUF does not match manifest: {path}")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)
    try:
        for run in args.runs:
            print(verify_gguf(args.root, run))
    except (OSError, ValueError, KeyError, TypeError) as e:
        parser.exit(2, f"retention check failed: {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
