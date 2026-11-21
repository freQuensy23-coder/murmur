#!/usr/bin/env python3
"""Upload an exported fine-tune (GGUF + README.md + LICENSE + catalog_entry.json) to the owner's
Hugging Face account. Token from env HF_TOKEN (never from a file in the repo). Private by default.

  HF_TOKEN=... python -m ml.train.upload_hf --run OUTPUT_DIR --repo OWNER/murmur-cleanup-qwen3.5-0.8b-gguf \
      [--files murmur-...-Q4_0.gguf] [--allow-private-upload | --dry-run]

After the upload the commit sha is written into catalog_entry.json (download_files[].repo_sha),
and the remote file's sha256 is checked against the local manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--files", nargs="*", help="GGUF names to upload (default: every non-deleted manifest file "
                                               "except BF16)")
    ap.add_argument("--allow-private-upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.dry_run and not a.allow_private_upload:
        ap.error("upload is disabled without --allow-private-upload; use --dry-run for a local plan")
    g = Path(a.run) / "gguf"
    man = json.loads((g / "manifest.json").read_text())
    files = a.files or [f["path"] for f in man["files"] if not f.get("deleted") and f["quant"] not in ("BF16", "F16")]
    base_lic = Path(man["source"]) / "LICENSE"
    if base_lic.exists() and not (g / "LICENSE").exists():
        shutil.copy2(base_lic, g / "LICENSE")  # LFM Open License 4(a): recipients get a copy of the licence
    extra = [p for p in ("README.md", "LICENSE", "catalog_entry.json") if (g / p).exists()]
    plan = {"repo": a.repo, "private": True, "files": files + extra}
    print(json.dumps(plan, indent=1))
    if a.dry_run:
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
    if api.model_info(a.repo).private is not True:
        raise SystemExit("refusing to upload: destination is not private")
    info = None
    for p in files + extra:
        info = api.upload_file(path_or_fileobj=str(g / p), path_in_repo=p, repo_id=a.repo,
                               commit_message=f"Murmur export: {p}")
    sha = getattr(info, "oid", None) or api.model_info(a.repo).sha
    remote = {s.rfilename: s for s in api.model_info(a.repo, files_metadata=True).siblings}
    for f in man["files"]:
        if f["path"] in files:
            lfs = getattr(remote.get(f["path"]), "lfs", None)
            rsha = (lfs or {}).get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            if rsha and rsha != f["sha256"]:
                raise SystemExit(f"sha256 mismatch for {f['path']}: local {f['sha256']} remote {rsha}")
    cat = g / "catalog_entry.json"
    if cat.exists():
        entry = json.loads(cat.read_text())
        for d in entry.get("download_files", []):
            d["repo"], d["repo_sha"] = a.repo, sha[:12] if sha else None
        cat.write_text(json.dumps(entry, indent=1, ensure_ascii=False))
        api.upload_file(path_or_fileobj=str(cat), path_in_repo="catalog_entry.json", repo_id=a.repo,
                        commit_message="catalog entry with revision")
    print(f"uploaded to https://huggingface.co/{a.repo} at {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
