#!/usr/bin/env python3
"""Merge a LoRA/DoRA adapter into its base model and save a bf16 HF checkpoint for GGUF export.

  python -m ml.train.merge --run OUTPUT_DIR            # reads OUTPUT_DIR/final/adapter
  python -m ml.train.merge --base Qwen/Qwen3.5-0.8B --adapter DIR --out DIR   # revision: base.yaml base_revisions

The base is loaded at the Hub commit the run trained on (config.resolved.json: base_revision /
base_revisions, ml.train.config.base_revision); a local base directory needs none.

The merge is done in fp32 and cast to bf16 once (avoids double rounding). Tokenizer files and the
official chat template are copied from the run's final/ directory. For a full fine-tune there is
nothing to merge: final/ already is the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from ml.train import config as cfglib


def merge(base: str, adapter: str, out: str, tok_dir: str | None = None, revision: str | None = None) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    outp = Path(out)
    if outp.exists():
        shutil.rmtree(outp)
    model.to(torch.bfloat16).save_pretrained(outp, safe_serialization=True)
    src = Path(tok_dir) if tok_dir else Path(adapter).parent
    AutoTokenizer.from_pretrained(src).save_pretrained(outp)
    for name in ("chat_template.jinja", "LICENSE", "NOTICE"):
        if (src / name).exists():
            shutil.copy2(src / name, outp / name)
    return outp


def resolve_run(run: str) -> tuple[str, str, str, str | None]:
    cfg = json.loads((Path(run) / "config.resolved.json").read_text())
    return cfg["base_model"], str(Path(run) / "final" / "adapter"), cfg["method"], cfglib.base_revision(cfg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--base")
    ap.add_argument("--adapter")
    ap.add_argument("--out")
    ap.add_argument("--revision", help="Hub commit of --base (default: the base.yaml base_revisions entry)")
    a = ap.parse_args(argv)
    if a.run:
        base, adapter, method, revision = resolve_run(a.run)
        if method == "full":
            print(f"full fine-tune: nothing to merge, use {a.run}/final")
            return 0
        out = a.out or str(Path(a.run) / "merged")
        p = merge(base, adapter, out, tok_dir=str(Path(a.run) / "final"), revision=revision)
    else:
        if not (a.base and a.adapter and a.out):
            ap.error("--base, --adapter and --out are required without --run")
        rev = cfglib.base_revision({"base_revision": a.revision, "base_revisions": cfglib.pinned_revisions()}, a.base)
        p = merge(a.base, a.adapter, a.out, revision=rev)
    print(f"merged -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
