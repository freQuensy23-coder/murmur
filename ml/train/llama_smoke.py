#!/usr/bin/env python3
"""llama.cpp GPU kernel smoke test (pod bootstrap): does the CUDA build really COMPUTE on this GPU?

`llama-server --list-devices` only proves the CUDA runtime sees the card; it runs no kernel. A build without
device code for the GPU (sm_120 on the RTX 5090), a failing or very slow PTX JIT, or a miscomputing kernel
would otherwise show up only at the first run's parity or eval, after its training time was spent. This
converts a real base model to Q4_0 once (the shipping quant, through ml.train.export; cached on the
volume), then greedily decodes a few fixed, strongly determined prompts (counting, weekdays, letters) on
the GPU build with every layer offloaded and on the CPU build, and compares the texts.

  python -m ml.train.llama_smoke --hf Qwen/Qwen3.5-0.8B --work $W/llama/smoke --gpu-bin DIR --cpu-bin DIR \
      --require-gpu --json $W/llama/smoke/result.json

PASS (exit 0) when the GPU server starts, offloads all layers (--require-gpu: `--device CUDA0` and the
offload line in its log), answers within --timeout seconds, and every prompt agrees with the CPU build
(identical, or a common prefix of at least --min-prefix characters: GPU and CPU kernels round differently,
a broken kernel diverges at the first token). FAIL = exit 1, with the reason in the JSON: the build under test
is bad. --hf takes a Hub repo id (loaded at its pinned base.yaml revision) or a local HF directory.

Preparing the smoke model (download + conversion to Q4_0 with the CPU build) is kept apart from testing a build:
if it fails, the status is PREP_FAIL and the exit code 3, because no build was tested and trying other builds
would only fail the same way. `--prepare-only` does just that step (bootstrap runs it, retried once, before it
tests any build; ml/pod/kernel_smoke.sh):

  python -m ml.train.llama_smoke --prepare-only --hf Qwen/Qwen3.5-0.8B --work $W/llama/smoke --cpu-bin DIR --json F
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROMPTS = ["1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,",
           "Monday, Tuesday, Wednesday, Thursday,",
           "A, B, C, D, E, F, G, H, I, J,",
           "январь, февраль, март, апрель,"]
OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare(cpu: list[str], gpu: list[str], min_prefix: int) -> list[dict]:
    rows = []
    for p, c, g in zip(PROMPTS, cpu, gpu):
        k = common_prefix(c, g)
        ok = bool(c.strip()) and (c == g or k >= min(min_prefix, max(len(c), len(g))))
        rows.append({"prompt": p, "cpu": c, "gpu": g, "prefix_chars": k, "agree": ok})
    return rows


def build_gguf(hf: str, work: Path, cpu_bin: str) -> Path:
    """Q4_0 GGUF of `hf` in `work` (reused when present)."""
    name = "llama-smoke-" + re.sub(r"[^a-z0-9.]+", "-", Path(hf).name.lower())
    out = work / name
    gguf = out / f"{name}-Q4_0.gguf"
    if gguf.exists():
        return gguf
    src = Path(hf)
    if not src.is_dir():
        from huggingface_hub import snapshot_download
        from ml.train import config as cfglib
        rev = cfglib.base_revision({"base_revisions": cfglib.pinned_revisions()}, hf)
        src = Path(snapshot_download(hf, revision=rev))
    env = {**os.environ, "MURMUR_LLAMA_BIN": cpu_bin}
    r = subprocess.run([sys.executable, "-m", "ml.train.export", "--hf", str(src), "--name", name, "--out", str(out),
                        "--quants", "Q4_0"], env=env, capture_output=True, text=True)
    if r.returncode or not gguf.exists():
        raise SystemExit(f"GGUF build failed: {r.stdout[-2000:]}{r.stderr[-2000:]}")
    return gguf


def decode(gguf: Path, bin_dir: str, args: list[str], log: Path, n: int, timeout: float, threads: int) -> dict:
    from ml.eval.backends import LlamaServer, _post
    t0 = time.time()
    srv = LlamaServer(str(gguf), bin_dir, threads=threads, ctx=512, extra_args=args, log_path=str(log),
                      startup_timeout=timeout)
    try:
        texts = []
        for p in PROMPTS:
            r = _post(srv.url + "/completion", {"prompt": p, "n_predict": n, "temperature": 0.0, "top_k": 1,
                                                "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0, "seed": 0,
                                                "cache_prompt": False, "stream": False},
                      timeout=max(5.0, timeout - (time.time() - t0)))
            texts.append(r.get("content", ""))
        return {"texts": texts, "load_s": round(srv.load_s, 1), "wall_s": round(time.time() - t0, 1)}
    finally:
        srv.close()


EXIT = {"PASS": 0, "PREPARED": 0, "FAIL": 1, "PREP_FAIL": 3}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", help="Hub repo id (pinned revision) or local HF dir; converted to Q4_0 once")
    ap.add_argument("--gguf", help="use this GGUF instead of --hf")
    ap.add_argument("--work", default=".cache/ml/train/llama-smoke")
    ap.add_argument("--gpu-bin", help="dir with the llama-server build under test (not with --prepare-only)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="only build the smoke GGUF from --hf (PREPARED, exit 0 / PREP_FAIL, exit 3)")
    ap.add_argument("--label", default="", help="name of the build under test, in its log file name (one log each)")
    ap.add_argument("--cpu-bin", required=True, help="dir with the reference CPU llama-server (+ llama-quantize)")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--require-gpu", action="store_true", help="--device CUDA0 and every layer offloaded")
    ap.add_argument("-n", type=int, default=16, help="tokens per prompt")
    ap.add_argument("--timeout", type=float, default=120, help="max seconds for the GPU server: start + all prompts")
    ap.add_argument("--min-prefix", type=int, default=16)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("MURMUR_THREADS", "6")))
    ap.add_argument("--json", help="write the result here")
    a = ap.parse_args(argv)
    if not a.prepare_only and not a.gpu_bin:
        ap.error("--gpu-bin is required (unless --prepare-only)")
    if a.prepare_only and not a.hf:
        ap.error("--prepare-only needs --hf")
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    res: dict = {"status": "FAIL", "gpu_bin": a.gpu_bin, "cpu_bin": a.cpu_bin}

    def done(status: str, why: str = "") -> int:
        res.update(status=status, reason=why)
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(json.dumps(res, ensure_ascii=False))
        return EXIT[status]

    t0 = time.time()
    try:   # any failure here is the smoke MODEL's (download, conversion), not the build's under test
        gguf = Path(a.gguf) if a.gguf else build_gguf(a.hf, work, a.cpu_bin)
        if not gguf.exists():
            raise FileNotFoundError(gguf)
    except (SystemExit, Exception) as e:
        return done("PREP_FAIL", f"could not prepare the smoke GGUF (not a GPU build problem): "
                                 f"{type(e).__name__}: {e}")
    res.update(gguf=str(gguf), gguf_build_s=round(time.time() - t0, 1))
    if a.prepare_only:
        return done("PREPARED")
    tag = gguf.stem
    gpu_args = ["-ngl", str(a.ngl)] + (["--device", "CUDA0", "-lv", "5"] if a.require_gpu else [])
    glog = work / f"{tag}{'.' + a.label if a.label else ''}.gpu.log"
    try:   # GPU first: a missing kernel image / JIT failure aborts the server or the request
        gpu = decode(gguf, a.gpu_bin, gpu_args, glog, a.n, a.timeout, a.threads)
    except Exception as e:
        tail = glog.read_text(errors="replace")[-1500:] if glog.exists() else ""
        return done("FAIL", f"GPU build did not decode: {type(e).__name__}: {e}; log tail: {tail}")
    res["gpu"] = {k: v for k, v in gpu.items() if k != "texts"}
    if gpu["wall_s"] > a.timeout:
        return done("FAIL", f"GPU build took {gpu['wall_s']} s > {a.timeout} s (slow PTX JIT?)")
    if a.require_gpu:
        m = OFFLOAD.findall(glog.read_text(errors="replace"))
        res["gpu"]["offloaded"] = m[-1] if m else None
        if not m or m[-1][0] != m[-1][1]:
            return done("FAIL", f"not every layer offloaded to the GPU: {m[-1] if m else 'no offload line'}")
    try:
        cpu = decode(gguf, a.cpu_bin, ["-ngl", "0"], work / f"{tag}.cpu.log", a.n, 600, a.threads)
    except Exception as e:
        return done("FAIL", f"CPU reference did not decode: {type(e).__name__}: {e}")
    res["cpu"] = {k: v for k, v in cpu.items() if k != "texts"}
    res["prompts"] = compare(cpu["texts"], gpu["texts"], a.min_prefix)
    bad = [r for r in res["prompts"] if not r["agree"]]
    if bad:
        return done("FAIL", f"{len(bad)} of {len(PROMPTS)} prompts differ between the GPU and the CPU build")
    return done("PASS")


if __name__ == "__main__":
    raise SystemExit(main())
