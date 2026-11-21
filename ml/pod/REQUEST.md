# GPU pod request (for the owner's Hermes agent)

Binding plan: `docs/ml/PLAN_V2.md`. Sequence, run matrix and stop rules: `docs/ml/TRAINING_RUNBOOK.md`
(section 1 = the turnkey command list). Legend: **[M]** measured (source given), **[V]** checked against a
primary source on the date given, **[A]** assumed, re-measured from `/workspace/logs/timings.jsonl` and
`runs/*/train_info.json` during the session.

## The request (copy as is)

> Please start **one RunPod pod named `alexm-training`** with **1x RTX 5090 (32 GB)**, community cloud
> (about 0.69 USD/h), for Murmur model training. **One session**, target 2-3 GPU-hours, **hard cap 4
> GPU-hours** (owner-approved). Our scripts finish all GPU work by 3 h 45 min at the latest, keeping the last
> 15 minutes for pulling the results and stopping; please stop the pod when we say the results are pulled, or at
> 4 h at the latest. **Stop, do not terminate**, until we confirm the
> results are on the dev server; then it may be terminated.
>
> * GPU: 1x RTX 5090 32 GB. Use a host whose `nvidia-smi` advertises **CUDA 13.2 or newer** (CUDA 13.2 GA:
>   Linux driver **>= 595.45.04**). The Linux PyPI torch 2.15.0 stack uses CUDA 13.2. CUDA 13.0 and minor-version
>   compatibility alone are not a sufficient preflight for JIT-compiled kernels.
> * Image: **`runpod/pytorch:2.1.0-cu1300-torch2130-ubuntu2404`** (CUDA 13.0, Ubuntu 24.04; tag listed on
>   Docker Hub 2026-11-20). Any current Ubuntu 22.04/24.04 CUDA 13 image works: we install our own Python
>   environment onto the volume, so the image's torch does not matter.
> * Container disk: **30 GB** (apt packages, Triton/compile caches, temp files).
> * Volume: **80 GB at `/workspace`** (code, data, Python env, base weights, runs, results).
> * Ports: **SSH only** (TCP 22, exposed). No HTTP ports; nothing on the pod listens publicly.
> * SSH: install the dedicated public key supplied by the operator and return `user@host` and the SSH port.
> * Secrets: **none**. The base models are public. Pull results to the private dev-server directory and keep
>   them there; no dataset or fine-tune uploads. No HF token, teacher or LLM API keys on the pod.
> * Please report the actual GPU-hours and cost at the stop.

## Sizing (disk from the real model sizes)

| Item | Size | Source |
|---|---:|---|
| Base weights in `$HF_HOME`: `LiquidAI/LFM3-1.7B` 3.46 GB + `Qwen/Qwen3.5-0.8B` 1.75 GB + `Qwen/Qwen3.5-2B` 4.55 GB | 9.8 GB | [M] HF API `?blobs=true`, 2026-11-18 |
| Training venv (torch 2.15.0 CUDA 13.2 + nvidia libs, transformers/peft, fla, liger) | ~8 GB | [A] CPU venv 1.1 GB [M]; the CUDA wheels dominate |
| llama.cpp b12503 source + CPU build + CUDA 12.8 build + cudart | ~1.5 GB | [A] CPU build 58 MB, source 196 MB [M] |
| Training data (train/dev/test + stats) | 12 MB | [M] `.cache/ml/data/final` |
| Per LFM3 LoRA run: adapter, BF16 GGUF 3.4 GB, Q4_0 1.05 GB (merged HF copy 3.4 GB exists only between merge and parity) | ~4.6 GB | [A] from the bf16 weight size; Q4_0 = vendor size [M] |
| Per Qwen3.5-2B / -0.8B LoRA run (text-only GGUF) | ~5.2 / ~2.1 GB | [A] same rule |
| Matrix: 4 LFM3 + DoRA + Q0.8B + Q2B, plus the full-FT fallback (bf16 final + GGUFs ~8 GB) | ~38 GB worst case | [A] |
| Transient (one merged HF copy, one checkpoint with optimizer state) | ~5 GB | [A] |
| **Total worst case** | **~62 GB** -> volume **80 GB** | |

## Time budget (one session; target <= 3 h, hard cap 4 h)

Measured here (dev server CPU, tiny random-weight models on the real final data, `.cache/ml/train/podsim-real/`)
and in earlier CPU dry runs (`docs/ml/TRAINING_RUNBOOK.md` section 4):

* data (current `.cache/ml/data/final`): 3,000 train / 300 dev / 300 test rows; rendered with the real tokenizers
  (finetuned.v2 + 5 % default.v1 = 141 rows + 15 % preferences): mean **373 (Qwen3.5) / 388 (LFM3) tokens**,
  max 1,463 / 1,504, max_len auto 1,472 / 1,536, **0 dropped** [M `sft.py --prepare-only`, runbook 2.1]; with
  length-grouped batches of 16 an epoch is **1.18M (Qwen3.5) / 1.23M (LFM3) padded tokens** for 1.12M / 1.16M real
  ones (padding efficiency 0.95; random order would be 0.41, fixed max_len 0.25) [M]; 188 optimizer steps per
  epoch, 564 for 3 epochs.
* render + tokenise the whole set: ~10 s per run [M]; merge 13 s, GGUF BF16 + Q4_0 23 s on Qwen3.5-0.8B real
  weights on CPU [M] (the GPU pod is not faster at these: disk-bound).

5090 throughput: 209.5 dense BF16 tensor TFLOP/s with FP32 accumulate [A: NVIDIA RTX Blackwell whitepaper figure
as quoted by secondary sources; the PDF could not be parsed here]. LoRA with gradient checkpointing costs ~6N FLOP
per token (forward 2N + recompute 2N + activation gradients 2N; frozen weights need no weight gradients; the
lm_head runs only on the ~8 % completion tokens): LFM3-1.7B, N ~1.45B without the embedding -> 8.7 GFLOP/token.
**Plan at 20 % MFU = ~4.8K padded tokens/s** [A] (worst case planned: 10 % = 2.4K tokens/s). The anchor run
measures it (`train_info.json` `padded_tokens_per_s`) and every later run is projected from the anchor's
measured wall time.

| Stage | Minutes plan (20 % MFU) | worst (10 %) | Basis |
|---|---:|---:|---|
| pod start, SSH, `push_code.sh WITH_DATA=1` | 5 | 5 | [A]; push of code + 12 MB data < 1 min [M locally] |
| `bootstrap.sh`: uv + Python, llama.cpp (CPU + CUDA builds), venv (~3.5 GB wheels), fla kernel test, self-test, llama.cpp GPU kernel smoke test (2 models to Q4_0 + decode on GPU and CPU), 9.8 GB weights | 14 | 17 | [A]; self-test 6-12 s, smoke GGUF build 16.5 s + decode 2.5 s per model on the dev CPU [M] |
| anchor `lfm3-r16`: train 3 epochs 3.68M padded tokens = 13 (26) + dev loss evals 1 + merge/export 1 + parity 1 + Q4_0 dev eval (422 items, CUDA llama.cpp) 4 | 21 | 34 | [A] tokens [M] |
| phase A extras `lfm3-r64` (1.1x), `lfm3-r16-lr5e-4` (1.0x), `lfm3-r16-e5` (1.67x): **optional**, each only if it fits 3 h with all mandatory work still ahead | 44 (r64 + lr5e-4; e5 does not fit) | 0 (none fits) | decided on the pod |
| phase B `q08-lora` (0.6x) + `q2-lora` (1.3x) with the phase-A winner's r/lr/epochs | 40 | 65 | [A] cost factors `train_matrix.sh` |
| finalists (one per base): BF16 dev eval, Q4_0 test eval, Q4_0 dev eval with `default.v1`, cards, leaderboard, `vs_stock.md` | 30 | 30 | [A] ~0.3-0.5 s per item on CUDA llama.cpp |
| `sync_results.sh` incl. 3 finalist Q4_0 GGUFs (~2.8 GB), stop: inside the 15 min the scripts keep back (`BUDGET_TAIL_S`) | 5 | 5 | [A] |
| **Total** | **159 min (2.65 h) = ~1.83 USD** | **156 min without extras** | |
| optional `FULL=1` full fine-tuning fallback (only if LoRA misses the gates; same rule as the extras) | +53 | +85 | [A] 2.5x anchor |

The mandatory part (pod start, bootstrap, anchor, phase B, finalist evals, sync and stop) is **~115 min** at plan
speed and ~156 min at half of it; the budget rule decides the optional runs from the measured anchor, so the
session lands between 2 h and 3 h. Both limits are applied 15 min early (`BUDGET_TAIL_S=900`): optional runs must
end with all mandatory work by 2 h 45, and nothing at all starts that would end past 3 h 45, which leaves one poll
interval (~5 min), the `WITH_GGUF=1` sync (~3 GB, ~3 min) and the stop inside 4 h. At plan speed the decision is
unchanged by the tail: after the anchor (~40 min billed) `lfm3-r64` needs 23 + 70 mandatory = 133 <= 165,
`lfm3-r16-lr5e-4` then 63 + 91 = 154 <= 165, `lfm3-r16-e5` then 84 + 105 = 189 > 165 (skipped).

## Stop rules (all enforced by the scripts, `ml/pod/env.sh`, `train_matrix.sh`, `run_one.sh`, `ml/train/sft.py`)

1. **Hard cap 4 h** (`MAX_GPU_HOURS=4`): billed time = now - container start (PID 1 age, so setup and idle count)
   + 5 min provisioning per session, summed over sessions on the volume (`/workspace/.budget/sessions.tsv`).
   Checked before every stage, run, step (train, merge/export, parity, eval) and finalist eval: nothing starts
   that would end past the cap minus the 15 min tail (`BUDGET_TAIL_S=900`, rc 5), so the final sync and the stop
   always fit before 4 h (`ml/train/tests/test_pod_budget.py`).
2. **Inside training**: `run_one.sh` exports `MURMUR_TRAIN_DEADLINE` = the moment the rest of the run (median
   merge + export + parity + dev eval) still fits the cap; `sft.py` stops after the current optimizer step,
   saves, and the run is exported and evaluated as "stopped_by: budget" (logged loudly). A NaN/inf training loss
   (checked on the raw loss of every micro-batch, not on the filtered log) stops the run at the end of that
   optimizer step without saving the diverged weights (`stopped_by: diverged`, exit 4, run marked failed, the
   matrix continues); proven through the real Trainer on CPU (runbook section 4). It leaves
   `runs/RUN/DIVERGED.json` (step, reason): a re-run of `train_matrix.sh` / `run_all.sh` skips that run (logged,
   counted as failed, never selected) instead of resuming into the same divergence; `FORCE_RETRAIN=1` trains it
   again from scratch.
3. **Target 3 h** (`TARGET_GPU_HOURS=3`): optional runs (phase A extras, DoRA, full FT) start only if their
   projection + all mandatory work still ahead fits.
4. **Preflight** (before any training): abort if flash-linear-attention failed its sm_120 kernel test while Qwen3.5
   runs are planned, or if no llama.cpp CUDA build passed bootstrap's GPU kernel smoke test (real Q4_0 decode on
   the GPU = CPU result; fallbacks CUDA 13.5 release build, then a source build for sm_120, runbook 3.2) (rc 6;
   overrides `ALLOW_SLOW_QWEN=1`, `ALLOW_CPU_EVAL=1` after a conscious decision). If the smoke MODEL cannot be
   downloaded or converted (retried once), that is reported as its own error: bootstrap exits 7 without testing
   or fetching any build, and the preflight refuses (rc 6, override `ALLOW_UNTESTED_LLAMA=1`) until a
   re-run of bootstrap prepares it.
5. **Pinned weights**: the three bases are downloaded and loaded at the commits recorded in the runbook (0.1,
   `ml/train/configs/base.yaml` `base_revisions`); a vendor push before the session changes nothing.
6. **Parity**: a run whose BF16 GGUF does not reproduce the merged HF model (< 80 % of 24 dev items agree) is not
   evaluated (rc 3); the matrix continues.
7. A crash or a pod stop loses nothing finished: every step skips existing outputs and training resumes from its
   newest checkpoint (`save_steps` 100), so the same command continues (a diverged run excepted, rule 2).

Hermes' billing number stays the authoritative one.
