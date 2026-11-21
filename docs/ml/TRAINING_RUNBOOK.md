# Murmur — training, export and evaluation runbook (PLAN_V2)

Date: 2026-11-18, review fixes 2026-11-19, turnkey pod plan on the final data 2026-11-20, second review fixes
2026-11-20 (divergence stop, budget tail, llama.cpp kernel smoke test, pinned base revisions; numbers from the current final),
final review minors 2026-11-20 (DIVERGED.json marker, smoke-model preparation separated from the build test). Binding plan: `docs/ml/PLAN_V2.md` (1–5K synthetic examples, teacher on the dev
server, one RTX 5090 for 2–3 GPU-hours, hard cap 4 h, LoRA first). Code: `ml/train`, `ml/eval`, `ml/pod` (overview in
`ml/README.md`). GPU request: `ml/pod/REQUEST.md`.
Legend: **[V]** verified here (command, file or source given), **[I]** design choice, **[A]** assumed.

## 0. Decisions

1. **Bases** (HF API, 2026-11-18) [V]:

   | Model | Train from | Size (safetensors) | Licence for redistributing fine-tunes | Template, thinking off |
   |---|---|---:|---|---|
   | LFM3-1.7B | `LiquidAI/LFM3-1.7B` (rev `1da06bec6a81`) | 3.46 GB | LFM Open License v1.0 (`research/lfm3_LICENSE`): Apache-2.0-style grant, **commercial use licensed only while the user's legal entity has annual revenue < US$10M** (s.1 "Threshold", s.5); redistributing a derivative requires a copy of the licence (4a) and notices on modified files (4b) | `<\|startoftext\|><\|im_start\|>…<\|im_start\|>assistant\n<think></think>`; reasoning is **on by default**, off with `enable_thinking=False` (card: "can be inconsistent", so we train with it off) |
   | Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` (rev `2fc06364715b`) | 1.75 GB | Apache-2.0: ship LICENSE, mark changes | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n`; off unless `enable_thinking=true` |
   | Qwen3.5-2B | `Qwen/Qwen3.5-2B` (rev `15852e8c1636`) | 4.55 GB | Apache-2.0 | identical `chat_template.jinja` to 0.8B [V cmp] |

   These revisions are **pinned**: `ml/train/configs/base.yaml` `base_revisions` holds the full commit shas
   (`1da06bec6a819da10b54c67e5c90972b46d81871`, `2fc06364715b967f1860aea9cf38778875588b17`,
   `15852e8c16360a2fea060d615a32b45270f8a8fc`), and `sft.py` (tokenizer, config, weights, template files),
   `merge.py`, `bootstrap.sh` (`snapshot_download`) and `export_eval.sh` (`STOCK_OURS`) load exactly them; a Hub base
   without an entry is refused, a local directory needs none (`ml/train/tests/test_base_revision.py`). Checked
   upstream 2026-11-20 with the public HF API (`/api/models/<repo>/revision/<sha>` resolves, `config.json` at the
   sha returns 200; all three are still the current `main`) [V].

   Qwen3.5 checkpoints are `Qwen3_5ForConditionalGeneration` (vision encoder + MTP head); `AutoModelForCausalLM`
   loads the text model only (752M params for 0.8B) [V]. LFM3: `Lfm3ForCausalLM`, hybrid short-conv +
   attention (GQA, per-head gate `g_proj`, sliding window on some layers) [V GGUF metadata, module names].
2. **Stack** [V installed + run on CPU]: torch 2.15.0, transformers 5.23.0 (first release with native `lfm3`;
   absent in 5.22.0, so Liquid's cookbook uses 5.20.2 + `trust_remote_code`), peft 0.21.3, accelerate 1.16.0;
   GPU extras flash-linear-attention 0.6.0 (Qwen3.5 Gated DeltaNet kernels) and liger-kernel 0.8.5.
   **torch 2.15.0 uses CUDA 13.2 on Linux PyPI**. Use an x86-64 host whose `nvidia-smi` advertises CUDA 13.2
   or later (13.2 GA: Linux driver >= 595.45.04). CUDA 13.x minor-version compatibility on R580 does not
   guarantee support for JIT-compiled kernels. The image's bundled CUDA/torch is separate. **No TRL, no Unsloth**: we render and mask ourselves (TRL
   would only re-implement that); Unsloth has no LFM3 module [V repo tree] and buys little at 0.8–2B on a 5090.
   Liquid's official LoRA recipe (cookbook `lfm3_sft_with_trl.ipynb`: r16, α32, dropout 0.05, q/k/v/o +
   in_proj/out_proj + w1/w2/w3, lr 2e-4, effective batch 16, cosine, warmup 5 %) [V notebook] is our anchor
   run `lfm3-r16` (we also adapt `g_proj`, the "optional" per-head gate in the notebook).
3. **Tokenizer quirks** [V `ml/train/tests`]: LFM3's BOS comes from the template text, the tokenizer adds none
   (GGUF `add_bos_token=false`), so tokenise with `add_special_tokens=False`; both families end a turn with
   `<|im_end|>` (Qwen 248046, LFM3 124900), but Qwen3.5's `config.json` has `eos_token_id` 248044
   (`<|endoftext|>`): training appends the `<|im_end|>` id and HF `generate` is given it explicitly.
   Training strings equal the vendors' `apply_chat_template(..., enable_thinking=False)` byte for byte, and
   `ml/eval/contract.render_prompt` equals `prompts/render.py render_prompt` for every profile, few-shot
   profiles included (tests).
4. **Export** [V on CPU, section 4]: PEFT merge (fp32, cast to bf16 once) → llama.cpp **b12503**
   `convert_hf_to_gguf.py --outtype bf16` (Qwen3.5 with `--no-mtp`, text-only) → `llama-quantize Q4_0` →
   eval of the quantised file. Use gguf-py from the same llama.cpp checkout (PyPI `gguf` 0.19.0 predates both
   architectures). LFM3 GGUFs need llama.cpp >= b11960 (vendor card); the app's runtime must be >= the
   conversion build (recorded in the manifest and catalog entry).
5. **QAD** [V vendor GGUF card]: `LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-QAD-Q4_0.gguf` is a
   Quantization-Aware-Distillation checkpoint "trained to recover the accuracy lost to Q4_0"; no float QAD
   checkpoint is published (33 LFM3 repos listed), so a fine-tune of `LiquidAI/LFM3-1.7B` **loses QAD** under
   plain post-training Q4_0. Handling, in order: measure (BF16 GGUF vs Q4_0 GGUF of the same fine-tune on the
   same set, and fine-tune Q4_0 vs stock QAD Q4_0); `--imatrix-data` calibration on our rendered prompts (wired);
   Q4_K_M for CPU/GPU builds; own QAT only if Q4_0 loses > 3 pp EM-noop or > 0.2 judge points
   (`ml.eval.compare` prints that trigger). The unsloth Qwen3.5 "Q4_0" files are imatrix mixed quants
   (Q8_0/Q5_K/Q4_1 tensors inside) [V GGUF metadata], so the baselines also include our own plain conversion
   of the stock weights (`STOCK_OURS=1`).
6. **LoRA first, full FT as one fallback** (PLAN_V2). The method comparison stays a first-class experiment:
   same data, same prompt mix, same eval, one config each (`lfm3-r16` vs `full-fallback`); `full-fallback`
   runs only with `FULL=1`, i.e. when LoRA misses the gates. "Misses the gates" is only meaningful because
   a perfect model passes them: an oracle (output := target) gives PASS or INCOMPLETE, never FAIL, on every
   eval set (`test_oracle_never_fails_the_gates`; before the 2026-11-19 fix it FAILed every set).
8. **Prompts** [I, 2026-11-19; mix revised 2026-11-20]: fine-tunes train and are evaluated with **`finetuned.v2`**
   (OCR wording; the ctx v2 data is what it was written for). The robustness mix renders **5 %** of examples with
   **`default.v1`**, the stock profile the app ships for every stock model since ADR-085 (PROMPT_EVAL R.5), i.e. the
   prompt a fine-tune most likely receives besides its own (a user who switches the prompt back, an app build
   that has not picked up the catalog's prompt). It replaces `default.v4s` (no longer the default, still selectable):
   the runbook's rule was "the stock profile a user can actually select", and that is now v1. Side effect: v1 has no
   shot turns (~1.25-1.3K tokens rendered vs ~2.2K), so the mean rendered length drops from 418 to 373 tokens
   (Qwen3.5) and max_len from 2,432 to 1,472 (Qwen3.5) / 2,496 to 1,536 (LFM3) on the final set [M, section 2.1]. The eval of a run always uses the
   `prompt.system` it trained with (read from its `config.resolved.json`), and the card / catalog entry record it,
   so the app must send `finetuned.v2` to these models (the Kotlin renderer needs that file). Stock baselines use
   what the app ships for stock models: profile `default.v1` + `prompts/guard.py`.
9. **Selection and ship sets are ctx v2 only** [I]: dev = synthetic dev split + `prompts/eval/sets/dev.v2.jsonl`
   + `smoke_cases.v2` (converted); test = synthetic test split + `test.v2.jsonl`. The v1 structured-context sets
   are a labelled legacy column (`LEGACY=1`), never used by `select.py`, the gates or the cards.
10. **Guard** [I]: every report holds the raw-output measurement (what `select.py` ranks) and the same
   predictions after `prompts/guard.py` (`guarded`: precheck skip -> empty, repair, fallback to the transcript),
   i.e. what the app inserts; leaderboard and cards show both, the stock-vs-fine-tune comparison is made raw vs
   raw and guarded vs guarded.
7. **Judge and teacher only on the dev server** (PLAN_V2: Responses API proxy, credentials never leave the
   server). The pod produces predictions; the dev server judges them (`ml.eval.judge`, 2 concurrent
   requests, Retry-After honoured) and recomputes the reports (`run_eval --rescore`).

## 1. Turnkey sequence (one pod session, 2026-11-20)

The round-trip probe already ran and the final data exists (`.cache/ml/data/final`: train 3,000 / dev 300 /
test 300, `docs/ml/DATA_CARD.md`), so the pod is used **once**, for training and evaluation only. Everything
else (stock baselines, optional judge, CPU ship gates, cards) runs on the dev server without a GPU. Every step
is idempotent: after an SSH drop, a crash or a stop, re-run the same command.

Before the pod (dev server, no GPU):

- `python -m ml.train.verify_inputs ASSEMBLED SOURCE_SPLIT_DIR --holdout HOLDOUT` checks split
  provenance and overlap without printing rows. Repeat `--holdout` to compare additional sets.
- `python -m ml.train.verify_models --model REPO LOCAL_DIRECTORY` compares cached model files to the
  configured public revisions. Weight shards not present locally are metadata-only evidence.
- `ml/train/rehearse_cpu.sh` is an offline two-step smoke using caller-supplied local inputs and a new
  private work directory. A passing CPU rehearsal does not qualify a GPU or model quality.

```
cd /path/to/murmur
make -C ml/train test                                   # 105 unit tests (ml/eval + ml/train), ~30-60 s
bash .cache/ml/train/podsim-real/simulate.sh            # optional: the whole sequence below on CPU with tiny
                                                        # models on the real data (section 4.2), ~25 min
```

Request the pod with `ml/pod/REQUEST.md` (name `alexm-training`, 1x RTX 5090, 80 GB volume, no secrets). Then:

```
export POD=root@<host> POD_PORT=<port>
export POD_KEY=/path/to/dedicated-ssh-key
S="ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=accept-new $POD"

# 1. code + data to the volume (rsync, < 1 min)
WITH_DATA=1 DATA_DIR=.cache/ml/data/final bash ml/pod/push_code.sh

# 2. bootstrap (host CUDA >= 13.2 check, uv + Python, llama.cpp CPU + CUDA, venv, fla sm_120 kernel test,
#    self-test, llama.cpp GPU kernel smoke test vs CPU (step 6b, section 3.2), 9.8 GB base weights at the pinned
#    revisions; ~14 min; re-run after an SSH drop). Check: grep 'kernel smoke' /workspace/logs/pod.log
#    must say PASS; otherwise section 3.2 (the matrix would refuse to start, rc 6). Exit 7 = the smoke MODEL could
#    not be downloaded/converted (retried once; not a CUDA problem, no build tested): fix, re-run bootstrap.
$S 'bash -o pipefail -c "umask 077; mkdir -p /workspace/logs && cd /workspace/murmur && bash ml/pod/bootstrap.sh 2>&1 | tee -a /workspace/logs/bootstrap.log | tail -20"'

# 3. the matrix + finalist evals, detached in tmux (~1.7-2.4 h; budget guard inside: the scripts end by
#    4 h - 15 min at the latest, BUDGET_TAIL_S, so steps 4 and 5 always fit inside the 4 h cap)
$S "tmux new -d -s murmur 'cd /workspace/murmur && STAGES=\"bootstrap train eval\" bash ml/pod/run_all.sh \
    >>/workspace/logs/run_all.log 2>&1; echo EXIT \$? >>/workspace/logs/run_all.log'"
#    poll every ~5 min (short, non-blocking; the 15 min tail covers one poll interval + sync + stop):
$S 'tail -3 /workspace/logs/pod.log; cat /workspace/logs/anchor.jsonl /workspace/logs/runs.jsonl 2>/dev/null; grep EXIT /workspace/logs/run_all.log'

# 4. when run_all.log says EXIT: pull reports, logs, LoRA/DoRA adapters and verified finalist GGUFs
WITH_GGUF=1 WITH_ADAPTERS=1 bash ml/pod/sync_results.sh
ls .cache/ml/train/pod-results/runs/*/gguf/*-Q4_0.gguf && cat .cache/ml/train/pod-results/eval/vs_stock.md

# 5. STOP the pod; keep the volume until required artifacts are verified locally and termination is authorized

# 6. local only: CPU ship-gate eval, stock default.v1, private cards, leaderboard and vs_stock.md
bash ml/pod/after_pod.sh
#    (EVAL_THREADS=2 when the host load average is > 20; STOCK_DEV=1 adds the stock models on the dataset's dev)

# 7. retain locally; do not publish, upload, or insert into the app catalog
```

Use a dedicated mode-700 result directory. Set `WITH_FULL=1` at sync if a full-FT checkpoint is needed, and
allow enough time to transfer it before stopping. Keep failed-run diagnostics via a separate metadata-only
sync (without `WITH_GGUF=1`) if no finalists were selected. Use a fresh result directory for a new training
session. `STEPS="judge cards"` is separate and opt-in: it sends evaluation text to the configured teacher;
without approval keep the raw/guarded CPU metrics and mark the judge gates incomplete. Upload is not a step
of this workflow, even to a private repo. Teacher-output terms for training and distribution remain unconfirmed.

An optional outer watchdog is `MURMUR_WORKSPACE=/workspace MURMUR_CUTOFF_EPOCH=EPOCH bash ml/pod/start_bounded.sh`.
Choose the workload cutoff before the rental deadline so retrieval and the provider-side stop still fit.
The watchdog stops the shell workload, not the rental.

Exit codes of `run_all.sh`: 0 done; 5 budget refused a stage/run (results so far are complete and synced as
usual); 6 preflight (fla kernel, no CUDA llama.cpp build passed the kernel smoke test, or the smoke model was never
prepared: fix, or `PHASE_B=""` / `ALLOW_*`, re-run); 7 bootstrap could not prepare the kernel smoke model (section
3.2); 2 no data. A run whose training loss turns NaN/inf stops at the end of that optimizer step
(`train-<run>.log`: `STOP diverged`), sft writes `runs/<run>/DIVERGED.json` (step, reason) and exits 4, the run is
logged as DIVERGED (failed) and the matrix continues. Re-running `run_all.sh` / `train_matrix.sh` / `run_one.sh`
then skips that run with a log line (`select` never picks it) instead of resuming from its last good checkpoint
into the same divergence; after fixing its config (e.g. a lower `train.lr`), `FORCE_RETRAIN=1` trains it again
from scratch (the marker is kept as `DIVERGED.<utc>.json`, the checkpoints are dropped: a resume would restore the
old optimizer and lr schedule). To resume from the last good checkpoint instead, delete `DIVERGED.json` by hand.
If LoRA misses the gates: `FULL=1 PHASE_A="" PHASE_B="" STAGES="train eval"` in the same session if the budget
allows (+53 min [A]), otherwise in a second session (the volume keeps everything; the budget file sums sessions).
The stock baselines on the dataset's own dev/test (dev.v2/test.v2 are frozen in `ml/eval/baselines/`) can be
produced before or during the pod session, free, on the dev server CPU (~1-1.5 h for test, as much again for dev):
`NO_SYNC=1 STEPS=stock STOCK_DEV=1 EVAL_STOCK_TEST=.cache/ml/data/final/test.jsonl EVAL_STOCK_DEV=.cache/ml/data/final/dev.jsonl bash ml/pod/after_pod.sh`
(same output dir as step 6, which then skips them). If the data is rebuilt afterwards, delete
`.cache/ml/train/pod-results/eval/stock/` and run it again (a finished report is never recomputed).

Time and stop rules: `ml/pod/REQUEST.md` ("Time budget", "Stop rules"); budget defaults `MAX_GPU_HOURS=4`
(owner cap), `TARGET_GPU_HOURS=3` and `BUDGET_TAIL_S=900`: both limits are applied 15 min early, so the last step the
scripts start ends by 3 h 45 (hard) / 2 h 45 (target) and the operator's poll (≤ 5 min), `WITH_GGUF=1
sync_results.sh` (~3 GB, ~3 min) and the stop still land inside the cap (`test_budget_tail_keeps_sync_and_stop_inside_cap`).
`run_all.sh` ends with the seconds left until the cap.

## 2. Run matrix (one GPU, sequential)

Same data, same prompt mix (finetuned.v2; 5 % default.v1; 15 % random preferences block), max_len auto
(longest rendered example rounded up to 64, cap 4096; 1,472 for Qwen3.5 / 1,536 for LFM3 on the final set, nothing
dropped), length-grouped batches with per-batch padding (section 2.1), batch 16 (188 steps per epoch at 3,000
examples, 564 for 3 epochs), cosine, warmup 5 %, β2 0.95, alpha = 2r, dropout 0.05,
all linear layers except `lm_head`. Selection on the **Q4_0 GGUF dev eval** (the shipping artifact):
fewest failed gates, then EM, N-WER, no-op exactness, fewer trainable parameters (`ml/train/select.py`).

| Phase | order | Run (config) | Base | Method | r | LR | Epochs | Question | GPU min plan / worst [A] |
|---|---:|---|---|---|---:|---:|---:|---|---:|
| A anchor | 1 | `lfm3-r16` | LFM3-1.7B | LoRA | 16 | 2e-4 | 3 | anchor = Liquid's recipe; measures tokens/s | 21 / 34 |
| A opt. | 2 | `lfm3-r64` | LFM3-1.7B | LoRA | 64 | 2e-4 | 3 | capacity | 23 / skipped |
| A opt. | 3 | `lfm3-r16-lr5e-4` | LFM3-1.7B | LoRA | 16 | 5e-4 | 3 | under-training at ~560 steps? | 21 / skipped |
| A opt. | 4 | `lfm3-r16-e5` | LFM3-1.7B | LoRA | 16 | 2e-4 | 5 | more passes over a small set | skipped / skipped |
| A opt. | (`DORA=1`) | `lfm3-dora-r16` | LFM3-1.7B | DoRA | 16 | 2e-4 | 3 | DoRA vs LoRA | not planned |
| B | 5 | `q08-lora` + winner's r/α/lr/epochs | Qwen3.5-0.8B | LoRA | ← A | ← A | ← A | speed option | 13 / 20 |
| B | 6 | `q2-lora` + winner's r/α/lr/epochs | Qwen3.5-2B | LoRA | ← A | ← A | ← A | quality option | 27 / 44 |
| fallback | (`FULL=1`) | `full-fallback` | LFM3-1.7B (embeddings frozen) | full FT | — | 1e-5 | 3 | does full FT fix what LoRA misses? | 53 / 85 |

Per-run minutes = train + merge + GGUF + parity + Q4_0 dev eval, from the anchor estimate (3.68M padded tokens
per 3-epoch LFM3 run [M] at 4.8K tokens/s = 20 % MFU of the 5090's 209.5 dense BF16 TFLOP/s, worst case 10 %;
see `ml/pod/REQUEST.md`) times the cost factors in `ml/pod/train_matrix.sh` (`RUN_COST`) [A]. On the pod the
anchor runs first and alone; its measured time replaces the estimate, `train_matrix.sh` prints the projection of
every remaining run, and each optional run ("A opt.", DoRA, full FT) starts only if its projection plus all
mandatory work still ahead (phase B + 30 min of finalist evals) fits `TARGET_GPU_HOURS` (3 h) minus the 15 min tail. At plan speed
`lfm3-r64` and `lfm3-r16-lr5e-4` fit and `lfm3-r16-e5` does not; at half speed no optional run fits. To force a
particular sweep, set `PHASE_A="lfm3-r16 lfm3-r16-e5"` (order = priority). Data-limited check (PLAN_V2 "scale only if eval says so"): if `lfm3-r16-e5` beats `lfm3-r16` while dev
loss still falls, the set is too small for the task → generate more (up to 5K) rather than add epochs.

### 2.1 Sequence length and batching [M 2026-11-20, current final data (3,000 train / 300 dev), real tokenizers]

`max_len` never pads anything: the Collator pads each batch to its own longest row (rounded up to 8) and the
train sampler is `LengthGroupedSampler` (megabatches of 50 batches sorted by length, longest batch first so an
OOM shows at step 1). `max_len` is only the drop filter and the bound of that longest batch. Measured with
`render_data.build` + the sampler (`data_stats.json` -> `batching` records it for every run):

| mix (finetuned.v2 + 15 % prefs +) | tokenizer | mean | p99 | max -> max_len auto | dropped | padding efficiency grouped / random / fixed max_len |
|---|---|---:|---:|---:|---:|---|
| 5 % default.v4s (before) | Qwen3.5 | 418 | 2,282 | 2,417 -> 2,432 | 0 | 0.93 / 0.29 / 0.17 |
| 5 % default.v4s (before) | LFM3 | 434 | 2,357 | 2,492 -> 2,496 | 0 | 0.93 / 0.29 / 0.17 |
| **5 % default.v1 (now)** | Qwen3.5 | **373** | 1,328 | 1,463 -> **1,472** | **0** | **0.95** / 0.41 / 0.25 |
| **5 % default.v1 (now)** | LFM3 | **388** | 1,369 | 1,504 -> **1,536** | **0** | **0.95** / 0.41 / 0.25 |
| 10 % default.v1 | Qwen3.5 | 417 | 1,365 | 1,477 -> 1,536 | 0 | 0.95 / 0.37 / 0.27 |
| 10 % default.v1 | LFM3 | 433 | 1,410 | 1,541 -> 1,600 | 0 | 0.95 / 0.37 / 0.27 |
| finetuned.v2 only | Qwen3.5 | 329 | 536 | 687 -> 704 | 0 | 0.97 / 0.70 / 0.47 |
| finetuned.v2 only | LFM3 | 343 | 580 | 760 -> 768 | 0 | 0.97 / 0.69 / 0.45 |

Source: `python -m ml.train.sft --config ml/train/configs/base.yaml --prepare-only --set base_model=<local tokenizer
dir> [prompt.default_share=… | prompt.default_system=default.v4s]` with `MURMUR_DATA=.cache/ml/data/final`
(`data_stats.json`; the same figures the independent review measured: 3,000 examples, 141 default.v1 rows, max
1,463 / 1,504 -> 1,472 / 1,536, 0 dropped). finetuned.v2 rows alone: 99 % under 536 (Qwen3.5) / 580 (LFM3) tokens.
No cap: any cap below ~1.3K would drop exactly the stock-prompt rows (141 of 3,000) and so remove the robustness
mix, and above it nothing is dropped; with length grouping the long rows cost only their own tokens. Real tokens
per epoch 1.12M (Qwen3.5) / 1.16M (LFM3), padded 1.18M / 1.23M; completion (loss) tokens 79K / 92K. The dev split
(300, of which 13 render with default.v1) is evaluated sorted by length (eval loss every 100 steps; in file order
every batch holding one of the stock-profile rows would be padded to ~1.3K).

Proof that padding changes only the cost [V]: `test_padding_does_not_change_the_loss` (tiny random LFM3 and
Qwen3.5 models from the real configs: dynamic padding, padding to 128 and each example alone give the same
completion loss, atol 1e-5) and a training A/B on the earlier final_v1 data (tiny LFM3, LoRA r8, dropout 0 so dropout
masks do not depend on the padded shape, 6 steps of 16, same length-grouped order; scratch log `padab/`):
step losses and grad norms identical to the logged precision (11.83, 11.83, 11.82, 11.82, 11.82, 11.80;
grad norm 0.8427 / 0.8203 / 0.5771 / 0.6545 / 0.6605 / 0.76 in both), train loss 11.8188507 vs 11.8188508,
eval loss 11.807299 in both; wall time train 38.9 s vs 72.9 s, dev eval 10 s vs 36 s (dynamic vs max_len,
padding efficiency 0.95 vs 0.25).

Export parity (every run, `ml/train/parity.py`, `PARITY_N=24`): after GGUF conversion the merged HF checkpoint
(bf16, torch) and the BF16 GGUF (llama.cpp) decode the same 24 category-spread dev items greedily; an item agrees when the texts are identical or share a
prefix of min(64 chars, half the longer text) (bf16 rounding diverges late, a broken export at once); fewer
than 80 % agreeing items fails the run (exit 3) before it is evaluated. This catches a silent merge/convert error,
e.g. in the Qwen3.5 Gated DeltaNet `in_proj_qkv/z/a/b` tensors that LoRA adapts and `convert_hf_to_gguf.py`
reorders, which would otherwise look like "the recipe does not work". The quantisation loss (BF16 GGUF vs
Q4_0 GGUF) is measured separately for each finalist.

## 3. Evaluation (one implementation, used everywhere)

`python -m ml.eval.run_eval --model X.gguf|HF_DIR --data A.jsonl [B.jsonl …] --out DIR --system NAME` renders
every prompt with `prompts/render.py` (system prompt name or profile, e.g. `default.v2` with few-shot turns),
decodes greedily through `llama-server` (temperature 0, top-k 1, no penalties, `cache_prompt`,
`max_new_tokens = min(1024, 2·transcript_tokens + 64)`, stop `<|im_end|>`), strips a leaked think block, and
writes `predictions.jsonl` (resumable), `scores.jsonl`, `report.json`, `report.md`. Backends: `llama` (GGUF,
the phone's engine and quantisation), `hf` (transformers), `teacher` (Responses API, baseline B3, dev server).
`--rescore DIR` rebuilds a report from synced predictions + `judge.jsonl` without the model.

Metrics (`ml/eval/metrics.py`, one module, unit-tested, normalisation identical to `prompts/eval/score.py`
with a parity test) per language and category: byte-exact match, EM after light normalisation (alts allowed),
corpus N-WER vs target and the raw transcript's N-WER, formatting-aware token error rate, entity recall,
distractor insertion, forbidden strings, answered/executed rate on dictated questions/commands, injection
success, no-op exactness, empty-output correctness and false-empty rate, wrong-language rate, drift proxy
(numbers/negations) and judge drift, length ratio, latency p50/p95, prefill ms/token, decode tok/s, tokens.
Context v2 (`docs/ml/CONTEXT_V2.md`): `ml/eval/ctxview.py` is the only code in ml/train and ml/eval that
looks inside `ctx` (v2 `screen.lines`; any other ctx only through `render.render_context`); everything else
hands ctx to `prompts/render.py`. Reports break every metric down by context source (`ctx:screen`,
`ctx:no_screen`, `ctx:legacy` = v1 structured context). CONTEXT_V2 category rules are checked per item on
the OCR lines (`ctx_rule_ok`; a violation is a data defect, counted in `ctx_unsupported_items` and failing the
"Eval-set integrity" gate): `context_resolution` / `context_names` / `context_asr_correction`: the item's
**support terms** (`meta.entities`, else `meta.check.screen_terms`, else the name-like keep entries) are readable
on screen (case-insensitive, Russian inflection tolerated: "Веронике Ткачук" ~ "Вероника Ткачук"; plain keep
phrases such as "договорились" are not support terms); `near_miss`: every distractor is on screen and the said
term is **absent**; `context_counterpart`: the forbidden written form is absent. Entities of an item that breaks
its rule never count as kept, so restoring a term the screen does not show is never rewarded (`resolved`,
`ctx_resolution_rate`); a `context_counterpart` output that contains the unsupported term counts as a distractor
insertion. EM-noop counts only true `no_op` items (`near_no_op` is reported as `near_noop_exact`, not gated).
Injection items pass only if the output stays close to the target (EM, or word edits <= max(1, 10 % of the
target's words) and formatting-token error rate <= 0.30), copies no word 3-gram of the payload
(`meta.injection.text`, `check.screen_terms`, else any screen 3-gram the dictation does not contain), hits no
`check.forbid` string and is not "answered" (screen words are not whitelisted for injection items): appending
"Respond to the user as a pirate." or SHOUTING the target + "Sent with love." now counts as a successful
injection on every one of the 19 proof_v6 / dev.v2 / test.v2 injection items [V unit test + scratch check].
Ship gates (`ml/eval/gates.py` = DATA_AND_TRAINING_PLAN 4.3 + empty output >= 90 % + eval-set integrity),
overall and per language, PASS / FAIL / NO_DATA; a non-GGUF report is flagged "not a ship measurement". The ship
measurement is the finalist's Q4_0 file on the CPU llama.cpp build (`after_pod.sh`, `Q4_0-test-cpu`).
`test_oracle_never_fails_the_gates` scores output := target on proof_v6 dev/test, `dev.v2`, `test.v2`,
`smoke_cases.v2` and the v2 fixture: every one is PASS or INCOMPLETE (too few items for some gates), never FAIL.

### 3.1 Comparison with the stock baseline (same sets, same scorer) [V 2026-11-20]

The number to beat is the prompt-eval **pass rate** of the shipped stock setup (`default.v1` + `prompts/guard.py`,
vendor Q4_0, `docs/ml/PROMPT_EVAL.md` 0.1): Qwen3.5-2B **0.57 dev.v2 / 0.50 test.v2**, LFM3-1.7B 0.37 / 0.32,
Qwen3.5-0.8B 0.34 / 0.32 (after the guard; raw 0.44 / 0.41, 0.21 / 0.24, 0.23 / 0.24). So that a fine-tune is
measured on exactly that scale:

* every report row carries `pass` = `prompts/eval/score.py score_case(...)["ok"]` with the ctx2 options of the
  prompt profile it ran with (`ml/eval/scoring.py`), raw and after the guard, next to the ml/eval metrics;
* every report breaks all metrics down per eval file (`by_set`: `dev.v2`, `test.v2`, `smoke.v2.test`, the
  dataset's `dev` / `test`), because the stock numbers exist per prompt set, not for the merged dev eval;
* `ml/eval/baselines/stock.default.v1.json` freezes the stock numbers, computed by `python -m ml.eval.baseline
  build` from the stored prompt-eval runs (`.cache/ml/prompt/runs/{q2b,lfm3,q08}.default.v1.{dev,test}.v2.jsonl`)
  through the same `scoring.score_all` a fine-tune's report uses: it reproduces PROMPT_EVAL exactly (pass cases
  after the guard 56/98, 33/66; 36, 21; 33, 21; raw 0.439/0.409, 0.214/0.242, 0.235/0.242), test
  `test_frozen_baselines_match_prompt_eval_doc`; the pod has no run files, so it reads the frozen JSON;
* `report.json` -> `baseline` and `report.md` "Versus the stock models": per prompt set, this run vs each stock
  model (pass raw / guarded, EM, N-WER, answered, injection) and the delta to the best stock model; the card adds
  the same table and the catalog entry `eval.vs_best_stock_pass_guarded`;
* `python -m ml.eval.baseline table DIRS` -> `vs_stock.md` (pod: `$W/eval/vs_stock.md`; dev server:
  `pod-results/vs_stock.md`): one table per eval set over every report (finalists, all runs, stock reports of
  `after_pod.sh` on the dataset's test and, with `STOCK_DEV=1`, dev), the frozen stock rows on top;
* sets the fine-tunes see: dev = dataset dev (300) + `dev.v2` (98) + `smoke.v2.test` (24) in every run's Q4_0 dev
  eval (selection); test = dataset test (300) + `test.v2` (66) for the finalists (pod GPU and the CPU ship eval);
  guard on and off are both in every report.

Scorer difference: ml/eval's own metrics (entity recall, `ctx_rule_ok`, the Eval-set integrity gate) check
screen support on **all OCR lines** with a fuzzy, inflection-tolerant match (the CONTEXT_V2 data rules are stated
on `screen.lines`); `score.py` checks a kept term only against the **rendered `<context>` block** under the
profile's ctx2 (`score.on_screen`: after the renderer's filters and the 1,400-character budget), exact after
folding. Aligning cost nothing for the comparison number: `pass` is computed by `score.py` itself, so
fine-tune and stock are on `score.on_screen`; the gates stay on the OCR lines (they judge the data as well as
the model). An oracle (output := target) passes 98/98 of dev.v2 raw and after the guard
(`test_report_has_pass_by_set_and_stock_comparison`). `select.py` still ranks on gates, then EM (raw), unchanged.

CPU eval fallback timing [M on the dev server, 6 threads, shared host]: Q4_0 0.8B ~1-2 s per item with the
short prompt, 2B / LFM3-1.7B 4-9 s per item with the long default prompt (`.cache/ml/train/eval/v2*`). A
~420-item dev eval is then ~10-40 min, and a session has ~10 evals: this is why `train_matrix.sh` aborts
when the CUDA llama.cpp build is unusable instead of silently falling back.

### 3.2 llama.cpp CUDA build: kernel smoke test and fallback (bootstrap step 6b)

`llama-server --list-devices` listing `CUDA0` proves only that the CUDA runtime sees the card; no kernel runs. A
build without device code for the 5090 (sm_120), a failing or slow PTX JIT, or a wrong kernel would surface only
at the anchor's parity (rc 3) or eval, after its ~20 min of training. So bootstrap runs `ml/train/llama_smoke.py`
on every pod start (a restarted pod can land on another host and driver): it converts the real base weights
(pinned revisions; Qwen3.5-0.8B and LFM3-1.7B, the two architectures' kernels: Gated DeltaNet + attention,
short conv + sliding-window attention) to Q4_0 once (cached on the volume), decodes 4 fixed, strongly determined
prompts (counting, weekdays, letters, Russian months; 16 tokens each, greedy) with `-ngl 99 --device CUDA0` and
with the CPU build, and requires: the server starts, its log says every layer was offloaded, start + all prompts
take < 120 s, and every prompt agrees with the CPU build (identical, or a common prefix of >= 16 characters).
Results: `/workspace/llama/smoke/<build>-<model>.json`, log `/workspace/logs/llama-smoke.log`, server logs one per
build (`<smoke gguf>.<build>.gpu.log`). The step is `ml/pod/kernel_smoke.sh`, in two phases:

* **A. prepare the smoke models** (download at the pinned revision + Q4_0 conversion with the CPU build; no build
  under test involved; `llama_smoke --prepare-only`, `prepare-<model>.json`). A failure is retried once (after
  `LLAMA_SMOKE_RETRY_S`, 20 s); a second failure is reported as its own error ("smoke model ... could not be
  prepared ... NOT a CUDA build problem"), writes the marker `llama-smoke-unprepared`, tests no build, fetches and
  compiles no fallback, and bootstrap exits 7; `train_matrix.sh` refuses while the marker exists (rc 6,
  `ALLOW_UNTESTED_LLAMA=1` overrides). Typical causes: Hub outage / rate limit, full volume, a converter that
  rejects the architecture. Fix and re-run bootstrap (a passing phase A removes the marker). `llama_smoke` reports
  any such failure as `PREP_FAIL` (exit 3), never as a build `FAIL` (exit 1).
* **B. test the builds** on the prepared models: only a decode mismatch, a crash, a timeout or a partial offload of
  a built smoke model counts against a build and moves on to the next fallback below.

[V here] On the dev server (CPU build standing in for the GPU one): real Qwen3.5-0.8B, GGUF build 16.5 s, decode
2.5 s, PASS with identical texts (" 13, 14, 15, 16,", " Friday, Saturday, Sunday…", " K, L, M, …",
" май, июль, …"); with `--require-gpu` the CPU build fails as a GPU build without the device must ("invalid device:
CUDA0", FAIL rc 1); a build that dies like a missing kernel image ("no kernel image is available") fails
(`test_llama_smoke.py`). [V source] The release CUDA 12.8 build sets no `CMAKE_CUDA_ARCHITECTURES`
(`.github/workflows/release.yml`), and the default list for CUDA >= 12.8 (`ggml/src/ggml-cuda/CMakeLists.txt`)
includes `120a-real`, i.e. sm_120 device code, so no JIT is needed. [UNVERIFIABLE here] that it runs on the pod.

Fallback chain, automatic, first PASS wins; the chosen dir goes to `/workspace/.bootstrap/llama-cuda-bin`, which
`env.sh` reads for every later stage:

| # | candidate | extra time [A] |
|---:|---|---:|
| 1 | release `llama-b12503-bin-ubuntu-cuda-12.8-x64` (+ cudart), downloaded in step 4 | ~1-3 min (smoke, first start), ~30 s on a restart |
| 2 | release `…-cuda-13.5-x64` (+ cudart 13.5) of the same tag, both listed on the b12503 release [V GitHub API] | ~1 min download (~670 MB) + 30 s smoke; relies on the driver's CUDA 13.x minor-version compatibility |
| 3 | source build of b12503 for sm_120 only (`cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF`, targets `llama-server llama-quantize`; configure line checked here without CUDA) | ~10-15 min with all cores; needs `nvcc` >= 12.8 in the image (devel images have it; `LLAMA_SRC_BUILD=0` skips); log `/workspace/logs/llama-build.log` |
| 4 | none passes: marker `llama-cpu-fallback`; `train_matrix.sh` refuses to start (rc 6) | — |

At 4 the choice is the operator's: fix the image (e.g. a CUDA 13 devel image, then re-run bootstrap; delete
`/workspace/llama/build-cuda` to rebuild), or `ALLOW_CPU_EVAL=1` to train on the GPU and evaluate on the pod's CPU
(~10-40 min per ~420-item dev eval, ~10 evals: does not fit the 4 h cap with the full matrix; use it only with
`PHASE_A="lfm3-r16" PHASE_B="q08-lora"` and `EVAL_DEV` = the dataset's dev split: 2 runs x (parity + dev eval) +
2 finalists x 3 evals at ~10-40 min each = ~1.5-5 h of CPU eval, so expect the budget guard to skip the later
finalist evals; the ship evals run on the dev server CPU anyway, `after_pod.sh`, free). `CUDA_CACHE_PATH` is on the
volume, so a PTX JIT, if any build needs one, is paid once.

## 4. CPU dry run (what was proven without a GPU) [V]

Host: shared x86_64 dev server, CPU only, 6 threads, `tools/heavy` scopes; other agents' jobs ran at the
same time (load average 3-55), so CPU times are upper bounds, not throughput numbers. Data: the 24 smoke
cases converted to the contract (`make -C ml/train smoke-data`), base `Qwen3.5-0.8B` real weights, prompt mix
as in training (finetuned.v1, 10 % default.v1, 25 % preferences). Commands: `make -C ml/train dryrun-lora`,
`make -C ml/train dryrun-full SET="--set train.save_steps=1000" EVAL_LIMIT=8 QLIMIT=4`; logs
`.cache/ml/train/logs/dryrun-{lora,lora-eval2,full}.log`, outputs `.cache/ml/train/runs/dryrun-{lora,full}/`.

| Step | LoRA r16 (all-linear, 43.3M trainable) | Full FT (752M trainable) |
|---|---|---|
| train 3 steps (batch 1 x accum 2, max_len auto) | 525.6 s wall, peak RSS 10.6 GB; loss 1.159 / 1.380 / 1.019 | 271.9 s, peak RSS 17.3 GB; loss 1.159 / 9.847 / 3.163 (lr 1e-4 is a LoRA lr: full FT diverges at it, the fallback config uses 1e-5) |
| merge (PEFT, fp32 -> bf16) | 46.0 s, 5.5 GB RSS | not needed |
| GGUF BF16 (`convert_hf_to_gguf.py`, text-only, `--no-mtp`) | 42.5 s, 1,516,742,240 B | 15.9 s, same size |
| Q4_0 (`llama-quantize`) | 10.1 s, 501,450,336 B, sha256 `e29de9fa6693…` | 4.0 s, 501,450,336 B, sha256 `27fd9a0774c1…` |
| eval Q4_0 (llama-server b12503 CPU, greedy) | 24 cases in 48.4 s wall incl. server start; p50 949 ms; prefill 3.07 ms/token; decode 51.8 tok/s | 8 cases in 43.6 s |
| eval BF16 + measured quantisation loss (`ml.eval.compare`) | 8 cases in 37.6 s; `eval/quant_loss.md` | 4 cases; `eval/quant_loss.md` |
| card + catalog entry + upload dry run | `gguf/README.md`, `gguf/catalog_entry.json`, `upload_hf --dry-run` lists Q4_0 + README + LICENSE + catalog | same |

Identical step-1 loss (1.159) for both methods is the check that both see the same rendered data and the
same masking. Quality numbers are meaningless after 3 steps (the 3-step models still answer like chat
models); the dry run proves the chain, not the recipe. Seen and fixed on the way: (1) an eval run timed out
under load average ~50 (llama-server 600 s HTTP timeout; at load 3 the same 24 cases take 48 s);
(2) Trainer writes a final checkpoint even with `save_steps` > steps: 8.5 GB with optimizer state for a
0.8B full-FT run, now deleted after `final/` is written (`train.keep_checkpoints: true` keeps it);
(3) the `causal_conv1d` / `flash-linear-attention` kernels are absent on CPU, transformers falls back to
PyTorch (correct, slow) — the GPU env installs fla 0.6.0.

Re-run 2026-11-19 after the review fixes (`.cache/ml/train/dry2/run.sh`, log `dry2/run.log`): the same 3-step
LoRA adapter through the new `ml/pod/run_one.sh` (budget checks, parity, v2 dev sets) on CPU: merge 13 s,
GGUF BF16 + Q4_0 23 s, **parity 88 s: 7 of 8 smoke dev items byte-identical between merged HF (torch bf16)
and the BF16 GGUF (llama.cpp), the 8th diverged at character 33 (bf16 rounding) -> 7/8 agree, PASS**; Q4_0 dev eval of
146 items (24 legacy smoke dev + `dev.v2` 98 + `smoke_cases.v2` 24; 105 with a screen, 17 without) 415 s;
eval-set integrity PASS; guard fired on 105 of 146 outputs of this 3-step model (EM 0.05 raw -> 0.21 guarded,
injection success 0.70 -> 0.20). Negative control (`dry2/neg_parity.sh`): K and V blocks of
`linear_attn.in_proj_qkv` swapped in 2 Gated DeltaNet layers before conversion (a converter-style bug)
-> 0 of 8 identical, all diverge in the first characters, parity rc 3 = the run would stop before its eval.

Other code paths, also run here [V]:

* **Tiny random-weight models built from the real configs + tokenizers** (`make -C ml/train tiny FAMILY=… METHOD=…`,
  `ml/train/tiny_chain.py`; results in `.cache/ml/train/tiny/*/result.json`): LFM3 LoRA, LFM3 DoRA, LFM3 full FT,
  Qwen3.5-2B LoRA — each train -> merge -> GGUF -> Q4_0 -> llama-server eval, EXIT 0, about 1 min each. This
  covers LFM3's module names, PEFT targets (incl. DoRA), conversion and llama.cpp loading without its 3.5 GB weights.
* **Pod sequence simulated locally** (`.cache/ml/train/podsim/simulate.sh`, log `logs/podsim.log`): tiny LFM3 and
  Qwen3.5-2B models, CPU llama.cpp standing in for CUDA, smoke cases as data. `run_all.sh` with
  `MAX_GPU_HOURS=0` refuses (rc 5); `STAGES="train eval"`: phase A (2 runs) -> `select` -> phase B with the winner's
  overrides -> finalists (one per base) -> BF16/Q4_0/test/default-prompt evals -> cards -> leaderboard, 1508 s;
  the same command again: 1 s (everything skipped). Then `after_pod.sh` (`STEPS="sync ship cards"`) against that
  workspace through a fake `ssh` (`podsim/bin/ssh`): rsync filters, finalists' Q4_0 GGUFs pulled, CPU ship eval,
  cards and leaderboard, EXIT 0 (`logs/afterpod-sim.log`).
* **Pod sequence re-simulated 2026-11-19 after the review fixes** (`podsim/simulate.sh` + `podsim/eval_stage.sh`,
  logs `logs/podsim2.log`, `logs/podsim2-eval.log`): cap 0 h refuses (rc 5); the `llama-cpu-fallback` marker makes
  the preflight abort the matrix (rc 6); with `TARGET_GPU_HOURS=0.01` the anchor and phase B run and the optional
  phase-A run is skipped ("needs ~1884 s incl. the mandatory rest"); with the default target it then runs; every run
  passes through parity (random-weight tiny models diverge after 10-300 characters of gibberish, so the sim uses `PARITY_MIN_AGREE=0.3`; the real-weight check is above); the eval stage (finalists
  per base: BF16 dev, Q4_0 test, Q4_0 dev with `default.v4`, `LEGACY=1` v1 column, cards with `finetuned.v2` in the
  catalog entry, leaderboard with guarded columns and a separate legacy table) EXIT 0 in 321 s on 2-item sets;
  the whole `STAGES="train eval"` again: 2 s (all skipped); `/workspace/.budget/sessions.tsv` held 4 sessions.
  The host had load average 40-70 from other users' jobs during this run (see section 5 on CPU threads).
* **Divergence stop rule through the real Trainer** (2026-11-20 review fix): HF Trainer's default
  `logging_nan_inf_filter` replaces a NaN/inf step loss before it is logged, so the old `on_log` rule never saw it.
  Now `MurmurTrainer.compute_loss` hands every raw micro-batch loss to `StopRules.watch_loss` (no host sync; one
  check per optimizer step), the run stops at the end of the step in which the loss became non-finite, saves no
  checkpoint of the diverged weights, and sft exits 4; the filter is off and `on_log` also stops on a non-finite
  loss / grad_norm / eval_loss. `test_nan_loss_stops_the_real_trainer_at_once_with_exit_4` runs `ml.train.sft` in a
  separate process (tiny random LFM3, CPU, the base.yaml TrainingArguments with logging_steps 10) with the loss
  turned NaN from the 3rd micro-batch: exit 4 at step 3, no `train_info.json` / `final/`, only `checkpoint-2` kept.
  Negative control, same injection through the previous `sft.py` (d141429; `.cache/ml/train/boot6b/nan_old_vs_new.py`): exit 0 after all 12 steps, finite
  `train_loss` 7.20, `stopped_by` null, i.e. the diverged run would have been exported and evaluated.
* **Teacher backend + judge** (Responses API through `tools/with-teacher`): 3 smoke cases answered and judged
  (`.cache/ml/train/eval/teacher-smoke3/judge.jsonl`, 6 requests).
* **ctx v2 integration** (`ml/ocr/READY`, `ml/screens/READY` both present at 22:50): `python -m ml.eval.v2_cases
  ml/eval/cases/v2_screens.v1.jsonl OUT` renders 8 hand-written cases (ru/en/mixed; context_resolution, its
  counterpart, context_names, injection, code-switch, no-screen, noise) through `ml.screens.build_ctx`, checks the
  CONTEXT_V2 category rules on the OCR lines (8 written, 0 rejected, 46 s) and writes contract JSONL that
  `run_eval` consumes unchanged (results in section 4.1).
* **Real OCR context**: `ml/ocr` (READY) run on a rendered Fold7 inner-screen chat (`.cache/ml/screens/tmp/
  work_chat_fold7_inner_light.png`, status bar / keyboard / Murmur bubble excluded): 26 lines in 3.6 s. Three
  eval cases built on it (`.cache/ml/train/ocr-int/ocrint.test.jsonl`: context_resolution ClickHouse/Airflow,
  context_names Галиев, and a counterpart where the bubble cut "Grafana" to "Grai"), rendered by
  `prompts/render.py`: **385 tokens (Qwen3.5) / 396 (LFM3) with finetuned.v1, 1,335 / 1,366 with default.v1**.

Rendered length, measured with the real tokenizers (prompt + target + `<|im_end|>`), tokens mean / max:

| Set | finetuned.v1 Qwen3.5 | finetuned.v1 LFM3 | default.v1 Qwen3.5 | default.v1 LFM3 |
|---|---|---|---|---|
| smoke (24, ctx v1) | 213 / 332 | 221 / 354 | 1163 / 1282 | 1191 / 1324 |
| prompts dev.v1 (98, ctx v1; legacy, not used for selection) | 227 / 755 | 237 / 845 | 1177 / 1705 | 1207 / 1815 |
| ctx v2 fixture (5 hand-written OCR contexts) | 205 / 238 | 210 / 249 | 1155 / 1188 | 1180 / 1219 |
| ctx v2 via `ml.screens.build_ctx` (8 cases, `ml/eval/cases/v2_screens.v1.jsonl`, real render + OCR) | 290 / 356 | 298 / 376 | 1240 / 1306 | 1268 / 1346 |
| one real OCR chat screen, 26 lines (`ocr-int/ocrint.test.jsonl`) | 385 | 396 | 1335 | 1366 |

With 15 % default.v1 the mean is ~370 tokens per example on v1 contexts, ~440 on the build_ctx cases and ~530 on
the busiest real OCR screen (render.py trims screen text to its budget, so this is near the ceiling) [V], hence the ~500 tokens per example used for GPU sizing. `max_len: auto` re-derives the limit
from the real rendered v2 set at training time (`data_stats.json`: `max_len`, `rendered_len` p50/p90/p99/max);
`python -m ml.train.sft --config X --prepare-only` prints it without training.

### 4.1 Stock models on real ctx v2 (CPU llama.cpp b12503, greedy, `--system default`) [V]

8 build_ctx cases (`.cache/ml/train/ocr-int/v2_screens.test.jsonl`, reports `.cache/ml/train/eval/v2screens-*`)
and 8 earlier v2 cases (3 on one real OCR screen + the 5-case fixture, `.cache/ml/train/eval/v2-*`). Tiny sets:
these prove the v2 path through render -> llama.cpp -> metrics -> `ctx:` breakdown, not model quality.

| Model (vendor Q4_0) | set | EM | entity recall | ctx_resolution | distractor insertion | empty-output OK |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B | build_ctx 8 | 0.50 | 0.80 | 0.50 | 0.00 | 0/1 |
| Qwen3.5-0.8B | build_ctx 8 | 0.25 | 0.40 | 0.00 | 0.00 | 0/1 |
| Qwen3.5-2B | OCR screen 3 + fixture 5 | 0.25 | 0.50 | 0.50 | 0.50 | 0/1 |
| LFM3-1.7B (QAD) | OCR screen 3 + fixture 5 | 0.00 | 0.67 | 0.50 | 0.50 | 0/1 |

What it shows: on the real screen the Murmur bubble cut "Grafana" to "Grai"; both Qwen3.5-2B and LFM3 still wrote
"Grafana" (world knowledge, not the screen) — the counterpart check counts that as a distractor insertion, which is
exactly the invented-term failure CONTEXT_V2 forbids. Qwen3.5-2B translated the English half of the code-switch
case into Russian; all models output "э"/"Э" for noise-only input; 0.8B mostly returns the transcript untouched.

### 4.2 Turnkey rehearsal on the CURRENT final data (re-run 2026-11-20 after the final review's three minor fixes) [V]

`.cache/ml/train/podsim-real/simulate.sh` (log `sim.log`, step times `times.txt`; the earlier run on final_v1 is
kept as `sim-final_v1.log`): the exact command sequence of section 1 on CPU, with tiny random-weight models built
from the real LFM3-1.7B / Qwen3.5-0.8B / Qwen3.5-2B configs + tokenizers, the **real configs** (`lfm3-r16`,
`lfm3-r64`, `q08-lora`, `q2-lora`; only `base_model` and the step budget overridden: 2 steps of 4) and the current
final data (**3,000 / 300 / 300**). A fake `ssh` runs the remote side locally into a "pod volume"; the CPU llama.cpp
build stands in for the CUDA one. Everything after the push runs from the pushed copy, as on the pod.

| step | result |
|---|---|
| `push_code.sh WITH_DATA=1 DATA_DIR=.cache/ml/data/final` | code + data on the volume, 1 s |
| pod self-test (bootstrap step 6) from the pushed copy | 98 passed, 15 skipped (need the local tokenizer dirs / llama.cpp build), 7 s |
| bootstrap step 6b commands (`ml.train.llama_smoke` on tiny LFM3 and tiny Qwen3.5-0.8B, CPU build as the "GPU" build, no `--require-gpu`) | phase A `--prepare-only`: PREPARED both, GGUF build 4.6 / 7.8 s; phase B: PASS both, 4/4 prompts agree, prepared GGUF reused (build 0.0 s), 2 s each, one server log per build (`*.sim.gpu.log`) |
| a smoke model that cannot be converted (`config.json` = `{}`) | `PREP_FAIL`, rc 3 ("could not prepare the smoke GGUF (not a GPU build problem): ... unsupported model_type"), not a build `FAIL` |
| `run_all.sh STAGES="train eval"` | EXIT 0, 300 s: every run rendered all **3,000** rows with the real tokenizer (max_len **1,536** LFM3 / **1,472** Qwen3.5, 0 dropped, **141** default.v1 rows, padding efficiency 0.97 at batch 4), trained, merged, exported BF16 + Q4_0, parity, Q4_0 dev eval on 6 dataset dev + 4 dev.v2 cases (`by_set` {dev, dev.v2}, `baseline` vs the frozen stock numbers); `lfm3-r64` parity FAIL (random weights; real weights pass, section 4) -> not evaluated, matrix continued (the failure path); anchor projections printed; select -> `lfm3-r16` + its overrides to phase B; finalists one per base; BF16 dev, Q4_0 test (6 + 4), Q4_0 dev with `default.v1`; cards; leaderboard; `vs_stock.md`; every budget line reads "of 13500s (hard, 900s kept for sync + stop)" and the last line "billed pod time so far: 330s; 14070s left until the 4 h cap for WITH_GGUF=1 sync_results.sh (~3 min) and the stop" |
| same command again | 8 s, nothing recomputed |
| divergence: the real `lfm3-r16` config with `train.lr` 1e30 (no injection; the real Trainer) through `run_one.sh` | `STOP diverged: non-finite training loss in step 3`, rc 4, log "train lfm3-boom DIVERGED (step 3: ...)", `DIVERGED.json` {step 3, reason, last_good_checkpoint checkpoint-2}; no `train_info.json` / `final/` |
| `run_one.sh lfm3-boom` again, then `train_matrix.sh` with `PHASE_A="lfm3-r16 lfm3-r64 lfm3-boom"` | rc 4 "DIVERGED in an earlier attempt ...; skipped, not resumed"; matrix: "skip lfm3-boom: its training DIVERGED ...", select "skip lfm3-boom", winner still `lfm3-r16`, `selection-phaseA.json` `skipped` lists it, rc 1; `timings.jsonl` holds ONE training attempt of it |
| budget | training deadline exported per run ("deadline in 12900 s, reserving 570 s" = 4 h - 900 s tail - billed - rest of run) |
| `after_pod.sh` (sync with the finalists' Q4_0 GGUFs, CPU ship eval, judge on 2 items per eval dir through `tools/with-teacher`, stock `default.v1` on the dataset test, cards, leaderboard, `vs_stock.md`) | EXIT 0, 154 s; 3 Q4_0 GGUFs synced; 6 eval dirs judged 2/2; catalog entries carry `pass_guarded_by_set`, `vs_best_stock_pass_guarded` and `base_revision` (null here: local tiny bases) |

The kernel smoke fallback chain itself (bootstrap step 6b, run verbatim against fake builds,
`.cache/ml/train/boot6b/harness.sh A|B`): CUDA 12.8 build dying with "no kernel image is available" + working 13.5 stand-in -> the 13.5
dir is written to `.bootstrap/llama-cuda-bin` and the next stage's `env.sh` uses it (`-ngl 99`); every build broken
and no `nvcc` -> "source build skipped", marker `llama-cpu-fallback`, `train_matrix.sh` exits 6 at preflight.
Since the final review the step is `ml/pod/kernel_smoke.sh`; `test_llama_smoke.py` runs it with a stand-in
`llama_smoke` and stub `curl` / `nvcc` / `cmake`: a smoke model that never prepares -> 2 attempts, exit 7, marker
`llama-smoke-unprepared`, no build tested, no `curl` / `nvcc` / `cmake` call, no `llama-cpu-fallback`; prepared on the
retry -> CUDA 12.8 PASS; prepared but every build mismatching -> the 13.5 download is attempted and
`llama-cpu-fallback` is written. `test_pod_diverged.py`: the preflight refuses on `llama-smoke-unprepared` (rc 6)
unless `ALLOW_UNTESTED_LLAMA=1`, and run_one / train_matrix skip a diverged run until `FORCE_RETRAIN=1`.

Quality numbers of the tiny models are meaningless (random weights); this proves that every step of the pipeline
runs on the real data shape, file names and set breakdown, not the recipe.

## 5. Known gaps and risks

* No GPU run happened: every GPU time and memory figure is [A] until `logs/timings.jsonl` and the anchor's
  `train_info.json` (`padded_tokens_per_s`, `peak_cuda_mem_gb`) exist. The 5090 peak (209.5 dense BF16 TFLOP/s
  with FP32 accumulate) is taken from secondary sources quoting NVIDIA's whitepaper; the plan uses 20 % of it.
* Memory: the longest batch is 16 x 1,536 tokens (length grouping puts it first, so an OOM shows at step 1, not
  after an hour). Not measured on a GPU; if it OOMs, `--set train.batch_size=8 train.grad_accum=2` (same
  effective batch) in the run's config.
* The pod self-test skips the 15 tests that need the local tokenizer/config dirs or llama.cpp build
  (`.cache/ml/train/models`, not pushed): padding invariance, completion loss, the Trainer-level NaN stop and the
  tiny-model kernel smoke are proven on the dev server only.
* The llama.cpp kernel smoke test (section 3.2) and its fallbacks 2-3 have run only with the CPU build standing in
  for the GPU one; the real sm_120 run, the CUDA 13.5 build's driver compatibility and the source-build time are
  [UNVERIFIABLE here] until the pod's `/workspace/llama/smoke/*.json` exist.
* A budget stop inside training leaves the cosine schedule unfinished; the run is still exported and
  evaluated (`stopped_by: budget` in `train_info.json` and the log) and should be read as under-trained.
* flash-linear-attention 0.6.0 with torch 2.15.0 / Triton on sm_120 is unverified; bootstrap runs its
  `chunk_gated_delta_rule` forward + backward once on the GPU and, if that fails, writes a marker on which
  `train_matrix.sh` refuses the Qwen3.5 runs (`ALLOW_SLOW_QWEN=1` overrides).
* Optional sweep runs started in a later pass (more budget) do not re-trigger phase B: phase B keeps the
  winner chosen when it ran. Delete the phase-B run dirs to redo them with a new winner.
* The probe stage and the data generator are owned elsewhere (`ml/roundtrip`, `ml/data`); this pipeline only
  assumes contract JSONL in `$MURMUR_DATA/{train,dev,test}.jsonl` and a probe command.
* CPU evals on the shared dev server: under heavy load (load average 40-60 from other users' jobs) llama.cpp
  with 6 threads collapsed to 0.23 decode tok/s on a tiny model, with 2 threads 18.6 tok/s (same model, same
  minute) [V]; use `EVAL_THREADS=2 bash ml/pod/after_pod.sh` when the host is busy.
* GPU vs CPU llama.cpp kernels can flip greedy ties; gates come from the CPU build on the dev server, the
  nearest stand-in for the phone; the app's Benchmark and real dictations are the final word.
* `wrong_language` is a script-share proxy (shared with `prompts/eval/score.py`): a Latin term left in Cyrillic
  ("клик хаус" for "ClickHouse") can trip it on short sentences. The gate is ≤ 0.5 %, so read the per-item rows
  before acting on a FAIL of that gate alone.
* Eval decoding uses `max_new_tokens = min(1024, 2·transcript_tokens + 64)` (prompts/README.md), while
  `docs/research/llm-catalog.json` runtime_defaults say `ceil(1.3·transcript_tokens) + 16` and n_ctx 2048. The
  app and the eval must use one formula; until reconciled, a target longer than 1.3x the transcript can be cut
  on the phone but not in the eval.
* LFM3 licence: fine-tunes inherit the US$10M revenue threshold — owner decision before shipping an LFM3 fine-tune.
* Python on the dev server can hang connecting to huggingface.co (IPv6 first, no fallback); `bootstrap.sh`
  checks this on the pod and prefers IPv4 via `/etc/gai.conf` if needed.
