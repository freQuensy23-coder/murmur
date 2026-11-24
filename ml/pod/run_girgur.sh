#!/usr/bin/env bash
# Turnkey Girgur (ru + en + he Qwen3.5-2B cleanup) training, export, parity, and eval pipeline
# for the 1x RTX 5090 (32 GB) pod. Idempotent and budget-guarded (ml/pod/env.sh).
#
# Steps:
#   1. Preflight check: verify combined RU+EN+HE dataset ($DATA_DIR/{train,dev,test}.jsonl)
#   2. Train LoRA on pinned Qwen/Qwen3.5-2B (ml/train/configs/girgur-q2-lora.yaml)
#   3. Merge LoRA adapter into BF16 HF weights
#   4. Export BF16 + Q4_0 GGUF:
#        $MURMUR_RUNS/girgur-q2-lora/gguf/girgur-cleanup-qwen3.5-2b-BF16.gguf
#        $MURMUR_RUNS/girgur-q2-lora/gguf/girgur-cleanup-qwen3.5-2b-Q4_0.gguf
#   5. Parity check: merged HF (bf16, torch) vs BF16 GGUF (llama.cpp) on PARITY_N dev items
#   6. Evaluate Q4_0 on dev (Girgur dev.jsonl + prompts dev.v2 + smoke.v2 -> RU+EN regression & Hebrew)
#   7. Evaluate Q4_0 on test (Girgur test.jsonl + prompts test.v2 -> RU+EN regression & Hebrew)
#   8. Optional BF16 dev eval + quantisation loss comparison (SKIP_BF16_EVAL=1 to skip)
#
# Usage (on pod):
#   bash ml/pod/run_girgur.sh
#   DATA_DIR=/workspace/data/girgur bash ml/pod/run_girgur.sh [extra --set overrides...]
#
# Env:
#   DATA_DIR          combined Girgur dataset dir (default: $W/data/girgur, fallback $MURMUR_DATA)
#   RUN               run name (default: girgur-q2-lora)
#   CFG               config path (default: ml/train/configs/girgur-q2-lora.yaml)
#   GGUF_NAME         exported GGUF base name (default: girgur-cleanup-qwen3.5-2b)
#   QUANTS            space-separated quant list (default: Q4_0)
#   PARITY_N          dev items for HF-vs-GGUF parity check (default: 24, 0 = skip)
#   PARITY_MIN_AGREE  minimum greedy parity agreement (default: 0.8)
#   KEEP_MERGED       set to 1 to keep $dir/merged after parity check
#   SKIP_BF16_EVAL    set to 1 to skip BF16 dev quantisation-loss eval
#   FORCE_RETRAIN     set to 1 to retrain from scratch after a prior divergence
#
# Exit: 0 done, 1 failed, 2 missing config/data, 3 parity FAIL, 4 diverged, 5 budget exceeded.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ml/pod/env.sh

PY="$VENV_TRAIN/bin/python"
RUN="${RUN:-girgur-q2-lora}"
CFG="${CFG:-ml/train/configs/$RUN.yaml}"
GGUF_NAME="${GGUF_NAME:-girgur-cleanup-qwen3.5-2b}"
QUANTS="${QUANTS:-Q4_0}"
PARITY_N="${PARITY_N:-24}"
PARITY_MIN_AGREE="${PARITY_MIN_AGREE:-0.8}"

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -f "$W/data/girgur/train.jsonl" ]]; then
    DATA_DIR="$W/data/girgur"
  else
    DATA_DIR="$MURMUR_DATA"
  fi
fi
export DATA_DIR

dir="$MURMUR_RUNS/$RUN"
[[ -f "$CFG" ]] || { log "no config $CFG"; exit 2; }
for split in train dev test; do
  [[ -f "$DATA_DIR/$split.jsonl" ]] || { log "missing $DATA_DIR/$split.jsonl (sync .cache/ml/data/girgur first)"; exit 2; }
done

step_check() { budget_check "$1" "$2" hard || exit 5; }

# ---- 1. Preflight dataset summary
log "Girgur preflight: DATA_DIR=$DATA_DIR RUN=$RUN GGUF_NAME=$GGUF_NAME"
"$PY" - "$DATA_DIR" <<'PY' | tee -a "$MURMUR_LOGS/pod.log" >&2
import json, sys
from collections import Counter
from pathlib import Path
d = Path(sys.argv[1])
for sp in ("train", "dev", "test"):
    rows = [json.loads(l) for l in (d / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_lang = dict(sorted(Counter(r["lang"] for r in rows).items()))
    print(f"  {sp}: {len(rows)} rows {by_lang}")
PY

SMOKE="$DATA_DIR/smoke.v2.test.jsonl"
[[ -f "$SMOKE" ]] || "$PY" -m ml.eval.smoke_to_contract --split test --out "$SMOKE" >/dev/null
read -r -a DEV_SETS <<<"${EVAL_DEV:-$DATA_DIR/dev.jsonl prompts/eval/sets/dev.v2.jsonl $SMOKE}"
read -r -a TEST_SETS <<<"${EVAL_TEST:-$DATA_DIR/test.jsonl prompts/eval/sets/test.v2.jsonl}"

# ---- 2. Train LoRA
diverged_why() { jq -r '"step \(.step // "?"): \(.reason // "?")"' "$dir/DIVERGED.json" 2>/dev/null || echo "unreadable marker"; }
if [[ -f "$dir/train_info.json" ]]; then
  log "train $RUN: done, skip"
else
  if [[ -f "$dir/DIVERGED.json" && "${FORCE_RETRAIN:-}" != 1 ]]; then
    log "train $RUN: DIVERGED in an earlier attempt ($(diverged_why)); skipped. Pass FORCE_RETRAIN=1 to retrain."
    exit 4
  fi
  step_check "train $RUN" "${RUN_RESERVE_S:-1500}"
  after_s=$(( $(median_seconds merge: 60) + $(median_seconds export: 120) + $(median_seconds parity: 90) + $(median_seconds eval: 360) * 2 ))
  export MURMUR_TRAIN_DEADLINE
  MURMUR_TRAIN_DEADLINE=$(budget_deadline "$after_s")
  log "train $RUN (deadline in $(( MURMUR_TRAIN_DEADLINE - $(date +%s) ))s, reserving ${after_s}s for merge/export/parity/eval)"
  rc=0
  timed "train:$RUN" "$PY" -m ml.train.sft --config "$CFG" --set "output_dir=$dir" "$@" \
    >>"$MURMUR_LOGS/train-$RUN.log" 2>&1 || rc=$?
  if (( rc == 4 )) && [[ -f "$dir/DIVERGED.json" ]]; then
    log "train $RUN DIVERGED ($(diverged_why); $dir/DIVERGED.json)"
    exit 4
  fi
  (( rc == 0 )) || { log "train $RUN FAILED (rc $rc, see $MURMUR_LOGS/train-$RUN.log)"; exit 1; }
  jq -c '{run: .run_name, tokens_per_s: .approx_tokens_per_s, train_s: .train_runtime_s,
          peak_gb: .peak_cuda_mem_gb, steps, planned_steps, stopped_by}' "$dir/train_info.json" >&2
fi

method=$(jq -r .method "$dir/train_info.json")
FTSYS=$(jq -r '.prompt.system // "finetuned.v2"' "$dir/config.resolved.json")
BF16_GGUF="$dir/gguf/$GGUF_NAME-BF16.gguf"
Q4_GGUF="$dir/gguf/$GGUF_NAME-Q4_0.gguf"

# ---- 3 & 4. Merge + Export BF16 & Q4_0 GGUF
parity_ok() { [[ -f "$dir/eval/parity.json" && "$(jq -r .status "$dir/eval/parity.json")" == PASS ]]; }
if [[ ! -f "$Q4_GGUF" || ! -f "$dir/gguf/manifest.json" ]] || { (( PARITY_N > 0 )) && ! parity_ok && [[ ! -f "$BF16_GGUF" ]]; }; then
  step_check "export $RUN" "$(median_seconds export: 120)"
  if [[ "$method" != full && ! -f "$dir/merged/config.json" ]]; then
    log "merge $RUN"
    timed "merge:$RUN" "$PY" -m ml.train.merge --run "$dir" >>"$MURMUR_LOGS/train-$RUN.log" 2>&1
  fi
  log "export $RUN -> $GGUF_NAME ($QUANTS + BF16)"
  # shellcheck disable=SC2086
  timed "export:$RUN" "$PY" -m ml.train.export --run "$dir" --name "$GGUF_NAME" --quants $QUANTS --keep-bf16 \
    >>"$MURMUR_LOGS/export-$RUN.log" 2>&1
fi

# ---- 5. Parity check: merged HF BF16 vs BF16 GGUF
if (( PARITY_N > 0 )) && ! parity_ok; then
  if [[ "$method" != full && ! -f "$dir/merged/config.json" ]]; then
    log "re-merge $RUN for parity"
    timed "merge:$RUN" "$PY" -m ml.train.merge --run "$dir" >>"$MURMUR_LOGS/train-$RUN.log" 2>&1
  fi
  step_check "parity $RUN" "$(median_seconds parity: 90)"
  log "parity $RUN: merged HF vs $GGUF_NAME-BF16.gguf ($PARITY_N dev items)"
  if ! timed "parity:$RUN" "$PY" -m ml.train.parity --run "$dir" --gguf "$BF16_GGUF" --data "$DATA_DIR/dev.jsonl" \
      -n "$PARITY_N" --min-agree "$PARITY_MIN_AGREE" --system "$FTSYS" \
      --llama-bin "$MURMUR_LLAMA_BIN" --ngl "$MURMUR_LLAMA_NGL" >>"$MURMUR_LOGS/export-$RUN.log" 2>&1; then
    log "PARITY FAILED for $RUN ($dir/eval/parity.json); stopping before eval"
    exit 3
  fi
fi
[[ -n "${KEEP_MERGED:-}" ]] || rm -rf "$dir/merged"

# ---- 6. Eval Q4_0 on Dev (RU+EN regression + Hebrew)
if [[ ! -f "$dir/eval/Q4_0-dev/report.json" ]]; then
  step_check "eval $RUN Q4_0 dev" "$(median_seconds eval: 360)"
  log "eval $GGUF_NAME-Q4_0 on dev (${DEV_SETS[*]})"
  timed "eval:$RUN:Q4_0-dev" "$PY" -m ml.eval.run_eval --model "$Q4_GGUF" --data "${DEV_SETS[@]}" \
    --out "$dir/eval/Q4_0-dev" --system "$FTSYS" --label "$GGUF_NAME Q4_0 dev" \
    --llama-bin "$MURMUR_LLAMA_BIN" --ngl "$MURMUR_LLAMA_NGL" \
    >>"$MURMUR_LOGS/eval-$RUN.log" 2>&1
fi

# ---- 7. Eval Q4_0 on Test (RU+EN regression + Hebrew)
if [[ ! -f "$dir/eval/Q4_0-test/report.json" ]]; then
  step_check "eval $RUN Q4_0 test" "$(median_seconds eval: 360)"
  log "eval $GGUF_NAME-Q4_0 on test (${TEST_SETS[*]})"
  timed "eval:$RUN:Q4_0-test" "$PY" -m ml.eval.run_eval --model "$Q4_GGUF" --data "${TEST_SETS[@]}" \
    --out "$dir/eval/Q4_0-test" --system "$FTSYS" --label "$GGUF_NAME Q4_0 test" \
    --llama-bin "$MURMUR_LLAMA_BIN" --ngl "$MURMUR_LLAMA_NGL" \
    >>"$MURMUR_LOGS/eval-$RUN.log" 2>&1
fi

# ---- 8. Optional BF16 dev eval for quantisation loss
if [[ "${SKIP_BF16_EVAL:-0}" != 1 && -f "$BF16_GGUF" && ! -f "$dir/eval/BF16-dev/report.json" ]]; then
  if budget_check "eval $RUN BF16 dev" "$(median_seconds eval: 360)" target; then
    log "eval $GGUF_NAME-BF16 on dev (quantisation loss check)"
    timed "eval:$RUN:BF16-dev" "$PY" -m ml.eval.run_eval --model "$BF16_GGUF" --data "${DEV_SETS[@]}" \
      --out "$dir/eval/BF16-dev" --system "$FTSYS" --label "$GGUF_NAME BF16 dev" \
      --llama-bin "$MURMUR_LLAMA_BIN" --ngl "$MURMUR_LLAMA_NGL" \
      >>"$MURMUR_LOGS/eval-$RUN.log" 2>&1 || log "BF16 dev eval failed (non-fatal)"
    if [[ -f "$dir/eval/BF16-dev/report.json" ]]; then
      "$PY" -m ml.eval.compare "$dir/eval/BF16-dev/report.json" "$dir/eval/Q4_0-dev/report.json" \
        --json "$dir/eval/quant_loss.json" --md "$dir/eval/quant_loss.md" >/dev/null || true
    fi
  fi
fi

# ---- 9. Print RU+EN regression & Hebrew breakdown summary
for split in dev test; do
  rep="$dir/eval/Q4_0-$split/report.json"
  [[ -f "$rep" ]] || continue
  jq -c --arg run "$RUN" --arg split "$split" '{
    run: $run, split: $split,
    verdict: .gates.verdict, guarded_verdict: .guarded.gates.verdict,
    overall: {n: .summary.overall.n, em: .summary.overall.em, nwer: .summary.overall.nwer,
              nwer_raw: .summary.overall.nwer_raw, noop: .summary.overall.noop_exact,
              wrong_lang: .summary.overall.wrong_language},
    by_lang: (.summary.by_lang | with_entries(.value = {n: .value.n, em: .value.em, nwer: .value.nwer,
              nwer_raw: .value.nwer_raw, wrong_lang: .value.wrong_language})),
    by_set_guarded_pass: (.guarded.summary.by_set // {} | map_values(.pass))
  }' "$rep" | tee -a "$MURMUR_LOGS/runs.jsonl"
done

log "Girgur pipeline complete! GGUF: $Q4_GGUF"
