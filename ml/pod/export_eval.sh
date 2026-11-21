#!/usr/bin/env bash
# After train_matrix.sh: baselines + finalists, with ONE eval implementation (ml/eval/run_eval.py),
# llama.cpp CUDA on the pod (greedy). Idempotent: every eval dir with a report.json is skipped.
#   1. stock baselines (B2), only with STOCK_ON_POD=1 (default: they run free on the dev server CPU in
#      ml/pod/after_pod.sh, the vendor GGUFs are already there): vendor Q4_0 GGUFs with the
#      prompt profile the app ships for stock models (default.v1, ADR-085) and the short fine-tune
#      prompt; every report also holds the after-guard (prompts/guard.py) measurement, which is what
#      the app inserts for a stock model; STOCK_OURS=1 adds the stock weights
#      through OUR conversion (plain Q4_0: isolates the fine-tuning effect from vendor imatrix/QAD)
#   2. finalists = best run per base model on the dev Q4_0 eval (ml.train.select):
#      BF16 GGUF on dev (-> measured quantisation loss), Q4_0 on the test sets, Q4_0 with the stock
#      default.v1 prompt (robustness: the prompt a user gets when switching back), model card + catalog entry
#   3. LEGACY=1 only: finalists' Q4_0 on the v1-context sets (dev.v1 + test.v1), labelled "legacy", in a
#      separate leaderboard table; never used by select.py, the gates or the cards
#   4. leaderboard over everything + vs_stock.md (ml.eval.baseline table): every report per eval set
#      (dev.v2, test.v2, the dataset's dev/test) next to the frozen stock numbers (pass raw / guarded)
# Budget: every eval is checked against MAX_GPU_HOURS before it starts (ml/pod/env.sh budget_check).
# The judge (teacher, dev server) and the CPU ship-gate eval run on the dev server afterwards:
# ml/pod/after_pod.sh. Env: EVAL_DEV, EVAL_TEST, PROFILES (stock-model profiles, default
# "default.v1", the first one is the card's comparison), HF_OWNER (for the card's repo id).
set -euo pipefail
cd "$(dirname "$0")/../.."
source ml/pod/env.sh
PY="$VENV_TRAIN/bin/python"
SMOKE="$MURMUR_DATA/smoke.v2.test.jsonl"
[[ -f "$SMOKE" ]] || "$PY" -m ml.eval.smoke_to_contract --split test --out "$SMOKE" >/dev/null
read -r -a DEV <<<"${EVAL_DEV:-$MURMUR_DATA/dev.jsonl prompts/eval/sets/dev.v2.jsonl $SMOKE}"
read -r -a TEST <<<"${EVAL_TEST:-$MURMUR_DATA/test.jsonl prompts/eval/sets/test.v2.jsonl}"
read -r -a LEGACY_SETS <<<"${EVAL_LEGACY:-prompts/eval/sets/dev.v1.jsonl prompts/eval/sets/test.v1.jsonl}"
PROFILES="${PROFILES:-default.v1}"
FT_SYSTEM="${FT_SYSTEM:-finetuned.v2}"   # = prompt.system of ml/train/configs/base.yaml (stock models with the FT prompt)
STOCK_P="${PROFILES%% *}"

evaluate() {  # $1=model file  $2=out dir  $3=system/profile  $4=label  $5=dev|test|legacy
  [[ -f "$2/report.json" ]] && { log "eval $4: done, skip"; return 0; }
  local sets=("${DEV[@]}"); [[ "$5" == test ]] && sets=("${TEST[@]}"); [[ "$5" == legacy ]] && sets=("${LEGACY_SETS[@]}")
  budget_check "eval $4" "$(median_seconds eval: 300)" hard || return 0
  [[ -f "prompts/profiles/$3.json" || -f "prompts/system/$3.md" || -f "prompts/system/$3.v1.md" ]] \
    || { log "eval $4: profile $3 not found, skip"; return 0; }
  log "eval $4 ($3, $5)"
  timed "eval:$4" "$PY" -m ml.eval.run_eval --model "$1" --data "${sets[@]}" --out "$2" --system "$3" --label "$4" \
    --llama-bin "$MURMUR_LLAMA_BIN" --ngl "$MURMUR_LLAMA_NGL" >>"$MURMUR_LOGS/eval-final.log" 2>&1 \
    || log "eval $4 FAILED (logs/eval-final.log); predictions so far are kept, re-run resumes it"
}

# ---- 1. stock baselines (optional on the pod)
if [[ -n "${STOCK_ON_POD:-}" ]]; then
mkdir -p "$W/stock"
dl() {
  local dest="$W/stock/$3"
  if ! printf '%s  %s\n' "$4" "$dest" | sha256sum -c --status; then
    curl -fL --retry 5 -o "$dest.part" "https://huggingface.co/$1/resolve/$2/$3"
    printf '%s  %s\n' "$4" "$dest.part" | sha256sum -c --status || { log "stock checksum mismatch: $3"; return 1; }
    mv "$dest.part" "$dest"
  fi
}
dl LiquidAI/LFM3-1.7B-GGUF cd0c74c575b5c416f3011b5b4c298077f26ca8c2 LFM3-1.7B-QAD-Q4_0.gguf \
  43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452
dl unsloth/Qwen3.5-0.8B-GGUF 6ab461498e2023f6e3c1baea90a8f0fe38ab64d0 Qwen3.5-0.8B-Q4_0.gguf \
  444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c
dl unsloth/Qwen3.5-2B-GGUF f6d5376be1edb4d416d56da11e5397a961aca8ae Qwen3.5-2B-Q4_0.gguf \
  cd70221bebaee0503e0f6717e174250cd7825aa88438b3aabec9ad55731d9bb1
for f in LFM3-1.7B-QAD-Q4_0 Qwen3.5-0.8B-Q4_0 Qwen3.5-2B-Q4_0; do
  for p in $PROFILES $FT_SYSTEM; do
    for s in dev test; do evaluate "$W/stock/$f.gguf" "$MURMUR_EVAL/stock/$f/$p-$s" "$p" "stock $f $p $s" "$s"; done
  done
done
fi
for base in ${STOCK_OURS:+LiquidAI/LFM3-1.7B Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B}; do
  sname="stock-$(basename "$base" | tr '[:upper:]' '[:lower:]')"; sdir="$W/stock/$sname"
  if [[ ! -f "$sdir/manifest.json" ]]; then
    rev=$("$PY" -m ml.train.config revision "$base")   # pinned commit (ml/train/configs/base.yaml base_revisions)
    snap=$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$base', revision='$rev'))")
    timed "export:$sname" "$PY" -m ml.train.export --hf "$snap" --name "$sname" --out "$sdir" --quants Q4_0 \
      >>"$MURMUR_LOGS/export-$sname.log" 2>&1
  fi
  evaluate "$sdir/$sname-Q4_0.gguf" "$MURMUR_EVAL/stock/$sname/$STOCK_P-dev" "$STOCK_P" "$sname ours-Q4_0 $STOCK_P dev" dev
done

# ---- 2. finalists: best run per base model
declare -A FAM=()
finalists=()
for d in "$MURMUR_RUNS"/*/train_info.json; do
  [[ -f "$d" ]] || continue
  r=$(basename "$(dirname "$d")"); b=$(jq -r .base_model "$d"); FAM[$b]+="$r "
done
for b in "${!FAM[@]}"; do
  # shellcheck disable=SC2086
  run=$("$PY" -m ml.train.select --runs-dir "$MURMUR_RUNS" ${FAM[$b]} | sed -n 1p) || continue
  [[ -n "$run" ]] || continue
  finalists+=("$b $run")
done
if ((${#finalists[@]} == 0)); then
  rm -f "$MURMUR_RUNS/finalists.txt"
  log 'no finalist has a scored dev run'
  exit 2
fi
printf '%s\n' "${finalists[@]}" | sort >"$MURMUR_RUNS/finalists.txt.new"
mv "$MURMUR_RUNS/finalists.txt.new" "$MURMUR_RUNS/finalists.txt"
[[ ${RETAIN_ONLY:-0} != 1 ]] || exit 0
while read -r b run; do
  log "finalist for $b: $run"
  dir="$MURMUR_RUNS/$run"; name=$(jq -r .name "$dir/gguf/manifest.json")
  ftsys=$(jq -r '.prompt.system // "finetuned.v2"' "$dir/config.resolved.json")
  evaluate "$dir/gguf/$name-BF16.gguf" "$dir/eval/BF16-dev" "$ftsys" "$run BF16 dev" dev
  evaluate "$dir/gguf/$name-Q4_0.gguf" "$dir/eval/Q4_0-test" "$ftsys" "$run Q4_0 test" test
  evaluate "$dir/gguf/$name-Q4_0.gguf" "$dir/eval/Q4_0-dev-$STOCK_P" "$STOCK_P" "$run Q4_0 dev $STOCK_P prompt" dev
  [[ -f "$dir/eval/BF16-dev/report.json" ]] && "$PY" -m ml.eval.compare "$dir/eval/BF16-dev/report.json" \
    "$dir/eval/Q4_0-dev/report.json" --json "$dir/eval/quant_loss.json" --md "$dir/eval/quant_loss.md" >/dev/null
  [[ -f "$dir/eval/Q4_0-test/report.json" ]] || { log "no Q4_0 test eval for $run (budget?): no card"; continue; }
  case "$b" in *LFM3*) st="LFM3-1.7B-QAD-Q4_0";; *0.8B*) st="Qwen3.5-0.8B-Q4_0";; *) st="Qwen3.5-2B-Q4_0";; esac
  "$PY" -m ml.train.card --run "$dir" --name "$name" --repo "${HF_OWNER:-OWNER}/$name-gguf" \
    --eval "$dir/eval/Q4_0-test/report.json" \
    $([[ -f "$dir/eval/quant_loss.json" ]] && echo --qloss-json "$dir/eval/quant_loss.json") \
    $([[ -f "$MURMUR_EVAL/stock/$st/$STOCK_P-test/report.json" ]] && echo --stock-eval "$MURMUR_EVAL/stock/$st/$STOCK_P-test/report.json") \
    >/dev/null
  [[ -z "${LEGACY:-}" ]] || evaluate "$dir/gguf/$name-Q4_0.gguf" "$dir/eval/legacy-v1-Q4_0" "$ftsys" "$run Q4_0 legacy-v1" legacy
done <"$MURMUR_RUNS/finalists.txt"

# ---- 4. leaderboard
"$PY" -m ml.eval.leaderboard "$MURMUR_EVAL/stock" "$MURMUR_RUNS" --out "$MURMUR_EVAL/leaderboard.md"
"$PY" -m ml.eval.baseline table "$MURMUR_EVAL/stock" "$MURMUR_RUNS" --out "$MURMUR_EVAL/vs_stock.md" >/dev/null
log "export_eval finished; leaderboard: $MURMUR_EVAL/leaderboard.md"
