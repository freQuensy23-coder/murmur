#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")/../.."
: "${CPU_RUN_DIR:?set a new local work directory}"
: "${CPU_DATA_DIR:?set the local assembled dataset directory}"
: "${CPU_TRAIN_VENV:?set the CPU training environment directory}"
: "${CPU_LLAMA_BIN:?set the CPU llama.cpp binary path}"
: "${CPU_LLAMA_SOURCE:?set the llama.cpp source checkout}"
: "${CPU_BASE_MODEL:?set the local base model directory}"
run=${1:-q08-lora}
[[ $run == q08-lora || $run == lfm3-r16 ]]
export MURMUR_WORKSPACE=$(realpath -m "$CPU_RUN_DIR") MURMUR_REPO=$PWD
mkdir "$MURMUR_WORKSPACE"
w=$MURMUR_WORKSPACE
export CONFIG_DIR=$w/cfg MURMUR_DATA=$w/data MURMUR_LLAMA_BIN=$(realpath "$CPU_LLAMA_BIN") MURMUR_LLAMA_NGL=0
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 OMP_NUM_THREADS=4 MURMUR_THREADS=4
export MURMUR_SESSION_START=$(date +%s) BUDGET_SESSION_OVERHEAD_S=0 PARITY_N=4 MURMUR_CONTAINER_MARKERS=$w/ctr
export EVAL_DEV=$w/small/dev.jsonl EVAL_TEST=$w/small/test.jsonl STOCK_ON_POD= STOCK_OURS=
mkdir -p "$CONFIG_DIR" "$MURMUR_DATA" "$w/small" "$w/llama"
ln -s "$(realpath "$CPU_TRAIN_VENV")" "$w/venv-train"
ln -s "$(realpath "$CPU_LLAMA_SOURCE")" "$w/llama/llama.cpp-${MURMUR_LLAMA_TAG:-b12503}"
for split in train dev test; do
    cp "$CPU_DATA_DIR/$split.jsonl" "$MURMUR_DATA/$split.jsonl"
    if [[ $split != train ]]; then head -n 6 "$MURMUR_DATA/$split.jsonl" > "$w/small/$split.jsonl"; fi
done
"$w/venv-train/bin/python" - "$CONFIG_DIR/$run.yaml" "ml/train/configs/$run.yaml" "$CPU_BASE_MODEL" <<'PY'
import sys
from pathlib import Path
import yaml
out, parent, base = map(Path, sys.argv[1:])
if not base.is_dir():
    raise SystemExit('the CPU base model must be local')
out.write_text(yaml.safe_dump(dict(extends=str(parent.resolve()), base_model=str(base.resolve()),
    train=dict(max_steps=2, batch_size=1, save_steps=2, eval_steps=100, threads=4,
               workers=0, optim='adamw_torch', gradient_checkpointing=True))))
PY
bash ml/pod/run_one.sh "$run"
bash ml/pod/export_eval.sh
