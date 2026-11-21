#!/usr/bin/env bash
# Run ON THE DEV SERVER: pull results from the pod volume (the pod never needs credentials for the dev
# server). Small files by default: reports, predictions, cards, manifests, logs, timings, train_info.
#   POD=root@1.2.3.4 POD_PORT=22022 bash ml/pod/sync_results.sh
#   WITH_GGUF=1     + finalists' Q4_0 GGUFs (for the CPU ship-gate eval and upload; ~0.5-1.3 GB each)
#   WITH_ADAPTERS=1 + LoRA/DoRA adapters and their tokenizers, WITH_FULL=1 + full-FT final checkpoints
#   WITH_PROBE=1    + the round-trip probe outputs under $W/roundtrip (owned by ml/roundtrip)
# Destination: .cache/ml/train/pod-results/ (gitignored). Incremental: safe to re-run.
set -euo pipefail
umask 077
cd "$(dirname "$0")/../.."
: "${POD:?set POD=user@host}"
PORT="${POD_PORT:-22}"; RW="${POD_WORKSPACE:-/workspace}"
SSH="ssh -p $PORT ${POD_KEY:+-i $POD_KEY} -o StrictHostKeyChecking=accept-new"
DST="${DST:-.cache/ml/train/pod-results}"; mkdir -p "$DST"
[[ ! -L "$DST" && $(stat -c %a "$DST") == 700 ]] || { echo "Use a dedicated mode-700 results directory: $DST" >&2; exit 2; }
# rsync: first matching rule wins -> big-dir excludes, then includes, then exclude everything else
exc=(--exclude='checkpoint-*/' --exclude='merged/' --exclude='hf/' --exclude='stock/' --exclude='train.jsonl')
if [[ ${WITH_FULL:-0} != 1 && ${WITH_ADAPTERS:-0} != 1 ]]; then
  exc+=(--exclude='final/')
elif [[ ${WITH_FULL:-0} != 1 ]]; then
  exc+=(--exclude='final/*.safetensors')
elif [[ ${WITH_ADAPTERS:-0} != 1 ]]; then
  exc+=(--exclude='final/adapter/')
fi
inc=(--include='*/' --include='*.json' --include='*.jsonl' --include='*.md' --include='*.log' --include='*.txt'
     --include='*.yaml' --include='*.jinja' --include='*.model' --include='LICENSE')
finalists=()
if [[ ${WITH_GGUF:-0} == 1 ]]; then
  receipt=$($SSH "$POD" "cat $RW/runs/finalists.txt")
  while read -r base run extra; do
    [[ -n "$base" && "$run" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ && -z "$extra" ]] || { echo "Invalid finalist receipt" >&2; exit 2; }
    finalists+=("$run")
    inc+=(--include="/runs/$run/gguf/*-Q4_0.gguf")
  done <<<"$receipt"
fi
[[ ${WITH_ADAPTERS:-0} != 1 ]] || inc+=(--include='final/adapter/**')
[[ ${WITH_FULL:-0} != 1 ]] || inc+=(--include='final/*.safetensors')
perms=(--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=)
rsync -az --no-links "${perms[@]}" --prune-empty-dirs -e "$SSH" "${exc[@]}" "${inc[@]}" --exclude='*' \
  "$POD:$RW/runs" "$POD:$RW/eval" "$POD:$RW/logs" "$POD:$RW/data" "$DST/"
if ((${#finalists[@]})); then
  python3 -m ml.train.verify_retained "$DST" "${finalists[@]}"
fi
[[ ${WITH_PROBE:-0} != 1 ]] || rsync -az --no-links "${perms[@]}" -e "$SSH" "$POD:$RW/roundtrip" "$DST/"
du -sh "$DST"
