#!/usr/bin/env bash
# Run ON THE DEV SERVER. Copies the code the pod needs (prompts/, ml/, docs/ml/) and, with WITH_DATA=1,
# the assembled training data (train/dev/test.jsonl + stats.json) to the pod volume. No weights, caches,
# credentials or teacher variables are ever sent. rsync is incremental: safe to re-run.
#   POD=root@1.2.3.4 POD_PORT=22022 bash ml/pod/push_code.sh
#   WITH_DATA=1 DATA_DIR=.cache/ml/data/final POD=... bash ml/pod/push_code.sh
set -euo pipefail
umask 077
cd "$(dirname "$0")/../.."
: "${POD:?set POD=user@host}"
PORT="${POD_PORT:-22}"; RW="${POD_WORKSPACE:-/workspace}"
SSH=(ssh -p "$PORT" ${POD_KEY:+-i "$POD_KEY"} -o StrictHostKeyChecking=accept-new)
"${SSH[@]}" "$POD" "umask 077; mkdir -p $RW/murmur $RW/data; chmod 700 $RW/data"
rsync -az --delete -e "${SSH[*]}" --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='*.pyc' \
  --exclude='ml/roundtrip/samples/' --relative prompts ml docs/ml "$POD:$RW/murmur/"
if [[ -n "${WITH_DATA:-}" ]]; then
  D="${DATA_DIR:?set DATA_DIR to the assembled data dir (ml.train.assemble --out)}"
  for f in train.jsonl dev.jsonl test.jsonl; do [[ -f "$D/$f" ]] || { echo "missing $D/$f"; exit 1; }; done
  files=("$D/train.jsonl" "$D/dev.jsonl" "$D/test.jsonl")
  [[ ! -f "$D/stats.json" ]] || files+=("$D/stats.json")
  rsync -az --chmod=Fu=rw,Fgo= -e "${SSH[*]}" "${files[@]}" "$POD:$RW/data/"
fi
echo "pushed to $POD:$RW"
