#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")/../.."
: "${MURMUR_WORKSPACE:?set the workload directory}"
: "${MURMUR_CUTOFF_EPOCH:?set the workload cutoff as UTC epoch seconds, allowing time to sync and stop}"
[[ $MURMUR_CUTOFF_EPOCH =~ ^[1-9][0-9]{0,9}$ ]] || exit 2
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export STAGES="${1:-bootstrap train eval}"
mkdir -p -- "$MURMUR_WORKSPACE"
exec 9>"$MURMUR_WORKSPACE/.murmur-start.lock"
flock -n 9 || { printf 'EXIT 2: another launcher holds the lock\n'; exit 2; }
remaining=$(( MURMUR_CUTOFF_EPOCH - $(date +%s) ))
((remaining > 0)) || { printf 'EXIT 5: reached workload cutoff\n'; exit 5; }
rc=0
timeout --kill-after=30s "$remaining" bash ml/pod/run_all.sh || rc=$?
printf 'EXIT %d\n' "$rc"
exit "$rc"
