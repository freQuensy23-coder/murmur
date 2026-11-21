#!/usr/bin/env bash
# Pod entry point. The training session (docs/ml/TRAINING_RUNBOOK.md section 1, one session since the final
# data exists): STAGES="bootstrap train eval" after push_code.sh WITH_DATA=1, then sync_results.sh and stop.
# (STAGES="bootstrap probe" is the round-trip probe stage of the earlier two-session plan.)
# Every stage is idempotent: after any failure or stop, fix the cause and re-run the same command.
# Run inside tmux so an SSH drop does not kill it:
#   tmux new -s murmur "STAGES='bootstrap train eval' bash ml/pod/run_all.sh 2>&1 | tee -a /workspace/logs/run_all.log"
# Budget guard (ml/pod/env.sh budget_check): billed pod time since the container started, summed over
# sessions and persisted on the volume, is checked before each stage here AND before every run, step
# and eval inside the stages; MAX_GPU_HOURS (default 4, the owner's hard cap) is never crossed by
# starting new work, TARGET_GPU_HOURS (default 3) gates the optional runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ml/pod/env.sh
STAGES="${STAGES:-bootstrap train eval}"
retain_completed() {
  STOCK_ON_POD= STOCK_OURS= RETAIN_ONLY=1 bash ml/pod/export_eval.sh || log 'no new finalist receipt; sync diagnostics separately'
}
for s in $STAGES; do
  if ! budget_check "stage $s" 0 hard; then
    [[ "$s" != eval && "$s" != train ]] || retain_completed
    exit 5
  fi
  log "=== stage $s"
  case $s in
    bootstrap)  # base weights only when this session trains
      if [[ " $STAGES " == *" train "* ]]; then timed stage:bootstrap bash ml/pod/bootstrap.sh
      else MODELS="" timed stage:bootstrap bash ml/pod/bootstrap.sh; fi ;;
    probe) timed stage:probe bash ml/pod/probe.sh ;;
    train)
      [[ -f "$MURMUR_DATA/train.jsonl" && -f "$MURMUR_DATA/dev.jsonl" ]] \
        || { log "no $MURMUR_DATA/{train,dev}.jsonl: push it with ml/pod/push_code.sh WITH_DATA=1"; exit 2; }
      # a failed run (crash, parity FAIL) must not cost the finalists of the others: log it and go on;
      # budget refusal (5) and preflight abort (6) stop the session
      rc=0; timed stage:train bash ml/pod/train_matrix.sh || rc=$?
      if (( rc == 5 )); then retain_completed; exit 5; fi
      if (( rc == 6 )); then exit 6; fi
      (( rc == 0 )) || log "stage train: some runs failed (logs/run_all.log, logs/runs.jsonl); continuing with the finished ones" ;;
    eval) timed stage:eval bash ml/pod/export_eval.sh ;;
    *) log "unknown stage $s"; exit 2 ;;
  esac
done
_used=$(budget_used); _cap=$(awk "BEGIN{print int($MAX_GPU_HOURS*3600)}")
log "run_all done ($STAGES); billed pod time so far: ${_used}s; $(( _cap - _used ))s left until the ${MAX_GPU_HOURS} h cap for WITH_GGUF=1 sync_results.sh (~3 min) and the stop"
[[ -f "$MURMUR_EVAL/leaderboard.md" ]] && cat "$MURMUR_EVAL/leaderboard.md" || true
