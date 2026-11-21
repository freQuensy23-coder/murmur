# Common environment for every ml/pod script. Source it: `source ml/pod/env.sh`.
# PLAN_V2 pod: 1x RTX 5090, stopped between stages. The container disk is wiped on stop; only the
# volume ($W, /workspace) persists. So EVERYTHING that must survive a stop lives under $W, including
# uv, uv's managed Python (a venv symlinks its interpreter: if the interpreter were on the container
# disk the venv would break after a stop) and the uv / HF caches.
# No teacher and no teacher credentials on the pod (the teacher runs on the dev server).
umask 077
export W="${MURMUR_WORKSPACE:-/workspace}"
export REPO="${MURMUR_REPO:-$W/murmur}"
export LLAMA_TAG="${MURMUR_LLAMA_TAG:-b12503}"
export LLAMA_SRC="$W/llama/llama.cpp-$LLAMA_TAG"            # convert_hf_to_gguf.py + gguf-py
export LLAMA_BIN_CPU="$W/llama/cpu/llama-$LLAMA_TAG"         # CPU build (same as the dev server)
# CUDA build for pod evals: the release CUDA 12.8 build, unless bootstrap's kernel smoke test (step 6b,
# ml/train/llama_smoke.py) picked another one (CUDA 13.5 release build or a source build) and wrote its dir here
export LLAMA_BIN_CUDA="$W/llama/cuda/llama-$LLAMA_TAG"
[[ -s "$W/.bootstrap/llama-cuda-bin" ]] && LLAMA_BIN_CUDA=$(<"$W/.bootstrap/llama-cuda-bin")
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$W/.nv/ComputeCache}"   # PTX JIT results survive a pod stop (volume)
export VENV_TRAIN="$W/venv-train"
export UV_INSTALL_DIR="$W/.uv/bin"
export UV_PYTHON_INSTALL_DIR="$W/.uv/python"
export UV_CACHE_DIR="$W/.uv/cache"
export UV_LINK_MODE=copy
export PATH="$UV_INSTALL_DIR:$PATH"
export HF_HOME="$W/hf"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MURMUR_DATA="${MURMUR_DATA:-$W/data}"
export MURMUR_RUNS="${MURMUR_RUNS:-$W/runs}"
export MURMUR_EVAL="${MURMUR_EVAL:-$W/eval}"
export MURMUR_LOGS="${MURMUR_LOGS:-$W/logs}"
export MURMUR_LLAMA_SRC="$LLAMA_SRC"
# llama.cpp for pod evals: the CUDA build, unless bootstrap found no CUDA build that computes correctly on the
# GPU. Values this file chose itself are re-derived on every source (a stage started by run_all.sh after
# bootstrap must see bootstrap's markers, not the values run_all.sh inherited); explicit ones are kept.
if [[ -f "$W/.bootstrap/llama-cpu-fallback" ]]; then _lb="$LLAMA_BIN_CPU"; _ngl=0; else _lb="$LLAMA_BIN_CUDA"; _ngl=99; fi
if [[ -z "${MURMUR_LLAMA_BIN:-}" || -n "${_MURMUR_LLAMA_BIN_AUTO:-}" ]]; then
  export MURMUR_LLAMA_BIN="$_lb" _MURMUR_LLAMA_BIN_AUTO=1; fi
if [[ -z "${MURMUR_LLAMA_NGL:-}" || -n "${_MURMUR_LLAMA_NGL_AUTO:-}" ]]; then
  export MURMUR_LLAMA_NGL="$_ngl" _MURMUR_LLAMA_NGL_AUTO=1; fi
export MURMUR_THREADS="${MURMUR_THREADS:-$(( $(nproc) > 16 ? 16 : $(nproc) ))}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# markers: volume-backed steps persist; container-disk steps (apt) are re-checked after every start
export MARK_VOL="$W/.bootstrap"
export MARK_CTR="${MURMUR_CONTAINER_MARKERS:-/var/tmp/murmur-bootstrap}"
mkdir -p "$MURMUR_DATA" "$MURMUR_RUNS" "$MURMUR_EVAL" "$MURMUR_LOGS" "$HF_HOME" "$MARK_VOL" "$MARK_CTR"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$MURMUR_LOGS/pod.log" >&2; }
# timed STAGE: appends {"stage","seconds","rc","utc"} to $MURMUR_LOGS/timings.jsonl (the time budget
# in ml/pod/REQUEST.md is re-measured from this file after every pod session)
timed() {
  local name=$1; shift; local t0 rc=0; t0=$(date +%s)
  "$@" || rc=$?
  printf '{"stage": "%s", "seconds": %d, "rc": %d, "utc": "%s", "gpu": "%s"}\n' "$name" $(( $(date +%s) - t0 )) "$rc" \
    "$(date -u +%FT%TZ)" "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" >>"$MURMUR_LOGS/timings.jsonl"
  return $rc
}

# ---- GPU-hour budget, enforced where the time is spent (run_all per stage, train_matrix per run,
# run_one per step, export_eval per eval). Billed time = sum over pod sessions of (last check - container
# start), container start = now - elapsed time of PID 1 (counts setup and idle time, not only stages),
# plus BUDGET_SESSION_OVERHEAD_S per session for provisioning / image pull before the container runs.
# Persisted on the volume in $W/.budget/sessions.tsv ("start<TAB>last" per session).
#   MAX_GPU_HOURS     hard cap (owner: 4): nothing starts that would end past it, and a training that is
#                     already running stops at MURMUR_TRAIN_DEADLINE (budget_deadline, ml/train/sft.py StopRules)
#   TARGET_GPU_HOURS  plan (3): optional runs (sweep extras, DoRA, full FT) start only if they fit
#   BUDGET_TAIL_S     kept back from BOTH limits (default 900 s = 15 min) for what happens after the scripts
#                     end: the operator's next poll (every ~5 min), sync_results.sh WITH_GGUF=1 (~3 GB, ~3 min)
#                     and stopping the pod. So the scripts end by 3 h 45 at the latest and the last sync and
#                     the stop still land inside the 4 h cap (test_budget_tail_keeps_sync_and_stop_inside_cap).
# MURMUR_SESSION_START (epoch seconds) overrides the container start (local simulation, odd runtimes).
export MAX_GPU_HOURS="${MAX_GPU_HOURS:-4}" TARGET_GPU_HOURS="${TARGET_GPU_HOURS:-3}"
export BUDGET_SESSION_OVERHEAD_S="${BUDGET_SESSION_OVERHEAD_S:-300}" BUDGET_TAIL_S="${BUDGET_TAIL_S:-900}"
export BUDGET_DIR="$W/.budget"; mkdir -p "$BUDGET_DIR"
_session_start() {
  if [[ -n "${MURMUR_SESSION_START:-}" ]]; then echo "$MURMUR_SESSION_START"; return; fi
  local now et; now=$(date +%s); et=$(ps -o etimes= -p 1 2>/dev/null | tr -d ' ')
  if [[ -n "$et" && "$et" -lt 43200 ]]; then echo $(( now - et )); return; fi
  # PID 1 older than 12 h: not a fresh pod container; fall back to the first check seen on this boot
  local boot; boot=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo noboot)
  [[ -f "$BUDGET_DIR/boot-$boot" ]] || echo "$now" >"$BUDGET_DIR/boot-$boot"
  cat "$BUDGET_DIR/boot-$boot"
}
# prints the billed seconds so far (and records this session's heartbeat)
budget_used() {
  local now st; now=$(date +%s); st=$(_session_start)
  touch "$BUDGET_DIR/sessions.tsv"
  awk -v st="$st" -v now="$now" -v ov="$BUDGET_SESSION_OVERHEAD_S" -v f="$BUDGET_DIR/sessions.tsv.new" '
    BEGIN { OFS = "\t" }
    { if ($1 - st < 90 && st - $1 < 90) { $2 = now; hit = 1 } print > f; used += $2 - $1 + ov }
    END { if (!hit) { print st, now > f; used += now - st + ov } print used }' "$BUDGET_DIR/sessions.tsv"
  mv "$BUDGET_DIR/sessions.tsv.new" "$BUDGET_DIR/sessions.tsv"
}
# budget_limit [hard|target]: seconds the scripts may use = the limit minus BUDGET_TAIL_S
budget_limit() {
  local h=$MAX_GPU_HOURS; [[ "${1:-hard}" == target ]] && h=$TARGET_GPU_HOURS
  awk -v h="$h" -v t="$BUDGET_TAIL_S" 'BEGIN{print int(h*3600) - int(t)}'
}
# budget_check WHAT RESERVE_S [hard|target]: rc 5 (and a log line) if WHAT would end past the limit
budget_check() {
  local what=$1 reserve=${2:-0} kind=${3:-hard} used lim
  used=$(budget_used); lim=$(budget_limit "$kind")
  if (( used + reserve > lim )); then
    log "budget: $what needs ~${reserve}s, used ${used}s, $kind limit ${lim}s (${BUDGET_TAIL_S}s kept for sync + stop): NOT started"
    return 5
  fi
  log "budget: used ${used}s of ${lim}s ($kind, ${BUDGET_TAIL_S}s kept for sync + stop); next $what (~${reserve}s)"
}
# budget_deadline AFTER_S: epoch second at which a training must stop so that AFTER_S more seconds (the rest
# of its run: merge, export, parity, dev eval) still end inside MAX_GPU_HOURS minus BUDGET_TAIL_S. Exported by
# run_one.sh as MURMUR_TRAIN_DEADLINE; ml/train/sft.py stops after the current optimizer step, saves and finishes.
budget_deadline() {
  local used lim; used=$(budget_used); lim=$(budget_limit hard)
  echo $(( $(date +%s) + lim - used - ${1:-0} ))
}
# median seconds of finished timed steps whose name starts with $1 (e.g. "eval:"), else $2
median_seconds() {
  [[ -f "$MURMUR_LOGS/timings.jsonl" ]] || { echo "$2"; return; }
  jq -s --arg p "$1" --argjson d "$2" '[.[] | select(.stage | startswith($p)) | select(.rc == 0) | .seconds] | sort
    | if length == 0 then $d else .[(length / 2 | floor)] end' "$MURMUR_LOGS/timings.jsonl"
}
