#!/usr/bin/env bash
# From a clean CUDA image to a ready environment on the 1x RTX 5090 pod. Idempotent and cheap to
# re-run after every pod start: volume-backed steps (uv, Python, venv, llama.cpp, base weights) leave
# a marker in $W/.bootstrap and are skipped; container-disk steps (apt) are re-checked each start.
# Delete a marker to redo a step. Expects the code at $REPO (ml/pod/push_code.sh).
#   bash ml/pod/bootstrap.sh                        # everything the training stages need
#   MODELS="LiquidAI/LFM3-1.7B" bash ml/pod/bootstrap.sh      # MODELS="" skips the base weights
# Image: any Ubuntu 22.04/24.04 CUDA image (e.g. runpod/pytorch:*), host driver >= 580 (CUDA 13.x):
# torch 2.15.0 ships only CUDA 13.2 wheels (PyPI default; download.pytorch.org cu128 stops at 2.11).
set -euo pipefail
cd "$(dirname "$0")/../.."
source ml/pod/env.sh
step() { [[ -f "$2/$1" ]] && { log "skip $1"; return 1; }; log "run $1"; return 0; }

# ---- 0. GPU + driver (fail fast, before any download)
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1); driver $drv; CUDA ${cuda:-?}"
if [[ -z "$cuda" || "$cuda" -lt 130 ]]; then
  log "driver supports CUDA ${cuda:-?} < 13.0: torch 2.15.0 (CUDA 13.2 wheels) will not run. Ask for a host with driver >= 580."
  exit 3
fi

# ---- 1. system packages (container disk: re-checked after every start)
if step apt "$MARK_CTR"; then
  need=()
  for c in git curl rsync tmux jq; do command -v "$c" >/dev/null || need+=("$c"); done
  ldconfig -p | grep -q libgomp.so.1 || need+=(libgomp1)
  if ((${#need[@]})); then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q && apt-get install -y -q --no-install-recommends ca-certificates "${need[@]}"
  fi
  touch "$MARK_CTR/apt"
fi

# ---- 2. network: Python's socket.create_connection tries IPv6 first; on hosts with a black-holed
# IPv6 route huggingface_hub hangs (seen on the dev server). Prefer IPv4 if the default fails.
if step net "$MARK_CTR"; then
  if ! timeout 15 python3 -c "import socket;socket.create_connection(('huggingface.co',443),timeout=10)" 2>/dev/null; then
    grep -q '^precedence ::ffff:0:0/96 100' /etc/gai.conf 2>/dev/null || echo 'precedence ::ffff:0:0/96 100' >>/etc/gai.conf
    timeout 15 python3 -c "import socket;socket.create_connection(('huggingface.co',443),timeout=10)" \
      || { log "cannot reach huggingface.co"; exit 1; }
  fi
  touch "$MARK_CTR/net"
fi

# ---- 3. uv + Python 3.12 on the volume
if step uv "$MARK_VOL" || ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" INSTALLER_NO_MODIFY_PATH=1 sh
  uv python install 3.12
  touch "$MARK_VOL/uv"
fi

# ---- 4. llama.cpp b12503: source (convert_hf_to_gguf.py, gguf-py), CPU build, CUDA 12.8 build + cudart
if step llama "$MARK_VOL"; then
  mkdir -p "$W/llama/cpu/llama-$LLAMA_TAG" "$W/llama/cuda/llama-$LLAMA_TAG"
  base="https://github.com/ggml-org/llama.cpp"
  curl -fsSL "$base/archive/refs/tags/$LLAMA_TAG.tar.gz" | tar --no-same-owner -xz -C "$W/llama"
  for pair in "cpu:llama-$LLAMA_TAG-bin-ubuntu-x64.tar.gz" "cuda:llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64.tar.gz" \
              "cuda:cudart-llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64.tar.gz"; do
    # every release tarball has exactly one top-level dir (llama-<tag>/ or cudart-.../): strip it so
    # binaries, libs AND their .so symlinks (libllama-common.so.1 -> ...) land together in one dir
    curl -fsSL "$base/releases/download/$LLAMA_TAG/${pair#*:}" \
      | tar --no-same-owner -xz -C "$W/llama/${pair%%:*}/llama-$LLAMA_TAG" --strip-components=1
  done
  LD_LIBRARY_PATH="$LLAMA_BIN_CPU" "$LLAMA_BIN_CPU/llama-server" --version 2>&1 | tail -1
  LD_LIBRARY_PATH="$LLAMA_BIN_CUDA" "$LLAMA_BIN_CUDA/llama-server" --version 2>&1 | tail -2
  touch "$MARK_VOL/llama"
fi
# (whether the CUDA build really computes on this GPU is decided by the kernel smoke test, step 6b)
LD_LIBRARY_PATH="$W/llama/cuda/llama-$LLAMA_TAG" "$W/llama/cuda/llama-$LLAMA_TAG/llama-server" --list-devices 2>&1 \
  | grep CUDA0 || log "WARNING: the CUDA 12.8 llama.cpp build lists no CUDA device (step 6b tries the fallbacks)"

# ---- 5. training venv: torch 2.15.0 (CUDA 13.2, PyPI default) + pinned stack + GPU kernels
if step venv-train "$MARK_VOL"; then
  uv venv -q -p 3.12 "$VENV_TRAIN"
  uv pip install -q -p "$VENV_TRAIN/bin/python" -r ml/train/requirements.txt -r ml/train/requirements-gpu.txt
  uv pip install -q -p "$VENV_TRAIN/bin/python" -e "$LLAMA_SRC/gguf-py"
  uv cache clean -q || true   # ~wheel-sized duplicate on the volume otherwise
  touch "$MARK_VOL/venv-train"
fi
rm -f "$MARK_VOL/fla-missing"
"$VENV_TRAIN/bin/python" - "$MARK_VOL/fla-missing" <<'EOF'
import sys, torch, transformers, peft
assert torch.cuda.is_available(), "torch sees no GPU"
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); (x @ x).sum().item()
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      "cap", torch.cuda.get_device_capability(0), "transformers", transformers.__version__, "peft", peft.__version__)
try:
    import fla  # Gated DeltaNet kernels for Qwen3.5
    # the op transformers resolves for Qwen3.5 ("fla" + "ops.gated_delta_rule"); run it once forward and
    # backward so a Triton kernel that does not compile for sm_120 is caught here, not mid-matrix
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    B, T, H, K = 1, 128, 2, 64
    q, k, v = (torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32))
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)
    o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    o.float().sum().backward()
    assert torch.isfinite(q.grad).all(), "fla backward produced non-finite grads"
    print("flash-linear-attention", fla.__version__, "chunk_gated_delta_rule fwd+bwd OK on", torch.cuda.get_device_name(0))
except Exception as e:
    print("WARNING flash-linear-attention unavailable, Qwen3.5 trains on the slow PyTorch path:", e)
    open(sys.argv[1], "w").write(repr(e) + "\n")  # train_matrix.sh preflight refuses Qwen runs on this marker
EOF

# ---- 6. self-test (unit tests: metrics, gates, rendering parity, judge/rescore plumbing via the stub)
if step selftest "$MARK_VOL"; then
  "$VENV_TRAIN/bin/python" -m pytest -q ml/eval/tests ml/train/tests
  touch "$MARK_VOL/selftest"
fi

# ---- 6b. llama.cpp GPU KERNEL SMOKE TEST, on every start: ml/pod/kernel_smoke.sh (phase A prepares the smoke
# models, retried once; a failure there is its own error, exit 7, with no build tested and no fallback build tried;
# phase B tests the CUDA 12.8 build, then CUDA 13.5, then a source build; none passes -> llama-cpu-fallback).
bash ml/pod/kernel_smoke.sh

# ---- 7. base weights (public repos; ~9.8 GB for all three) into $HF_HOME on the volume, each at the commit
# pinned in ml/train/configs/base.yaml (base_revisions; TRAINING_RUNBOOK 0.1). sft/merge load the same commit.
MODELS="${MODELS-LiquidAI/LFM3-1.7B Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B}"   # MODELS="" (probe session): none
for m in $MODELS; do
  rev=$("$VENV_TRAIN/bin/python" -m ml.train.config revision "$m")   # refuses a repo without a pinned commit
  key="weights-$(echo "$m" | tr '/' '_')-${rev:0:12}"
  if step "$key" "$MARK_VOL"; then
    "$VENV_TRAIN/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('$m', revision='$rev'))"
    touch "$MARK_VOL/$key"
  fi
done

# ---- 8. ROUND-TRIP PROBE ENVIRONMENT (owned by ml/roundtrip, NOT implemented here)
# The probe (TTS + ASR on GPU, docs/ml/ROUNDTRIP.md) brings its own setup. If the roundtrip team
# ships ml/roundtrip/pod_setup.sh it is run here, otherwise this is a no-op.
if [[ -x ml/roundtrip/pod_setup.sh ]] && step roundtrip-setup "$MARK_VOL"; then
  bash ml/roundtrip/pod_setup.sh && touch "$MARK_VOL/roundtrip-setup"
fi

df -h "$W" | tail -1
log "bootstrap complete"
