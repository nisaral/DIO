#!/usr/bin/env bash
# Regime D — GPU worker bootstrap (run ON each rented GPU node).
#
# Installs vLLM into a venv and pre-downloads the model. Idempotent: safe to
# re-run. Does NOT start the server — use start_vllm.sh for that.
#
#   bash setup_worker.sh [MODEL]
#
# Default model is Qwen2.5-3B-Instruct: fits in 24 GB (L4) with room for KV
# cache, is ungated on HuggingFace (no token needed), and is the same weights
# on every SKU so the only difference between workers is the hardware.
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"
VENV="${VENV:-$HOME/dio-venv}"

echo "=== DIO Regime D worker setup ==="
echo "model: $MODEL"
echo "venv:  $VENV"

# --- 1. Sanity: is there actually a GPU here? ---------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi not found. This image has no NVIDIA driver." >&2
  echo "Pick an image with CUDA preinstalled when you create the node." >&2
  exit 1
fi
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo

# --- 2. OS packages -----------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git curl jq >/dev/null
echo "apt packages OK"

# --- 3. Python venv + vLLM ----------------------------------------------------
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip wheel

# Pin: latest vLLM (0.26+) pulls torch builds that need drivers newer than
# E2E's CUDA 12.8 stack (error: "NVIDIA driver ... too old (found version 12080)").
# 0.8.5 is OpenAI-compatible, has /metrics, and works on T4/A30 with driver 570/CUDA 12.8.
VLLM_VERSION="${VLLM_VERSION:-0.8.5}"
CUR=""
if python -c "import vllm" 2>/dev/null; then
  CUR="$(python -c 'import vllm; print(vllm.__version__)')"
fi
if [ "$CUR" != "$VLLM_VERSION" ]; then
  echo "installing vLLM==$VLLM_VERSION (was: ${CUR:-none}; takes 5-15 min)..."
  pip install --quiet "vllm==${VLLM_VERSION}"
else
  echo "vLLM already present: $CUR"
fi

# --- 4. Pre-download weights --------------------------------------------------
# Do this now, not at server start, so the timed experiment never waits on a
# download and every worker starts from an identical warm cache.
echo "pre-downloading $MODEL ..."
python - "$MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], ignore_patterns=["*.pth", "*.msgpack", "*.h5"])
print("cached at:", path)
PYEOF

echo "$MODEL" > "$HOME/.dio_model"
echo
echo "=== worker setup COMPLETE ==="
echo "next: bash start_vllm.sh"
