#!/usr/bin/env bash
# Start stock vLLM on this worker, bound so the gateway node can reach it.
#
#   bash start_vllm.sh [PORT] [MODEL]
#
# Nothing here is a DIO-specific patch: these are stock upstream flags. That is
# the point — the paper claims zero engine modification, so the worker must be
# an unmodified `vllm.entrypoints.openai.api_server`.
set -euo pipefail

PORT="${1:-8000}"
MODEL="${2:-$(cat "$HOME/.dio_model" 2>/dev/null || echo Qwen/Qwen2.5-3B-Instruct)}"
VENV="${VENV:-$HOME/dio-venv}"
LOG="$HOME/vllm_${PORT}.log"

# 0.85 leaves headroom for the CUDA context; the same fraction on every SKU so
# KV-cache capacity scales with physical VRAM rather than with a hand-tuned
# per-node number (which would be a confound).
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LEN="${MAX_LEN:-4096}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "vLLM already serving on :${PORT} — nothing to do."
  exit 0
fi

echo "starting vLLM: model=$MODEL port=$PORT util=$GPU_UTIL max_len=$MAX_LEN"

# --host 0.0.0.0 is REQUIRED: the gateway runs on a different node. Do not use
# 127.0.0.1 here or the gateway will see connection-refused.
# --dtype half: T4 is CC 7.5 (no bf16). Use half on all SKUs so dtype is not a confound.
DTYPE="${DTYPE:-half}"
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --enable-prefix-caching \
  > "$LOG" 2>&1 &

echo $! > "$HOME/.vllm_${PORT}.pid"
echo "pid $(cat "$HOME/.vllm_${PORT}.pid"), log -> $LOG"

# Weights are already cached by setup_worker.sh, so first load is graph capture
# and KV allocation — usually 60-180 s.
echo -n "waiting for readiness"
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo " READY after ${i}0s"
    # Record the SKU next to the endpoint — this is what the paper's hardware
    # table is populated from, so capture it from the machine, not from memory.
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$HOME/.dio_gpu"
    echo "GPU: $(cat "$HOME/.dio_gpu")"
    echo "--- /metrics sanity (DIO scrapes these) ---"
    # Metric names vary by vLLM version; never fail the start if grep misses.
    curl -s "http://127.0.0.1:${PORT}/metrics" \
      | grep -E 'vllm:' \
      | head -8 || true
    exit 0
  fi
  sleep 10
  echo -n "."
done

echo
echo "FATAL: vLLM did not become ready in 20 min. Last 40 log lines:" >&2
tail -40 "$LOG" >&2
exit 1
