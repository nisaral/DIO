#!/usr/bin/env bash
# Wrap two already-running (or start) vLLM servers with DIO.
# Usage: bash examples/wrap_two_vllm.sh meta-llama/Llama-3.2-3B-Instruct

set -euo pipefail
MODEL="${1:-meta-llama/Llama-3.2-3B-Instruct}"

echo "==> Starting vLLM on GPU0 :8000"
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port 8000 --host 0.0.0.0 &
PID0=$!

echo "==> Starting vLLM on GPU1 :8001"
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port 8001 --host 0.0.0.0 &
PID1=$!

cleanup() {
  kill $PID0 $PID1 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Waiting for vLLM..."
for p in 8000 8001; do
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 || \
       curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1; then
      echo "  :$p ready"
      break
    fi
    sleep 2
  done
done

echo "==> Starting DIO gateway on :8085"
dio serve \
  -b "gpu0=http://127.0.0.1:8000" \
  -b "gpu1=http://127.0.0.1:8001" \
  --strategy nlms --nlms-mode dual \
  --slo-ms 60000 \
  --port 8085
