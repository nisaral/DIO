#!/usr/bin/env bash
# Optional ~20-30 min validation: compare emulated slow worker vs a second real GPU.
# Run on RunPod/Vast (T4/L4) OR Lightning with 2× GPU when available.
#
# Usage:
#   bash benchmarks/run_t2_validation.sh              # 2 real GPUs (auto-detect)
#   VALIDATION_GPU_SLOW=0 VALIDATION_GPU_FAST=1 bash benchmarks/run_t2_validation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PATH="/usr/local/go/bin:${PATH:-}"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-3B-Instruct}"
PAIRING="${PAIRING:-t4_vs_a100}"
MANAGER_URL="http://127.0.0.1:8085"
WORKER_SCRIPT="$ROOT_DIR/benchmarks/worker_gpu.py"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/benchmarks/results_validation}"
LOCUST_FILE="$ROOT_DIR/benchmarks/real_world/locustfile.py"

GPU_COUNT=1
if command -v nvidia-smi &>/dev/null; then
  GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
fi

cleanup() {
  pkill -9 -f dio-manager 2>/dev/null || true
  pkill -9 -f worker_gpu.py 2>/dev/null || true
  pkill -9 -f locust 2>/dev/null || true
  sleep 2
}

go build -o dio-manager ./cmd/manager/main.go
cleanup
./dio-manager > validation_manager.log 2>&1 &
for _ in $(seq 1 25); do curl -sf "$MANAGER_URL/api/test" >/dev/null 2>&1 && break; sleep 1; done

mkdir -p "$RESULTS_DIR"

if [[ "$GPU_COUNT" -ge 2 ]]; then
  echo "=== Real 2-GPU validation (physical heterogeneity) ==="
  GPU_FAST="${VALIDATION_GPU_FAST:-0}"
  GPU_SLOW="${VALIDATION_GPU_SLOW:-1}"
  CUDA_VISIBLE_DEVICES=$GPU_FAST python3 "$WORKER_SCRIPT" --worker-id val_fast --port 50060 \
    --tier large --vram 70000 --model-id "$MODEL_ID" --manager-addr 127.0.0.1:50055 \
    > validation_fast.log 2>&1 &
  sleep 25
  CUDA_VISIBLE_DEVICES=$GPU_SLOW python3 "$WORKER_SCRIPT" --worker-id val_slow --port 50061 \
    --tier large --vram 16000 --model-id "$MODEL_ID" --manager-addr 127.0.0.1:50055 \
    > validation_slow.log 2>&1 &
  sleep 25
  MODE="real_2gpu"
else
  echo "=== Single GPU: 1 real + 1 calibrated mock (emulated) ==="
  CUDA_VISIBLE_DEVICES=0 python3 "$WORKER_SCRIPT" --worker-id val_fast --port 50060 \
    --tier large --vram 70000 --model-id "$MODEL_ID" --manager-addr 127.0.0.1:50055 \
    > validation_fast.log 2>&1 &
  sleep 25
  python3 "$WORKER_SCRIPT" --mock --worker-id val_slow_emulated --port 50061 \
    --tier small --vram 8000 --latency-profile "$PAIRING" --profile-role slow \
    --manager-addr 127.0.0.1:50055 > validation_slow.log 2>&1 &
  MODE="real_plus_emulated"
fi

for _ in $(seq 1 60); do
  count=$(curl -sf "$MANAGER_URL/debug/workers" | python3 -c "import sys,json; print(json.load(sys.stdin).get('worker_count',0))" 2>/dev/null || echo 0)
  [[ "$count" -ge 2 ]] && break
  sleep 2
done

export WORKLOAD_FILE="$ROOT_DIR/benchmarks/data/sharegpt.jsonl"
export MODEL_ID TTFT_SLO_MS=2000
export SCHEDULER_STRATEGY=nlms
locust -f "$LOCUST_FILE" --headless -u 10 -r 2 -t 60s \
  --host "$MANAGER_URL" --csv "$RESULTS_DIR/T2_validation_${MODE}"

python3 benchmarks/compare_emulation_to_real.py \
  --pairing "$PAIRING" \
  --real-log validation_fast.log \
  --output "$ROOT_DIR/benchmarks/emulation_validation.json"

cleanup
echo "Done. See benchmarks/emulation_validation.json and $RESULTS_DIR/"