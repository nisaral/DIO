#!/usr/bin/env bash
# Run BEFORE the full benchmark. Confirms GPU is real and inference is not mock/CPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PATH="/usr/local/go/bin:${PATH:-}"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-3B-Instruct}"
MANAGER_URL="http://127.0.0.1:8085"
PASS=0
FAIL=0

ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
info() { echo "  [INFO] $1"; }

echo "============================================"
echo "DIO GPU Preflight Check"
echo "============================================"

# 1. NVIDIA driver
if command -v nvidia-smi &>/dev/null; then
  ok "nvidia-smi available"
  nvidia-smi -L
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
else
  bad "nvidia-smi not found — no GPU driver"
fi

# 2. PyTorch CUDA
python3 <<'PY'
import sys
try:
    import torch
    if torch.cuda.is_available():
        print(f"  [PASS] PyTorch CUDA: {torch.cuda.device_count()} device(s)")
        for i in range(torch.cuda.device_count()):
            print(f"         GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  [FAIL] PyTorch: cuda.is_available() == False (will run CPU/mock)")
        sys.exit(1)
except Exception as e:
    print(f"  [FAIL] PyTorch check: {e}")
    sys.exit(1)
PY
[[ $? -eq 0 ]] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# 3. Build manager
go build -o dio-manager ./cmd/manager/main.go && ok "Manager built" || bad "Manager build failed"

# 4. Start manager + ONE real worker
pkill -9 -f dio-manager 2>/dev/null || true
pkill -9 -f worker_gpu.py 2>/dev/null || true
sleep 2

./dio-manager > preflight_manager.log 2>&1 &
for i in $(seq 1 20); do
  curl -sf "$MANAGER_URL/api/test" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "$MANAGER_URL/api/test" >/dev/null 2>&1 && ok "Manager HTTP up" || bad "Manager not responding"

info "Starting ONE real worker on GPU 0 (model load ~20-60s)..."
CUDA_VISIBLE_DEVICES=0 python3 benchmarks/worker_gpu.py \
  --worker-id preflight_gpu0 --port 50060 --tier large \
  --vram 70000 --model-id "$MODEL_ID" \
  --manager-addr 127.0.0.1:50055 > preflight_worker.log 2>&1 &

for i in $(seq 1 90); do
  count=$(curl -sf "$MANAGER_URL/debug/workers" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('worker_count',0))" 2>/dev/null || echo 0)
  [[ "$count" -ge 1 ]] && break
  sleep 2
done
[[ "$count" -ge 1 ]] && ok "Worker registered" || bad "Worker never registered — see preflight_worker.log"

# 5. Check worker log for GPU not mock
if grep -q "LOAD FAILURE\|Falling back to MOCK\|on CPU" preflight_worker.log 2>/dev/null; then
  bad "Worker log shows MOCK or CPU fallback — NOT admissible"
  tail -15 preflight_worker.log
elif grep -q "on CUDA\|cuda:0" preflight_worker.log 2>/dev/null; then
  ok "Worker log confirms CUDA load"
else
  info "Worker log (last 10 lines):"
  tail -10 preflight_worker.log
  bad "Cannot confirm CUDA in worker log — inspect preflight_worker.log"
fi

# 6. GPU utilization during inference
info "Sending test request (watch nvidia-smi)..."
UTIL_BEFORE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')

START_MS=$(python3 -c "import time; print(int(time.time()*1000))")
HTTP_CODE=$(curl -sf -o /tmp/preflight_resp.json -w "%{http_code}" \
  -X POST "$MANAGER_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Preflight GPU test. Reply with one word.\",\"model_id\":\"$MODEL_ID\",\"tier\":\"small\"}" \
  --max-time 120 || echo "000")
END_MS=$(python3 -c "import time; print(int(time.time()*1000))")
LAT=$((END_MS - START_MS))

sleep 2
UTIL_AFTER=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')

if [[ "$HTTP_CODE" == "200" ]]; then
  ok "Test request returned HTTP 200 (latency ${LAT}ms)"
  python3 -c "import json; d=json.load(open('/tmp/preflight_resp.json')); print('         tokens:', d.get('tokens_used'), 'latency_ms:', d.get('latency_ms'))"
else
  bad "Test request failed HTTP $HTTP_CODE"
fi

# Latency sanity for 3B on A100: not instant (mock) and not 60s+ (CPU thrash)
if [[ "$LAT" -lt 200 ]]; then
  bad "Latency ${LAT}ms too fast — likely MOCK worker"
elif [[ "$LAT" -gt 60000 ]]; then
  bad "Latency ${LAT}ms too slow — likely CPU or VRAM thrash"
else
  ok "Latency ${LAT}ms in plausible GPU range (200ms–60s for 3B)"
fi

if [[ "$UTIL_AFTER" -gt "$UTIL_BEFORE" ]] || [[ "$UTIL_AFTER" -gt 10 ]]; then
  ok "GPU utilization detected (${UTIL_BEFORE}% -> ${UTIL_AFTER}%)"
else
  bad "GPU utilization did not spike (before=${UTIL_BEFORE}% after=${UTIL_AFTER}%) — may not be using GPU"
fi

# 7. NLMS telemetry in manager
if grep -q "NLMS_TELEMETRY\|SCHED_OVERHEAD" preflight_manager.log 2>/dev/null; then
  ok "Manager NLMS telemetry present"
else
  info "NLMS telemetry may appear after more requests (not a hard fail)"
fi

pkill -9 -f dio-manager 2>/dev/null || true
pkill -9 -f worker_gpu.py 2>/dev/null || true

echo ""
echo "============================================"
echo "Preflight: $PASS passed, $FAIL failed"
if [[ "$FAIL" -gt 0 ]]; then
  echo "DO NOT run full benchmark until failures are fixed."
  echo "Logs: preflight_worker.log, preflight_manager.log"
  exit 1
fi
echo "Safe to run: bash benchmarks/run_lightning_full.sh"
echo "  (1×A100: core matrix uses 1 real + 1 slow mock — not 2 models on one GPU)"
echo "============================================"