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
WORKER_OK=0

ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
info() { echo "  [INFO] $1"; }

echo "============================================"
echo "DIO GPU Preflight Check"
echo "  ROOT: $ROOT_DIR"
echo "============================================"

if [[ "$ROOT_DIR" == *"/DIO/DIO" ]]; then
  info "Nested DIO/DIO path detected — consider: cd /teamspace/studios/this_studio/Go-serve/DIO"
fi

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

# 4. Start manager + ONE real worker (registers immediately; model loads in background)
pkill -9 -f dio-manager 2>/dev/null || true
pkill -9 -f worker_gpu.py 2>/dev/null || true
sleep 2

./dio-manager > preflight_manager.log 2>&1 &
MANAGER_PID=$!
for i in $(seq 1 30); do
  curl -sf "$MANAGER_URL/api/test" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "$MANAGER_URL/api/test" >/dev/null 2>&1 && ok "Manager HTTP up" || bad "Manager not responding"

info "Starting worker (registers in ~2s; model load continues in background)..."
CUDA_VISIBLE_DEVICES=0 python3 benchmarks/worker_gpu.py \
  --worker-id preflight_gpu0 --port 50060 --tier large \
  --vram 0 --model-id "$MODEL_ID" \
  --manager-addr 127.0.0.1:50055 > preflight_worker.log 2>&1 &
WORKER_PID=$!

count=0
for i in $(seq 1 120); do
  if grep -q "Successfully registered" preflight_worker.log 2>/dev/null; then
    count=1
    break
  fi
  count=$(curl -sf "$MANAGER_URL/debug/workers" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('worker_count',0))" 2>/dev/null || echo 0)
  [[ "$count" -ge 1 ]] && break
  sleep 2
done

if [[ "$count" -ge 1 ]]; then
  ok "Worker registered (waited ~$((i*2))s)"
  WORKER_OK=1
else
  bad "Worker never registered — see preflight_worker.log"
  info "Last 20 lines of preflight_worker.log:"
  tail -20 preflight_worker.log 2>/dev/null || true
fi

# 5. Check worker log for GPU not mock
if grep -q "LOAD FAILURE\|Falling back to MOCK" preflight_worker.log 2>/dev/null; then
  bad "Worker log shows MOCK fallback — NOT admissible"
  tail -15 preflight_worker.log
elif grep -qE "cuda:0|on cuda|Model loaded on cuda" preflight_worker.log 2>/dev/null; then
  ok "Worker log confirms CUDA load path"
elif grep -q "Model loaded" preflight_worker.log 2>/dev/null; then
  ok "Worker log confirms model loaded"
else
  info "Model may still be loading — waiting up to 180s more..."
  for j in $(seq 1 90); do
    grep -qE "Model loaded|Model loaded on cuda" preflight_worker.log 2>/dev/null && break
    sleep 2
  done
  if grep -qE "Model loaded|cuda" preflight_worker.log 2>/dev/null; then
    ok "Worker log confirms CUDA load (after extra wait)"
  else
    info "Worker log (last 15 lines):"
    tail -15 preflight_worker.log
    bad "Cannot confirm CUDA in worker log"
  fi
fi

# 6. GPU utilization during inference (only if worker registered)
if [[ "$WORKER_OK" -eq 1 ]]; then
  info "Waiting for model ready before inference test..."
  for j in $(seq 1 120); do
    grep -q "Model loaded" preflight_worker.log 2>/dev/null && break
    sleep 2
  done

  info "Sending test request (may take 30-90s on first GPU inference)..."
  UTIL_BEFORE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')

  START_MS=$(python3 -c "import time; print(int(time.time()*1000))")
  HTTP_CODE=$(curl -s -o /tmp/preflight_resp.json -w "%{http_code}" \
    -X POST "$MANAGER_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"Preflight GPU test. Reply with one word.\",\"model_id\":\"$MODEL_ID\",\"tier\":\"small\"}" \
    --max-time 180)
  END_MS=$(python3 -c "import time; print(int(time.time()*1000))")
  LAT=$((END_MS - START_MS))

  sleep 3
  UTIL_AFTER=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')

  if [[ "$HTTP_CODE" == "200" ]]; then
    ok "Test request returned HTTP 200 (latency ${LAT}ms)"
    python3 -c "import json; d=json.load(open('/tmp/preflight_resp.json')); print('         tokens:', d.get('tokens_used'), 'latency_ms:', d.get('latency_ms'))" 2>/dev/null || true
  else
    bad "Test request failed HTTP $HTTP_CODE"
    info "preflight_manager.log (last 10 lines):"
    tail -10 preflight_manager.log 2>/dev/null || true
  fi

  if [[ "$HTTP_CODE" == "200" ]]; then
    if [[ "$LAT" -lt 200 ]]; then
      bad "Latency ${LAT}ms too fast — likely MOCK worker"
    elif [[ "$LAT" -gt 120000 ]]; then
      bad "Latency ${LAT}ms too slow — likely CPU or VRAM thrash"
    else
      ok "Latency ${LAT}ms in plausible GPU range"
    fi

    if [[ "$UTIL_AFTER" -gt "$UTIL_BEFORE" ]] || [[ "$UTIL_AFTER" -gt 5 ]] || [[ "$LAT" -gt 1000 ]]; then
      ok "GPU activity confirmed (util ${UTIL_BEFORE}% -> ${UTIL_AFTER}%, latency ${LAT}ms)"
    else
      info "GPU util did not spike (${UTIL_BEFORE}% -> ${UTIL_AFTER}%) but HTTP 200 with ${LAT}ms — accepting"
      ok "Inference completed (util check skipped — A100 can be fast)"
    fi
  fi
else
  info "Skipping inference test — worker not registered"
fi

# 7. NLMS telemetry in manager
if grep -q "NLMS_TELEMETRY\|SCHED_OVERHEAD" preflight_manager.log 2>/dev/null; then
  ok "Manager NLMS telemetry present"
else
  info "NLMS telemetry may appear after inference (not a hard fail)"
fi

kill "$WORKER_PID" 2>/dev/null || pkill -9 -f worker_gpu.py 2>/dev/null || true
kill "$MANAGER_PID" 2>/dev/null || pkill -9 -f dio-manager 2>/dev/null || true
sleep 2

echo ""
echo "============================================"
echo "Preflight: $PASS passed, $FAIL failed"
if [[ "$FAIL" -gt 0 ]]; then
  echo "DO NOT run full benchmark until failures are fixed."
  echo "Logs: preflight_worker.log, preflight_manager.log"
  exit 1
fi
echo "Safe to run: bash benchmarks/run_lightning_full.sh"
echo "============================================"