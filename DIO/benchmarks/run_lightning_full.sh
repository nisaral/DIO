#!/usr/bin/env bash
# Full DIO GPU benchmark suite for Lightning AI (2×A100 recommended, 1×H100 supported).
# Usage: bash benchmarks/run_lightning_full.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

export PATH="/usr/local/go/bin:${PATH:-}"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-3B-Instruct}"
LOCUST_USERS="${LOCUST_USERS:-20}"
LOCUST_RATE="${LOCUST_RATE:-4}"
LOCUST_DURATION="${LOCUST_DURATION:-120s}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/benchmarks/results_final}"
MANAGER_BIN="$ROOT_DIR/dio-manager"
WORKER_SCRIPT="$ROOT_DIR/benchmarks/worker_gpu.py"
LOCUST_FILE="$ROOT_DIR/benchmarks/real_world/locustfile.py"
MANAGER_URL="http://127.0.0.1:8085"
STRATEGIES=("nlms" "rls" "round_robin" "least_load")
DATASETS=("sharegpt.jsonl" "arxiv.jsonl" "azure_code.jsonl")

GPU_COUNT=1
if command -v nvidia-smi &>/dev/null; then
  GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
  echo "Detected $GPU_COUNT GPU(s)"
  nvidia-smi -L
fi

if [[ "$GPU_COUNT" -ge 2 ]]; then
  NUM_REAL_WORKERS=2
  WORKER_GPU_MODE="multi"
  echo "Mode: 2 real workers (1 per GPU) — real heterogeneity"
else
  NUM_REAL_WORKERS="${NUM_REAL_WORKERS:-2}"
  WORKER_GPU_MODE="single"
  echo "Mode: $NUM_REAL_WORKERS workers on GPU 0 (single-GPU)"
fi

cleanup() {
  pkill -9 -f dio-manager 2>/dev/null || true
  pkill -9 -f worker_gpu.py 2>/dev/null || true
  pkill -9 -f locust 2>/dev/null || true
  sleep 3
}

build_manager() {
  echo "=== Building manager ==="
  go build -o "$MANAGER_BIN" ./cmd/manager/main.go
}

start_manager() {
  local strategy="$1"
  export SCHEDULER_STRATEGY="$strategy"
  cleanup
  "$MANAGER_BIN" > manager.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf "$MANAGER_URL/api/test" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "Manager failed"; tail -30 manager.log; exit 1
}

wait_workers() {
  local expected="$1"
  for _ in $(seq 1 90); do
    count=$(curl -sf "$MANAGER_URL/debug/workers" | python3 -c "import sys,json; print(json.load(sys.stdin).get('worker_count',0))" 2>/dev/null || echo 0)
    [[ "$count" -ge "$expected" ]] && echo "Workers registered: $count" && return 0
    sleep 2
  done
  echo "Timeout waiting for $expected workers"; exit 1
}

start_real_workers() {
  local count="$1"
  local port=50060
  for ((i=0; i<count; i++)); do
    local gpu=0
    if [[ "$WORKER_GPU_MODE" == "multi" ]]; then gpu=$i; fi
    echo "Starting real worker w_$i on GPU $gpu port $port"
    CUDA_VISIBLE_DEVICES=$gpu python3 "$WORKER_SCRIPT" \
      --worker-id "w_$i" --port "$port" --tier large \
      --vram 70000 --model-id "$MODEL_ID" \
      --manager-addr 127.0.0.1:50055 \
      > "worker_${i}.log" 2>&1 &
    port=$((port+1))
    sleep 20
  done
  wait_workers "$count"
}

start_mock_workers() {
  local count="$1"
  local port=50060
  for ((i=0; i<count; i++)); do
    python3 "$WORKER_SCRIPT" --mock --worker-id "mock_$i" --port "$port" \
      --tier small --vram 2000 --latency-mult 1.0 \
      --manager-addr 127.0.0.1:50055 > "mock_${i}.log" 2>&1 &
    port=$((port+1))
    sleep 1
  done
  wait_workers "$count"
}

run_locust() {
  local test_id="$1"
  local dataset="$2"
  export WORKLOAD_FILE="$ROOT_DIR/benchmarks/data/$dataset"
  export MODEL_ID TTFT_SLO_MS=2000
  mkdir -p "$RESULTS_DIR"
  locust -f "$LOCUST_FILE" --headless \
    -u "$LOCUST_USERS" -r "$LOCUST_RATE" -t "$LOCUST_DURATION" \
    --host "$MANAGER_URL" --csv "$RESULTS_DIR/$test_id"
}

smoke_tests() {
  echo "=== Smoke tests ==="
  curl -sf "$MANAGER_URL/api/test" | head -c 200; echo
  curl -sf -X POST "$MANAGER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"llama","messages":[{"role":"user","content":"ping"}]}' | head -c 300; echo
  curl -sf "$MANAGER_URL/debug/metrics" | python3 -c "import sys,json; d=json.load(sys.stdin); print('strategy',d.get('strategy'),'workers',len(d.get('workers',[])))"
}

mkdir -p "$RESULTS_DIR"
build_manager

echo ""; echo "=== Preflight (GPU + single real inference) ==="
bash "$SCRIPT_DIR/preflight_gpu.sh" || { echo "Preflight failed — aborting."; exit 1; }

# T7 — mock scalability (no GPU inference)
echo ""; echo "=== T7 Scalability (32 mock) ==="
start_manager "nlms"
smoke_tests || true
port=50060
for ((i=0; i<32; i++)); do
  python3 "$WORKER_SCRIPT" --mock --worker-id "m_$i" --port "$port" --tier small --vram 500 \
    --manager-addr 127.0.0.1:50055 > /dev/null 2>&1 &
  port=$((port+1))
done
wait_workers 32
run_locust "T7_Scalability_32" "sharegpt.jsonl"

# T1 — NLMS convergence probes
echo ""; echo "=== T1 Convergence ==="
start_manager "nlms"
start_real_workers 1
sleep 30
for i in $(seq 1 25); do
  curl -sf -X POST "$MANAGER_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"probe $i\",\"model_id\":\"$MODEL_ID\",\"tier\":\"small\"}" >/dev/null || true
  sleep 2
done
echo "T1 probes done — check manager.log for NLMS_TELEMETRY"

# T2 — heterogeneity routing (2 real GPUs or 1 real + 1 slow mock)
echo ""; echo "=== T2 Heterogeneity ==="
for strat in nlms round_robin; do
  start_manager "$strat"
  if [[ "$GPU_COUNT" -ge 2 ]]; then
    start_real_workers 2
  else
    CUDA_VISIBLE_DEVICES=0 python3 "$WORKER_SCRIPT" --worker-id fast --port 50060 --tier large \
      --vram 70000 --model-id "$MODEL_ID" --manager-addr 127.0.0.1:50055 > w_fast.log 2>&1 &
    sleep 20
    python3 "$WORKER_SCRIPT" --mock --worker-id slow --port 50061 --tier small \
      --latency-mult 2.5 --vram 8000 --manager-addr 127.0.0.1:50055 > w_slow.log 2>&1 &
    wait_workers 2
  fi
  sleep 20
  run_locust "T2_${strat}_hetero_2w" "sharegpt.jsonl"
done

# Core paper matrix
echo ""; echo "=== Core matrix (${NUM_REAL_WORKERS} workers × ${#STRATEGIES[@]} strategies × ${#DATASETS[@]} datasets) ==="
for strat in "${STRATEGIES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    tag="${ds%.jsonl}"
    test_id="${strat}_${tag}_${NUM_REAL_WORKERS}w"
    echo "--- $test_id ---"
    start_manager "$strat"
    start_real_workers "$NUM_REAL_WORKERS"
    echo "Warmup 45s..."
    sleep 45
    run_locust "$test_id" "$ds"
  done
done

cleanup

echo ""; echo "=== Analyze ==="
python3 benchmarks/real_world/analyze_results.py \
  --results-dir "$RESULTS_DIR" \
  --out-json "$ROOT_DIR/benchmarks/results_summary.json" \
  --out-tex "$ROOT_DIR/benchmarks/results_table.tex"

python3 benchmarks/generate_figures_from_json.py --out "$ROOT_DIR/../figs" 2>/dev/null || true

echo ""; echo "=== Admissibility validation ==="
python3 benchmarks/validate_results.py --json "$ROOT_DIR/benchmarks/results_summary.json" || VALID_FAIL=1

echo ""
echo "============================================"
echo "COMPLETE"
echo "  CSVs:    $RESULTS_DIR"
echo "  Summary: $ROOT_DIR/benchmarks/results_summary.json"
echo "  Figures: $ROOT_DIR/../figs/"
echo ""
echo "Download results_final/ and results_summary.json to your laptop."
if [[ "${VALID_FAIL:-0}" -eq 1 ]]; then
  echo "WARNING: validate_results.py reported failures — review before paper update"
fi
echo "============================================"