#!/usr/bin/env bash
# Credit-efficient DIO benchmark suite for Lightning AI (single GPU).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Auto-detect Lightning path
if [[ -d "/teamspace/studios/this_studio" ]]; then
  export PATH="/usr/local/go/bin:$PATH"
fi

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-3B-Instruct}"
NUM_REAL_WORKERS="${NUM_REAL_WORKERS:-2}"
LOCUST_USERS="${LOCUST_USERS:-15}"
LOCUST_RATE="${LOCUST_RATE:-3}"
LOCUST_DURATION="${LOCUST_DURATION:-120s}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/benchmarks/results_final}"
MANAGER_BIN="$ROOT_DIR/dio-manager"
WORKER_SCRIPT="$ROOT_DIR/benchmarks/worker_gpu.py"
LOCUST_FILE="$ROOT_DIR/benchmarks/real_world/locustfile.py"
MANAGER_URL="http://127.0.0.1:8085"

# Use 1B on small GPUs
if command -v nvidia-smi &>/dev/null; then
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ "$VRAM_MB" -lt 40000 ]]; then
    MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
    echo "Small GPU detected (${VRAM_MB}MB). Defaulting to MODEL_ID=$MODEL_ID"
  fi
fi

STRATEGIES=("nlms" "rls" "round_robin" "least_load")
DATASETS=("sharegpt.jsonl" "arxiv.jsonl")

cleanup() {
  echo "Cleaning up processes..."
  pkill -9 -f dio-manager 2>/dev/null || true
  pkill -9 -f worker_gpu.py 2>/dev/null || true
  pkill -9 -f locust 2>/dev/null || true
  sleep 3
}

build_manager() {
  echo "Building DIO manager..."
  go build -o "$MANAGER_BIN" ./cmd/manager/main.go
}

start_manager() {
  local strategy="$1"
  export SCHEDULER_STRATEGY="$strategy"
  cleanup
  "$MANAGER_BIN" > manager.log 2>&1 &
  for i in $(seq 1 20); do
    if curl -sf "$MANAGER_URL/api/test" >/dev/null 2>&1; then
      echo "Manager up (strategy=$strategy)"
      return 0
    fi
    sleep 1
  done
  echo "Manager failed to start"; tail -20 manager.log; exit 1
}

wait_workers() {
  local expected="$1"
  for i in $(seq 1 60); do
    count=$(curl -sf "$MANAGER_URL/debug/workers" | python3 -c "import sys,json; print(json.load(sys.stdin).get('worker_count',0))" 2>/dev/null || echo 0)
    if [[ "$count" -ge "$expected" ]]; then
      echo "Registered $count workers"
      return 0
    fi
    sleep 2
  done
  echo "Worker registration timeout (expected $expected)"; exit 1
}

start_workers() {
  local real_count="$1"
  local mock_count="$2"
  local slow_mock="${3:-false}"
  local port=50060
  local i=0

  for ((i=0; i<real_count; i++)); do
    echo "Starting real worker w_$i on port $port"
    CUDA_VISIBLE_DEVICES=0 python3 "$WORKER_SCRIPT" \
      --worker-id "w_$i" --port "$port" --tier large \
      --vram 30000 --model-id "$MODEL_ID" \
      --manager-addr 127.0.0.1:50055 \
      > "worker_${i}.log" 2>&1 &
    port=$((port+1))
    sleep 15
  done

  for ((j=0; j<mock_count; j++)); do
    local wid="mock_$j"
    local mult=1.0
    if [[ "$slow_mock" == "true" && "$j" -eq 0 ]]; then mult=2.5; fi
    echo "Starting mock worker $wid (latency_mult=$mult)"
    python3 "$WORKER_SCRIPT" --mock \
      --worker-id "$wid" --port "$port" --tier small \
      --vram 8000 --latency-mult "$mult" \
      --manager-addr 127.0.0.1:50055 \
      > "worker_mock_${j}.log" 2>&1 &
    port=$((port+1))
    sleep 1
  done

  wait_workers $((real_count + mock_count))
}

run_locust() {
  local test_id="$1"
  local dataset="$2"
  export WORKLOAD_FILE="$ROOT_DIR/benchmarks/data/$dataset"
  export MODEL_ID
  mkdir -p "$RESULTS_DIR"
  echo "Locust: $test_id ($dataset)"
  locust -f "$LOCUST_FILE" --headless \
    -u "$LOCUST_USERS" -r "$LOCUST_RATE" -t "$LOCUST_DURATION" \
    --host "$MANAGER_URL" \
    --csv "$RESULTS_DIR/$test_id"
}

mkdir -p "$RESULTS_DIR"
build_manager

# --- T7: mock scalability (no GPU inference) ---
echo ""
echo "=== T7 Scalability (32 mock workers) ==="
start_manager "nlms"
start_workers 0 32 false
run_locust "T7_Scalability_32" "sharegpt.jsonl"

# --- T2: heterogeneity (1 real + 1 slow mock) ---
echo ""
echo "=== T2 Heterogeneity (1 real + 1 slow mock) ==="
for strat in nlms round_robin; do
  start_manager "$strat"
  start_workers 1 1 true
  sleep 20
  run_locust "T2_${strat}_hetero" "sharegpt.jsonl"
done

# --- Core matrix: 2 real workers × strategies × datasets ---
echo ""
echo "=== Core paper matrix (${NUM_REAL_WORKERS} real workers) ==="
for strat in "${STRATEGIES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    tag="${ds%.jsonl}"
    test_id="${strat}_${tag}_${NUM_REAL_WORKERS}w"
    echo ""
    echo "--- $test_id ---"
    start_manager "$strat"
    start_workers "$NUM_REAL_WORKERS" 0 false
    echo "Warmup 30s..."
    sleep 30
    run_locust "$test_id" "$ds"
  done
done

cleanup

echo ""
echo "=== Analyzing results ==="
python3 benchmarks/real_world/analyze_results.py \
  --results-dir "$RESULTS_DIR" \
  --out-json "$ROOT_DIR/benchmarks/results_summary.json"

echo ""
echo "DONE. Results:"
echo "  CSVs:    $RESULTS_DIR"
echo "  Summary: $ROOT_DIR/benchmarks/results_summary.json"
echo ""
echo "Download results_summary.json and results_final/ to your laptop, then:"
echo "  python benchmarks/generate_figures_from_json.py --out ../figs"