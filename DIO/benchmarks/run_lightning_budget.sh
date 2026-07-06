#!/usr/bin/env bash
# Credit-efficient DIO benchmark suite for Lightning AI (1×A100).
# Core matrix: 1 real worker + 1 slow mock (2 workers total — never 2 models on one GPU).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Auto-detect Lightning path
if [[ -d "/teamspace/studios/this_studio" ]]; then
  export PATH="/usr/local/go/bin:$PATH"
fi

export DIO_SLO_MS="${DIO_SLO_MS:-120000}"
export DIO_ADMISSION_OFF="${DIO_ADMISSION_OFF:-1}"
export SKIP_T7="${SKIP_T7:-1}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-3B-Instruct}"
NUM_REAL_WORKERS=1
NUM_PAPER_WORKERS=2
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
  export DIO_SLO_MS DIO_ADMISSION_OFF
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
      --vram 0 --model-id "$MODEL_ID" \
      --manager-addr 127.0.0.1:50055 \
      > "worker_${i}.log" 2>&1 &
    port=$((port+1))
    sleep 5
  done

  for ((j=0; j<mock_count; j++)); do
    local wid="mock_$j"
    local profile_args=()
    if [[ "$slow_mock" == "true" && "$j" -eq 0 ]]; then
      profile_args=(--latency-profile "${LATENCY_PAIRING:-t4_vs_a100}" --profile-role slow)
      echo "Starting mock worker $wid (calibrated profile ${LATENCY_PAIRING:-t4_vs_a100})"
    else
      echo "Starting mock worker $wid (baseline mock)"
    fi
    local mock_tier="small"
    local mock_vram=8000
    if [[ "$slow_mock" == "true" && "$j" -eq 0 ]]; then
      mock_tier="large"
      mock_vram=32000
    fi
    python3 "$WORKER_SCRIPT" --mock \
      --worker-id "$wid" --port "$port" --tier "$mock_tier" \
      --vram "$mock_vram" "${profile_args[@]}" \
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
  set +e
  locust -f "$LOCUST_FILE" --headless \
    -u "$LOCUST_USERS" -r "$LOCUST_RATE" -t "$LOCUST_DURATION" \
    --host "$MANAGER_URL" \
    --csv "$RESULTS_DIR/$test_id"
  local rc=$?
  set -e
  [[ "$rc" -ne 0 ]] && echo "WARN: $test_id locust exit $rc — continuing"
}

mkdir -p "$RESULTS_DIR"
build_manager

echo ""
echo "=== Preflight (GPU + single real inference) ==="
bash "$SCRIPT_DIR/preflight_gpu.sh" || { echo "Preflight failed — aborting."; exit 1; }

if [[ "${SKIP_T7:-1}" != "1" ]]; then
  echo ""
  echo "=== T7 Scalability (32 mock workers) ==="
  start_manager "round_robin"
  port=50060
  for ((i=0; i<32; i++)); do
    python3 "$WORKER_SCRIPT" --mock --worker-id "m_$i" --port "$port" --tier small \
      --latency-profile scalability_fast --vram 32000 \
      --manager-addr 127.0.0.1:50055 > /dev/null 2>&1 &
    port=$((port+1))
    sleep 0.2
  done
  wait_workers 32
  set +e
  WORKLOAD_FILE="$ROOT_DIR/benchmarks/data/t7_scalability.jsonl" \
  locust -f "$LOCUST_FILE" --headless \
    -u "${T7_LOCUST_USERS:-12}" -r "${T7_LOCUST_RATE:-3}" -t "60s" \
    --host "$MANAGER_URL" --csv "$RESULTS_DIR/T7_Scalability_32"
  set -e
else
  echo ""
  echo "=== T7 — SKIPPED (set SKIP_T7=0 to run) ==="
fi

# --- T2: heterogeneity (1 real + 1 slow mock) ---
echo ""
echo "=== T2 Heterogeneity (1 real + 1 slow mock) ==="
for strat in nlms round_robin; do
  start_manager "$strat"
  start_workers 1 1 true
  sleep 20
  run_locust "T2_${strat}_hetero" "sharegpt.jsonl"
done

# --- Core matrix: 1 real + 1 slow mock × strategies × datasets ---
echo ""
echo "=== Core paper matrix (${NUM_PAPER_WORKERS} workers: 1 real + 1 slow mock) ==="
for strat in "${STRATEGIES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    tag="${ds%.jsonl}"
    test_id="${strat}_${tag}_${NUM_PAPER_WORKERS}w"
    echo ""
    echo "--- $test_id ---"
    start_manager "$strat"
    start_workers "$NUM_REAL_WORKERS" 1 true
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
echo "=== Admissibility validation ==="
python3 benchmarks/validate_results.py --json "$ROOT_DIR/benchmarks/results_summary.json" || VALID_FAIL=1

echo ""
echo "DONE. Results:"
echo "  CSVs:    $RESULTS_DIR"
echo "  Summary: $ROOT_DIR/benchmarks/results_summary.json"
echo ""
echo "Download results_summary.json and results_final/ to your laptop, then:"
echo "  python benchmarks/generate_figures_from_json.py --out ../figs"
if [[ "${VALID_FAIL:-0}" -eq 1 ]]; then
  echo "WARNING: validate_results.py reported failures — review before paper update"
fi