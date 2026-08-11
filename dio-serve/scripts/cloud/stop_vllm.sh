#!/usr/bin/env bash
# Stop vLLM on this worker and confirm the VRAM was actually released.
#
#   bash stop_vllm.sh [PORT]
set -euo pipefail

PORT="${1:-8000}"
PIDFILE="$HOME/.vllm_${PORT}.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  PID="$(cat "$PIDFILE")"
  echo "stopping vLLM pid $PID"
  kill "$PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$PID" 2>/dev/null || true
  rm -f "$PIDFILE"
else
  # Fall back to a pattern match: a crashed run may have left no usable pidfile.
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
  echo "no live pidfile; sent pattern kill"
fi

sleep 3
echo "--- residual GPU processes ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
echo "--- free VRAM ---"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "stopped."
