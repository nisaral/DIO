#!/usr/bin/env bash
# Preflight from the GATEWAY node — run this BEFORE the experiment.
#
#   bash preflight.sh "l4=http://10.0.0.11:8000,a30=http://10.0.0.12:8000"
#
# Every check here has cost you real money to learn the hard way if you skip it:
# a firewall rule you forgot means the harness sits in wait_url() for 15 minutes
# while both GPUs bill by the hour.
set -euo pipefail

SPEC="${1:-}"
if [ -z "$SPEC" ]; then
  echo "usage: bash preflight.sh 'id=http://host:port,id2=http://host2:port'" >&2
  exit 2
fi

FAIL=0
echo "=== Regime D preflight ==="
# A leftover file from an earlier run would silently poison the same-model check.
rm -f /tmp/dio_served_models

IFS=',' read -ra ENTRIES <<< "$SPEC"
for entry in "${ENTRIES[@]}"; do
  ID="${entry%%=*}"
  URL="${entry#*=}"
  echo
  echo "--- $ID -> $URL ---"

  # 1. TCP + HTTP reachability across the network boundary.
  if ! curl -sf --max-time 10 "$URL/v1/models" -o /tmp/dio_models.json; then
    echo "  FAIL: cannot GET $URL/v1/models"
    echo "        -> vLLM not started, bound to 127.0.0.1 instead of 0.0.0.0,"
    echo "           or the port is not open in the security group / firewall."
    FAIL=1
    continue
  fi
  # Extract the served model id. Prefer jq; fall back to sed so this works on a
  # bare image. An unreadable id must FAIL, not degrade to "?" — otherwise the
  # same-model check below passes vacuously and a model mismatch reaches the paper.
  if command -v jq >/dev/null 2>&1; then
    SERVED=$(jq -r '.data[0].id // empty' /tmp/dio_models.json 2>/dev/null || true)
  else
    SERVED=$(sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
      /tmp/dio_models.json | head -1)
  fi
  if [ -z "$SERVED" ]; then
    echo "  FAIL: could not read served model id from $URL/v1/models"
    echo "        response was: $(head -c 200 /tmp/dio_models.json)"
    FAIL=1
    continue
  fi
  echo "  models OK   served_model=$SERVED"

  # 2. Same weights everywhere. Different models across workers would make the
  #    latency difference a model difference, not a hardware difference.
  echo "$SERVED" >> /tmp/dio_served_models

  # 3. /metrics must exist, or the hybrid engine-telemetry path is silently dead
  #    and the paper's "hybrid" claim would be unsupported.
  if curl -sf --max-time 10 "$URL/metrics" -o /tmp/dio_metrics.txt; then
    RUN=$(grep -c 'vllm:num_requests_running' /tmp/dio_metrics.txt || true)
    KV=$(grep -c 'vllm:gpu_cache_usage_perc' /tmp/dio_metrics.txt || true)
    PFX=$(grep -c 'vllm:prefix_cache_queries' /tmp/dio_metrics.txt || true)
    echo "  metrics OK  running=$RUN kv=$KV prefix=$PFX"
    [ "$RUN" -eq 0 ] && { echo "  WARN: num_requests_running absent"; }
    [ "$PFX" -eq 0 ] && { echo "  WARN: prefix cache gauges absent (--enable-prefix-caching?)"; }
  else
    echo "  FAIL: /metrics unreachable — hybrid telemetry would be dead"
    FAIL=1
  fi

  # 4. One real completion, to prove inference works end to end (not just that
  #    the HTTP server is up), and to get a rough latency baseline.
  T0=$(date +%s%N)
  RESP=$(curl -sf --max-time 120 "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":16}" || echo "")
  if [ -z "$RESP" ]; then
    echo "  FAIL: chat completion request failed"
    FAIL=1
  else
    MS=$(( ($(date +%s%N) - T0) / 1000000 ))
    echo "  inference OK  ${MS}ms for 16 tokens"
  fi
done

echo
if [ -f /tmp/dio_served_models ]; then
  UNIQ=$(sort -u /tmp/dio_served_models | wc -l)
  if [ "$UNIQ" -ne 1 ]; then
    echo "FAIL: workers serve DIFFERENT models — latency gaps would not be hardware:"
    sort -u /tmp/dio_served_models
    FAIL=1
  else
    echo "all workers serve the same model: $(sort -u /tmp/dio_served_models)"
  fi
  rm -f /tmp/dio_served_models
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "PREFLIGHT PASSED — safe to run the experiment."
else
  echo "PREFLIGHT FAILED — fix the above before spending GPU-hours." >&2
fi
exit "$FAIL"
