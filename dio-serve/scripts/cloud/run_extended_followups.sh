#!/bin/bash
# After priority followups: ShareGPT replay + optional 3rd T4 N=3 study.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
LOG=~/extended_followups.log
exec > >(tee -a "$LOG") 2>&1
echo "=== extended followups $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Wait for priority pipeline if still running
for i in $(seq 1 180); do
  if grep -q 'ALL PRIORITY FOLLOWUPS DONE' ~/priority_followups.log 2>/dev/null; then
    echo "priority done"
    break
  fi
  if ! pgrep -f run_priority_followups.sh >/dev/null 2>&1 \
     && ! pgrep -f run_gpu_abc_suite.py >/dev/null 2>&1 \
     && ! pgrep -f run_post_d_priority.py >/dev/null 2>&1; then
    echo "priority processes gone"
    break
  fi
  echo "waiting priority... ($i)"
  sleep 60
done

T4U=http://127.0.0.1:8000
A30U=http://216.48.191.114:8000
MODEL=Qwen/Qwen2.5-3B-Instruct
REPO=~/Go-serve/dio-serve
# shellcheck disable=SC1091
source ~/dio-gw-venv/bin/activate
cd "$REPO"

curl -sf --max-time 5 "$T4U/v1/models" >/dev/null
curl -sf --max-time 8 "$A30U/v1/models" >/dev/null
echo "engines OK"

# ---------- ShareGPT-style replay on dual T4+A30 ----------
echo "=== ShareGPT hetero replay (dual) ==="
python scripts/run_sharegpt_hetero.py \
  --backends "t4=$T4U,a30=$A30U" \
  --vram "t4=16000,a30=24000" \
  --model "$MODEL" --tokenizer "$MODEL" \
  --n-prompts 60 --seeds 3 --max-tokens 96 \
  --strategies nlms,round_robin \
  --out /root/results_sharegpt_hetero \
  || echo "WARN sharegpt failed"

# ---------- Optional 3rd T4 ----------
# Write public IP of second T4 into /root/t4b.url as http://IP:8000 after setup_worker+start_vllm
# Or set env T4B_URL. Script polls up to ~90 min while you create the node.
echo "=== N=3 wait for second T4 ==="
T4B=""
for i in $(seq 1 90); do
  if [ -n "${T4B_URL:-}" ]; then
    T4B="$T4B_URL"
  elif [ -f /root/t4b.url ]; then
    T4B=$(tr -d ' \n\r' </root/t4b.url)
  fi
  if [ -n "$T4B" ]; then
    if curl -sf --max-time 5 "${T4B}/v1/models" >/dev/null; then
      echo "T4B ready: $T4B"
      break
    fi
    echo "T4B url set but not ready yet: $T4B ($i)"
  else
    echo "no T4B yet — create 3rd T4, run setup_worker+start_vllm, write URL to /root/t4b.url ($i/90)"
  fi
  T4B=""
  sleep 60
done

if [ -n "$T4B" ]; then
  echo "=== N=3 hetero t4a+t4b+a30 ==="
  python scripts/run_n3_hetero.py \
    --backends "t4a=$T4U,t4b=$T4B,a30=$A30U" \
    --vram "t4a=16000,t4b=16000,a30=24000" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --seeds 3 --requests-per-seed 30 \
    --strategies nlms,rls,round_robin \
    --fast-id a30 \
    --out /root/results_n3_hetero \
    || echo "WARN n3 failed"

  echo "=== ShareGPT on N=3 if possible ==="
  python scripts/run_sharegpt_hetero.py \
    --backends "t4a=$T4U,t4b=$T4B,a30=$A30U" \
    --vram "t4a=16000,t4b=16000,a30=24000" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --n-prompts 40 --seeds 2 --max-tokens 64 \
    --strategies nlms,round_robin \
    --out /root/results_sharegpt_n3 \
    || echo "WARN sharegpt n3 failed"
else
  echo "SKIP N=3 — no second T4 within wait window"
fi

echo "=== ALL EXTENDED FOLLOWUPS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -la /root/results_sharegpt_hetero /root/results_n3_hetero /root/results_sharegpt_n3 2>/dev/null || true
