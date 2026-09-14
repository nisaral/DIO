#!/bin/bash
# Keep GPUs busy with paper-useful experiments until DONE file or max hours.
# Safe to leave unattended. Writes under /root/results_* and ~/autonomous_soak.log
set -u
export DEBIAN_FRONTEND=noninteractive
LOG=~/autonomous_soak.log
exec > >(tee -a "$LOG") 2>&1
echo "=== autonomous soak START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

T4U=http://127.0.0.1:8000
A30U=http://216.48.191.114:8000
MODEL=Qwen/Qwen2.5-3B-Instruct
REPO=~/Go-serve/dio-serve
MAX_HOURS="${MAX_HOURS:-6}"
END_TS=$(( $(date +%s) + MAX_HOURS * 3600 ))

engines_ok() {
  curl -sf --max-time 5 "$T4U/v1/models" >/dev/null \
    && curl -sf --max-time 8 "$A30U/v1/models" >/dev/null
}

wait_prior() {
  echo "waiting for priority + extended pipelines..."
  for i in $(seq 1 240); do
    if [ -f /root/STOP_SOAK ]; then echo "STOP_SOAK present"; return 1; fi
    pri=$(grep -c 'ALL PRIORITY FOLLOWUPS DONE' ~/priority_followups.log 2>/dev/null || echo 0)
    # also continue if follow processes dead for a while and we have abc+post_d
    if [ "$pri" -ge 1 ] 2>/dev/null; then
      if grep -q 'ALL EXTENDED FOLLOWUPS DONE' ~/extended_followups.log 2>/dev/null \
         || ! pgrep -f run_extended_followups.sh >/dev/null 2>&1; then
        # if extend still waiting only for t4b, don't block forever — start dual-only soaks
        if pgrep -f run_extended_followups.sh >/dev/null 2>&1; then
          if grep -q 'ShareGPT hetero replay' ~/extended_followups.log 2>/dev/null \
             || grep -q 'SKIP N=3' ~/extended_followups.log 2>/dev/null \
             || grep -q 'ALL EXTENDED' ~/extended_followups.log 2>/dev/null; then
            echo "extended far enough or done"
            return 0
          fi
          # priority done; if extend only polling t4b, run dual soaks in parallel path
          if grep -q 'no T4B yet' ~/extended_followups.log 2>/dev/null; then
            echo "extend waiting on T4B — starting dual-only soaks now"
            return 0
          fi
        else
          return 0
        fi
      fi
    fi
    # fallback: if post_d and abc summaries exist and no abc/post process
    if [ -f /root/results_gpu_abc_hetero/summary.json ] \
       && [ -f /root/results_post_d_priority/summary.json ] \
       && ! pgrep -f run_gpu_abc_suite.py >/dev/null 2>&1 \
       && ! pgrep -f run_post_d_priority.py >/dev/null 2>&1; then
      echo "priority artifacts present, processes idle"
      return 0
    fi
    sleep 60
  done
  echo "timeout waiting prior — proceeding carefully"
  return 0
}

still_time() { [ "$(date +%s)" -lt "$END_TS" ] && [ ! -f /root/STOP_SOAK ]; }

# shellcheck disable=SC1091
source ~/dio-gw-venv/bin/activate
cd "$REPO" || exit 1
wait_prior || exit 0

if ! engines_ok; then
  echo "FATAL engines down"; exit 1
fi

# ---- S1: longer ShareGPT dual (if not already large) ----
if still_time && [ ! -f /root/results_sharegpt_long/summary.json ]; then
  echo "=== S1 long ShareGPT dual ==="
  python scripts/run_sharegpt_hetero.py \
    --backends "t4=$T4U,a30=$A30U" \
    --vram "t4=16000,a30=24000" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --n-prompts 100 --seeds 3 --max-tokens 128 \
    --strategies nlms,rls,round_robin \
    --out /root/results_sharegpt_long \
    || echo "WARN S1 failed"
fi

# ---- S2: more NLMS/RLS seeds (offset 200) ----
if still_time && [ ! -f /root/results_extra_seeds_200/summary.json ]; then
  echo "=== S2 extra seeds offset 200 ==="
  python scripts/run_post_d_priority.py \
    --backends "t4=$T4U,a30=$A30U" \
    --vram "t4=16000,a30=24000" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --prior-summary /root/results_regime_d/summary.json \
    --out /root/results_extra_seeds_200 \
    --skip-discovery \
    --extra-seeds 5 --seed-offset 200 --requests-per-seed 40 \
    || echo "WARN S2 failed"
fi

# ---- S3: fixed long-decode D1-style (max_tokens high) via sharegpt-like short harness ----
if still_time && [ ! -f /root/results_long_decode_d1/summary.json ]; then
  echo "=== S3 long-decode strategy compare (synthetic chat, max_tokens=256) ==="
  python scripts/run_sharegpt_hetero.py \
    --backends "t4=$T4U,a30=$A30U" \
    --vram "t4=16000,a30=24000" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --n-prompts 40 --seeds 3 --max-tokens 256 \
    --strategies nlms,round_robin,least_loaded \
    --out /root/results_long_decode_d1 \
    || echo "WARN S3 failed"
fi

# ---- S4: second discovery with stricter margin ----
if still_time && [ ! -f /root/results_discovery_strict/summary.json ]; then
  echo "=== S4 discovery margin=0.25 ==="
  python scripts/run_post_d_priority.py \
    --backends "t4=$T4U,a30=$A30U" \
    --vram "t4=16000,a30=24000" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --prior-summary /root/results_regime_d/summary.json \
    --out /root/results_discovery_strict \
    --discovery-max 120 --discovery-margin 0.25 \
    --skip-extra-seeds \
    || echo "WARN S4 failed"
fi

# ---- S5: ABC again with longer tokens (mechanism suite stress) ----
if still_time && [ ! -f /root/results_gpu_abc_hetero_mt96/summary.json ]; then
  echo "=== S5 ABC seeds=3 max_tokens=96 ==="
  python scripts/run_gpu_abc_suite.py \
    --engine-mode external \
    --backends "$T4U,$A30U" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --seeds 3 --sessions 8 --turns 3 --max-tokens 96 --adm-n 30 \
    --out /root/results_gpu_abc_hetero_mt96 \
    || echo "WARN S5 failed"
fi

# ---- S6: N=3 if t4b appeared late ----
if still_time; then
  T4B=""
  if [ -n "${T4B_URL:-}" ]; then T4B=$T4B_URL; fi
  if [ -f /root/t4b.url ]; then T4B=$(tr -d ' \n\r' </root/t4b.url); fi
  if [ -n "$T4B" ] && curl -sf --max-time 5 "${T4B}/v1/models" >/dev/null; then
    if [ ! -f /root/results_n3_hetero/summary.json ]; then
      echo "=== S6 N=3 late bind ==="
      python scripts/run_n3_hetero.py \
        --backends "t4a=$T4U,t4b=$T4B,a30=$A30U" \
        --vram "t4a=16000,t4b=16000,a30=24000" \
        --seeds 3 --requests-per-seed 30 \
        --out /root/results_n3_hetero || true
    fi
  fi
fi

# ---- S7: pack all results for easy scp ----
echo "=== packing all results ==="
mkdir -p /tmp/all_dio_results
cp -a /root/results_* /tmp/all_dio_results/ 2>/dev/null || true
cp -a ~/regime_d_run.log ~/priority_followups.log ~/extended_followups.log ~/autonomous_soak.log \
  /tmp/all_dio_results/ 2>/dev/null || true
cd /tmp && tar czf /root/ALL_DIO_RESULTS.tgz all_dio_results
ls -lah /root/ALL_DIO_RESULTS.tgz

echo "=== AUTONOMOUS SOAK COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Touch /root/STOP_SOAK to stop early next time."
ls -la /root/results_*/summary.json 2>/dev/null | head -40
