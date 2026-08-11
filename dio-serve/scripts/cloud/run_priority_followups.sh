#!/bin/bash
# After Regime D: #1 ABC on real T4+A30, then discovery + extra seeds + calibration.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
LOG=~/priority_followups.log
exec > >(tee -a "$LOG") 2>&1
echo "=== priority followups $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

T4U=http://127.0.0.1:8000
A30U=http://216.48.191.114:8000
MODEL=Qwen/Qwen2.5-3B-Instruct
REPO=~/Go-serve/dio-serve

# engines
curl -sf --max-time 5 "$T4U/v1/models" >/dev/null
curl -sf --max-time 8 "$A30U/v1/models" >/dev/null
echo "engines OK"

# gateway venv
if [ ! -d ~/dio-gw-venv ]; then
  python3 -m venv ~/dio-gw-venv
fi
# shellcheck disable=SC1091
source ~/dio-gw-venv/bin/activate
pip install -q --upgrade pip
pip install -q -e "$REPO" httpx transformers

# ---------- #1 Hybrid / affinity / admission on real T4+A30 ----------
echo "=== P1: gpu_abc_suite on real hetero (seeds=5) ==="
cd "$REPO"
python scripts/run_gpu_abc_suite.py \
  --engine-mode external \
  --backends "$T4U,$A30U" \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --seeds 5 \
  --sessions 10 \
  --turns 4 \
  --max-tokens 48 \
  --adm-n 40 \
  --out /root/results_gpu_abc_hetero \
  || echo "WARN: abc suite failed"

# ---------- #2+#3+#4 discovery, extra seeds, calibration from D summary ----------
echo "=== P2-4: discovery + extra nlms/rls seeds + calibration ==="
python scripts/run_post_d_priority.py \
  --backends "t4=$T4U,a30=$A30U" \
  --vram "t4=16000,a30=24000" \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --prior-summary /root/results_regime_d/summary.json \
  --out /root/results_post_d_priority \
  --discovery-max 80 \
  --discovery-margin 0.15 \
  --extra-seeds 5 \
  --seed-offset 100 \
  --requests-per-seed 40 \
  || echo "WARN: post_d_priority failed"

# ---------- cheap SKU baseline (no DIO) ----------
echo "=== E1: raw SKU baseline ==="
python - <<'PY'
import json, time, statistics
from pathlib import Path
import httpx
MODEL = "Qwen/Qwen2.5-3B-Instruct"
backends = {"t4": "http://127.0.0.1:8000", "a30": "http://216.48.191.114:8000"}
out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL, "per_sku": {}}
prompt = "Explain KV cache reuse in transformer inference in one short paragraph."
for name, base in backends.items():
    lat = []
    ok = 0
    with httpx.Client(timeout=120.0) as c:
        for i in range(10):
            t0 = time.perf_counter()
            r = c.post(f"{base}/v1/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": f"{prompt} (i={i})"}],
                "max_tokens": 64, "temperature": 0.0,
            })
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                ok += 1
                lat.append(ms)
    lat_s = sorted(lat)
    def pct(p):
        if not lat_s: return None
        k = (p/100)*(len(lat_s)-1)
        lo, hi = int(k), min(len(lat_s)-1, int(k)+1)
        return lat_s[lo]*(1-(k-lo))+lat_s[hi]*(k-lo)
    out["per_sku"][name] = {
        "ok": ok, "n": 10,
        "mean_ms": statistics.mean(lat) if lat else None,
        "p50_ms": pct(50), "p99_ms": pct(99),
    }
    print(name, out["per_sku"][name])
t = out["per_sku"]["t4"]["mean_ms"]
a = out["per_sku"]["a30"]["mean_ms"]
out["mean_ratio_t4_over_a30"] = (t/a) if t and a else None
Path("/root/results_sku_baseline").mkdir(parents=True, exist_ok=True)
Path("/root/results_sku_baseline/summary.json").write_text(json.dumps(out, indent=2))
print("ratio T4/A30 mean", out["mean_ratio_t4_over_a30"])
PY

echo "=== ALL PRIORITY FOLLOWUPS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Dirs: results_regime_d results_gpu_abc_hetero results_post_d_priority results_sku_baseline"
ls -la /root/results_regime_d/summary.json /root/results_gpu_abc_hetero/summary.json /root/results_post_d_priority/summary.json /root/results_sku_baseline/summary.json 2>/dev/null || true
