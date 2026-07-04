# DIO Cloud — Quick Start

DIO Cloud is a drop-in inference routing layer: OpenAI-compatible API, NLMS predictive scheduling, VRAM-safe admission, and a live dashboard.

## One-command deploy (mock workers)

```powershell
cd DIO
docker compose -f docker-compose.cloud.yaml up --build
```

Wait for:
- `DIO Manager HTTP API listening at :8085`
- `MockWorker Registered successfully`

## Endpoints

| URL | Purpose |
|-----|---------|
| http://localhost:8085/v1/chat/completions | OpenAI-compatible chat API |
| http://localhost:8085/api/generate | Legacy benchmark API |
| http://localhost:8085/dashboard/ | Live routing dashboard |
| http://localhost:8085/debug/metrics | Worker NLMS state + routing log |

## OpenAI SDK example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8085/v1", api_key="dio-local")
resp = client.chat.completions.create(
    model="llama-3.2-3b",
    messages=[{"role": "user", "content": "Hello from DIO Cloud"}],
)
print(resp.choices[0].message.content)
```

## curl example

```bash
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-DIO-Tier: small" \
  -d '{"model":"llama","messages":[{"role":"user","content":"Explain NLMS"}]}'
```

## Chaos demo (dashboard)

1. Open http://localhost:8085/dashboard/
2. Click **Inject VRAM Pressure** on `worker_slow`
3. Click **Burst 10 Requests** — watch NLMS steer traffic to `worker_fast`

## Scheduler strategies

Set `SCHEDULER_STRATEGY` on the manager:

- `NLMS` (default) — dual-timescale predictive routing
- `RLS` — recursive least squares baseline for ablation
- `RoundRobin`, `LeastLoaded` — baselines

## Reproducible benchmark figures

```powershell
cd DIO
.\benchmarks\make_results.ps1
```

Outputs:
- `benchmarks/results_summary.json`
- `figs/fig_6_line_comparison_clean.png` (from real CSVs)

## Autoscaling (optional)

```powershell
$env:AUTOSCALER_ENABLED="true"
go run ./cmd/manager/main.go
```

Spawns `dio-worker` containers when average queue depth exceeds 5 (max 1 spawn / 30s).