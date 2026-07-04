# Lightning AI — Complete Run Guide

Use this after committing and pulling on your Lightning studio.

## GPU choice

| Option | Recommendation |
|--------|----------------|
| **2× A100 80GB** @ ~5.73 credits/hr | **Pick this** — 1 worker per GPU, real heterogeneity, paper-grade |
| 1× H100 | Works, but both workers share one GPU (less heterogeneity story) |

**Estimated runtime:** ~1.5–2 hours → **~9–12 credits** for the full suite.

---

## Part 1 — On your Windows machine (commit & push)

```powershell
cd C:\Users\nisar\OneDrive\Desktop\Go-serve

git add DIO/ README.md README_USAGE.md paper_drafts_latex/
git status   # verify no .exe binaries staged

git commit -m "DIO Cloud: NLMS+RLS scheduler, OpenAI gateway, Lightning benchmark scripts"

git push origin feat-v2-rls-scheduler
```

If push fails, use your branch name: `git branch` to check.

---

## Part 2 — On Lightning AI studio

### 2a. Create studio
- Template: **GPU** → **2× A100 80GB** (or 1× H100 if 2× unavailable)
- Open **Terminal** in the studio (no SSH needed)

### 2b. Clone and setup (one-time)

```bash
cd /teamspace/studios/this_studio

# Clone (replace with your repo URL)
git clone https://github.com/YOUR_USER/Go-serve.git
cd Go-serve
git checkout feat-v2-rls-scheduler
git pull

cd DIO
bash benchmarks/real_world/setup_lightning.sh
```

### 2c. HuggingFace (required for Llama 3.2)

```bash
pip install huggingface_hub
huggingface-cli login
# Paste token from https://huggingface.co/settings/tokens
# Accept Llama license on huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

### 2d. Preflight ONLY (5 min — run this first)

```bash
cd /teamspace/studios/this_studio/Go-serve/DIO
chmod +x benchmarks/preflight_gpu.sh benchmarks/run_lightning_full.sh
bash benchmarks/preflight_gpu.sh
```

**Must see before continuing:**
- `[PASS] PyTorch CUDA: 2 device(s)`
- `[PASS] Worker log confirms CUDA load`
- `[PASS] Latency XXXXms in plausible GPU range`
- `[PASS] GPU utilization detected`
- Final line: `Safe to run: bash benchmarks/run_lightning_full.sh`

If any `[FAIL]` — stop. Do not burn credits.

### 2e. Run full benchmark

```bash
bash benchmarks/run_lightning_full.sh 2>&1 | tee lightning_run.log
```

Ends with `validate_results.py` — must say **ADMISSIBLE**.

**Shorter run (save credits):** ShareGPT only, ~45 min

```bash
export DATASETS_OVERRIDE=sharegpt.jsonl   # not wired yet — use budget script:
bash benchmarks/run_lightning_budget.sh 2>&1 | tee lightning_budget.log
```

---

## Part 3 — What gets tested (GPU-required)

| Test | Script section | Needs GPU | New feature tested |
|------|----------------|-----------|-------------------|
| Manager build | `go build` | No | — |
| OpenAI `/v1/chat/completions` | smoke_tests | After manager up | **NEW gateway** |
| `/debug/metrics` | smoke_tests | No | **NEW dashboard API** |
| T7 scalability | 32 mock workers | No | Control-plane O(1) |
| T1 NLMS convergence | 25 probes | **Yes** | **NLMS hyperparams 0.1/0.01/0.005** |
| T2 heterogeneity | 2 workers | **Yes** | 2×A100 routing / NLMS vs RR |
| ShareGPT matrix | Locust 120s | **Yes** | NLMS, **RLS**, RR, LL |
| arXiv matrix | Locust 120s | **Yes** | VRAM roofline admission |
| Azure matrix | Locust 120s | **Yes** | Bursty decode |
| results_summary.json | analyze_results.py | No | **Reproducible pipeline** |

**Not in automated script (manual if time):**
- 503 admission: flood with `LOCUST_USERS=100` until 503s appear
- Autoscaler: `AUTOSCALER_ENABLED=true` (needs Docker socket — skip on Lightning)

---

## Part 4 — Download results

### Option A — Lightning file browser
Download these folders/files:
- `Go-serve/DIO/benchmarks/results_final/` (all `*.csv`)
- `Go-serve/DIO/benchmarks/results_summary.json`
- `Go-serve/DIO/benchmarks/results_table.tex`
- `Go-serve/figs/` (if figures generated)
- `Go-serve/DIO/lightning_run.log`

### Option B — Git push from studio

```bash
cd /teamspace/studios/this_studio/Go-serve
git add DIO/benchmarks/results_final DIO/benchmarks/results_summary.json figs/
git commit -m "Lightning benchmark results $(date +%Y-%m-%d)"
git push
```

Then on Windows: `git pull`

---

## Part 5 — On Windows after download

```powershell
cd DIO
python benchmarks/real_world/analyze_results.py
python benchmarks/generate_figures_from_json.py --out ../figs
```

Send me `results_summary.json` or push to git — I'll sync the paper.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `go: command not found` | `export PATH=/usr/local/go/bin:$PATH` |
| Llama download 403 | `huggingface-cli login` + accept model license |
| OOM on GPU | `export MODEL_ID=meta-llama/Llama-3.2-1B-Instruct` |
| Only 30 requests in 90s | Normal if model cold; use `LOCUST_DURATION=120s` (default) |
| Workers not registering | `tail -50 worker_0.log` — wait 20s per worker for model load |

---

## File checklist (all in repo after pull)

```
DIO/
├── cmd/manager/main.go              # Manager + OpenAI API
├── internal/scheduler/
│   ├── loadbalancer.go              # NLMS scheduler
│   ├── rls_predictor.go             # RLS baseline
│   ├── constants.go                 # Hyperparams
│   └── admission.go                 # 503 logic
├── internal/api_gateway/openai.go   # /v1/chat/completions
├── benchmarks/
│   ├── run_lightning_full.sh        # ← RUN THIS (full suite)
│   ├── run_lightning_budget.sh      # ← shorter/cheaper
│   ├── real_world/setup_lightning.sh
│   ├── real_world/analyze_results.py
│   ├── real_world/locustfile.py
│   ├── generate_figures_from_json.py
│   └── data/sharegpt.jsonl          # datasets
└── dashboard/static/                # optional UI test at :8085/dashboard/
```