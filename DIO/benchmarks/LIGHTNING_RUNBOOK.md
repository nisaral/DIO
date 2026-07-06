# Lightning AI Benchmark Runbook (Credit-Efficient)

## Why your last run was slow

`unified_master_test.py` spawns **4 real HuggingFace workers**, each loading Llama-3.2-3B on `cuda:0`. On a single GPU they fight for VRAM and thrash — that explains ~50–80s p99 with only ~30 requests in 90s.

**Fix:** use **1 real + 1 calibrated mock** on one GPU (never 2 full models). Mock uses `heterogeneity_profiles.json` (intercept/slope/jitter/thermal), not a flat multiplier.

---

## GPU tiers (pick based on credits)

| Tier | GPU | Est. cost | What you can prove | Paper value |
|------|-----|-----------|-------------------|-------------|
| **A (minimum)** | 1× L4 24GB | ~$0.50–1/hr | NLMS vs RR on ShareGPT; T2 heterogeneity (1 real + 1 mock) | Enough for credible scheduling comparison |
| **B (recommended)** | 1× A100 80GB | ~$1.5–3/hr | 2 real workers, 3B model, ShareGPT + arXiv | Best bang-for-buck for paper |
| **C (stretch)** | 2× L4 or A10 | 2× tier A | Real 2-GPU heterogeneity (no `latency_mult` emulation) | Strong "real-world proof" paragraph |

**Do not need:** 4× GPU, H100, or multi-hour full 9-cell matrix on first pass.

---

## Recommended path (limited credits)

### Session 1 — ~45–60 min on **1× A100 80GB**

```bash
# On Lightning studio terminal
cd /teamspace/studios/this_studio
git clone <your-repo-url> Go-serve   # or git pull
cd Go-serve/DIO
bash benchmarks/real_world/setup_lightning.sh
bash benchmarks/run_lightning_budget.sh
```

Downloads when done:
- `benchmarks/results_final/*.csv`
- `benchmarks/results_summary.json`

### Session 2 (optional) — ~30 min on **1× L4**

Run mock-heavy tests only (T7 scalability, admission) — no real model load.

---

## What `run_lightning_budget.sh` runs

| Test | Workers | Time | Purpose |
|------|---------|------|---------|
| T7 scalability | 32 mock | ~2 min | Control-plane overhead (CPU only) |
| T2 heterogeneity | 1 real + 1 calibrated mock (`t4_vs_a100`) | ~5 min | Routing split NLMS vs RR |
| ShareGPT | 1 real + 1 calibrated mock | ~15 min × 4 strategies | **Core paper numbers** |
| arXiv (optional) | 1 real + 1 calibrated mock | ~15 min × 4 strategies | Long-context stress |

**Total:** ~60–90 min GPU time (one A100 session).

---

## HuggingFace + Lightning setup

1. Create Lightning studio with **A100 80GB** (or L4 if broke)
2. `huggingface-cli login` — Llama 3.2 needs gated access
3. Ensure Go 1.22+ and Python 3.10+

---

## After the run (on your laptop or studio)

```bash
cd DIO
python benchmarks/real_world/analyze_results.py
python benchmarks/generate_figures_from_json.py --out ../figs
```

Commit `results_final/` + `results_summary.json` → paper auto-syncs.

---

## SSH vs scripts

| Approach | When |
|----------|------|
| **Scripts only (recommended)** | Clone repo on studio, run `setup_lightning.sh` + `run_lightning_budget.sh`, download results |
| **SSH** | Only if you want to debug mid-run; not required |

You do **not** need me connected via SSH. Run the scripts, paste back `results_summary.json` or push to git.

---

## Environment variables (tuning)

```bash
export MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"   # use 1B on L4 to save VRAM
export NUM_REAL_WORKERS=2
export LOCUST_USERS=15
export LOCUST_DURATION=120s
bash benchmarks/run_lightning_budget.sh
```