# GPU cluster grand validation — runbook

One script for **all real-engine / multi-seed tests** needed for paper camera-ready and production confidence.

## Script

```bash
cd dio-serve
pip install -e .
python scripts/run_gpu_cluster_validation.py --help
```

## What it runs

| ID | Test | Needs GPU? | Output for paper |
|----|------|------------|------------------|
| P0 | Env probe (CUDA, nvidia-smi, dio) | probe only | env block |
| G1 | Real engine smoke through DIO | yes | pass/fail |
| G2 | Multi-seed strategy matrix (NLMS/RR/LL/…) | yes | **mean±std p99** |
| G3 | Dual-backend hetero NLMS vs RR (delay-proxy throttle if mult>1) | 2 engines | frac to fast, p99 impr%, MAPE |
| G4 | Dual vs single NLMS | yes | MAPE / p99 |
| G5 | Admission ON vs OFF | yes | 503 counts, p99 |
| G6 | TTFT fields when engine provides them | yes | ttft_p50/p99 if present |
| G7 | CPU T2 multiseed (always) | no | Table t2_multiseed |

## Recipes

### A) Recommended cluster (2× GPU + vLLM) — homogeneous Regime A

```bash
export HF_TOKEN=...   # if gated model
python scripts/run_gpu_cluster_validation.py \
  --engine-mode vllm \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --gpus 0,1 \
  --seeds 3 \
  --requests-per-seed 40 \
  --max-tokens 32 \
  --max-model-len 2048 \
  --gpu-mem-util 0.85 \
  --strategies nlms,round_robin,least_loaded \
  --out results_gpu_cluster
```

### A2) One more round — real dual-GPU **heterogeneity** (throttled peer, 5 seeds)

Homogeneous G2 is not enough for hetero claims. This keeps both engines **real**
(vLLM on two T4s) but multiplies the second backend’s wall-clock via
`latency_delay_proxy.py` so NLMS sees slope skew. Prefer **5 seeds**.

```bash
python scripts/run_gpu_cluster_validation.py \
  --engine-mode vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --gpus 0,1 \
  --skip-g2 --skip-g4 --skip-g5 \
  --hetero-slow-mult 2.0 \
  --hetero-seeds 5 \
  --requests-per-seed 30 \
  --max-tokens 32 \
  --out results_gpu_cluster_hetero
```

Report from `summary.json` → `G3_hetero`: `nlms_frac_e0`, `p99_improvement_pct`,
and **`nlms_mape`** (honest absolute error). Framing: ranking quality, not low MAPE.

### B) Engines already running

```bash
# start your own vLLM/SGLang on :8000 :8001 first
python scripts/run_gpu_cluster_validation.py \
  --engine-mode external \
  --backends http://127.0.0.1:8000,http://127.0.0.1:8001 \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --seeds 3 \
  --requests-per-seed 40 \
  --hetero-slow-mult 2.0 \
  --hetero-seeds 5
```

### C) Laptop / RTX 4050 (small model, HF server)

```bash
# Prefer CUDA PyTorch for GPU; otherwise falls back to CPU (still real weights)
python scripts/run_gpu_cluster_validation.py \
  --engine-mode hf \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --gpus 0 \
  --seeds 3 \
  --requests-per-seed 12 \
  --max-tokens 24 \
  --hetero-slow-mult 2.0 \
  --quick
```

### D) Paper minimum bar (do this before submission)

```bash
# 1) Homogeneous multi-seed matrix (Regime A)
python scripts/run_gpu_cluster_validation.py \
  --engine-mode vllm \
  --model <YOUR_MODEL> \
  --gpus 0,1 \
  --seeds 3 \
  --requests-per-seed 50 \
  --strategies nlms,round_robin,least_loaded,rls

# 2) Throttled real-GPU hetero (Regime C hardware complement)
python scripts/run_gpu_cluster_validation.py \
  --engine-mode vllm \
  --model <YOUR_MODEL> \
  --gpus 0,1 \
  --skip-g2 --skip-g4 --skip-g5 \
  --hetero-slow-mult 2.0 --hetero-seeds 5 \
  --out results_gpu_cluster_hetero
```

Copy numbers from `results_gpu_cluster/paper_snippets.md` into the LaTeX tables.
**Never** mix Regime B Locust ShareGPT seconds (67s/81s) with Regime A dual-T4 ms.

## Outputs

```text
results_gpu_cluster/
  summary.json         # full nested results
  tables.csv           # flat for Excel
  paper_snippets.md    # mean±std ready for LaTeX
  logs/                # vllm_*, dio_*, hf_* logs
```

## Prerequisites

| Item | Notes |
|------|--------|
| `pip install -e .` | dio-serve |
| GPUs | `nvidia-smi` works |
| vLLM mode | `pip install vllm` + CUDA torch |
| HF mode | `transformers`, `torch` (CUDA recommended) |
| Disk | model weights download |
| Ports | free `18000+`, `19000+` (configurable) |

## Failures

- **G1 smoke fails** → engines not healthy; read `logs/*.log`
- **OOM** → lower `--gpu-mem-util`, smaller `--model`, or `--max-model-len`
- **Single GPU** → G3 uses `--hetero-slow-mult` peer (real tokens + delay); prefer 2 GPUs for claims
- **Gated HF models** → set `HF_TOKEN`

## Related scripts

| Script | Role |
|--------|------|
| `run_gpu_cluster_validation.py` | **This grand GPU suite** |
| `run_paper_experiments.py` | CPU algorithmic suite (no GPU required) |
| `validate_local_realtime.py` | Smaller local real-model check |
| `run_t2_multiseed.py` | CPU multi-seed hetero table only |
