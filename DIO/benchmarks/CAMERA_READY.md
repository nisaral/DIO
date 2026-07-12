# Camera-ready suite (novelty + e2e)

## What was implemented (code)

| Novelty | Implementation |
|---------|----------------|
| **Dual-timescale NLMS** | `NLMS_MODE=DUAL\|SINGLE`, dual blend in `PerWorkerPredictor`, MAPE history at `/debug/predictions` |
| **Admission goodput** | Reject if `min_w S_w > DIO_SLO_MS`; counters at `/debug/admission` |
| **Joint cost** | Queue + latency + tier + VRAM soft/hard + cache; ablations via `DIO_ABLATION` |
| **STATIC baseline** | `SCHEDULER_STRATEGY=STATIC` freezes slopes (offline calib) |
| **Artifact** | `camera_ready_suite.py`, `demo_openai_client.py`, `docker-compose.camera.yaml` |

### Env reference

```bash
SCHEDULER_STRATEGY=NLMS|RLS|STATIC|RoundRobin|LeastLoaded
NLMS_MODE=DUAL|SINGLE
DIO_ABLATION=full|no_queue|no_vram|no_vram_hard|no_tier|no_cache|no_dual
DIO_SLO_MS=5000
DIO_ADMISSION_OFF=0|1
STATIC_SLOPE=1.2
STATIC_INTERCEPT=80
```

### Debug APIs

- `GET /healthz`
- `GET /debug/metrics` — workers + admission + ablation + prediction MAPE
- `GET /debug/admission`
- `GET /debug/predictions?limit=5000`
- `POST /debug/reset_stats`
- `POST /debug/chaos/vram` — `{"worker_id":"...","free_vram_mb":1500}`

---

## Run on a GPU cluster IDE (Lightning / RunPod / 2×T4 / etc.)

### 0. Setup once

```bash
cd DIO   # this repo folder
go version          # need Go 1.21+
python -m pip install -U grpcio grpcio-tools torch transformers locust matplotlib requests
# proto python stubs already in benchmarks/
```

### 1. Mock-only novelty (no GPU, ~10–30 min)

Proves N1–N5 without model download:

```bash
python benchmarks/camera_ready_suite.py --mode mock --quick
# or full novelty:
python benchmarks/camera_ready_suite.py --mode mock --only novelty
```

### 2. Real 2× GPU (Kaggle 2×T4 / dual L4 / A100+T4)

```bash
export HF_TOKEN=...   # if gated models
python benchmarks/camera_ready_suite.py \
  --mode real \
  --gpus 0,1 \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --slo-ms 30000 \
  --e2e-duration 120 \
  --concurrency 6 \
  --seeds 3
```

For 8B (when VRAM allows one worker per GPU):

```bash
python benchmarks/camera_ready_suite.py --mode real --gpus 0,1 \
  --model meta-llama/Llama-3.1-8B-Instruct --vram-mb 20000 --slo-ms 60000
```

### 3. OpenAI artifact demo

```bash
# terminal 1: manager + mocks via suite partial or docker
go build -o dio-manager ./cmd/manager
SCHEDULER_STRATEGY=NLMS NLMS_MODE=DUAL ./dio-manager &
python benchmarks/worker_gpu.py --worker-id w0 --port 50060 --mock --latency-profile a100_hf_fast &
python benchmarks/worker_gpu.py --worker-id w1 --port 50061 --mock --latency-profile t4_emulated_slow &
python benchmarks/demo_openai_client.py
```

### 4. Outputs

```
benchmarks/results_camera_ready/
  summary.json              # all metrics
  tables.json               # flat table for paper
  predictions_n1_dual.json  # MAPE traces
  predictions_n1_single.json
  figures/n1_dual_vs_single.png
  figures/n2_admission_goodput.png
  figures/n4_ablations.png
  figures/e1_main_p99.png
  logs/
```

Copy numbers from `summary.json` into the paper; regenerate figures into `figs/`.

---

## Experiment map → paper claims

| Exp | Claim | Figure |
|-----|-------|--------|
| N1 | Dual µ better under burst + thermal | MAPE / p99 dual vs single |
| N2 | Admission raises goodput under overload | goodput_under_slo_rps on vs off |
| N2b | Hard VRAM blocks long prompts | 503 + rejected_vram |
| N3 | Tier joint cost | routing_counts small vs large |
| N4 | Ablations + RR/LL/RLS/STATIC | p99 bar chart |
| N5 | Control plane scales | rps with many mocks |
| E1 | Main e2e | strategy table mean±std |

---

## Optional theory (paste into paper appendix)

Scalar NLMS with step $\mu\in(0,2)$ is BIBO stable for the normalized update (Haykin). Dual timescales use $\mu_f=0.1$, $\mu_s=0.01$, $s^{\mathrm{eff}}=\alpha s_f+(1-\alpha)s_s$ with $\alpha=0.8$. Steady-state excess MSE scales roughly as $\mu/(2-\mu)$; the slow path lowers prediction variance under drift while the fast path tracks bursty interference. Admission rejects when $\min_w S_w > \mathrm{SLO}$, converting overload into controlled 503s so goodput of *accepted* work stays higher than uncontrolled queues.
