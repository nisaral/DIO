# DIO Serve — Performance & Usage Guide

Style notes follow docs for systems like [vLLM](https://docs.vllm.ai): what it is, prerequisites, how to call APIs, measured behavior, and limits.

---

## 1. What DIO does

**DIO (Distributed Inference Orchestrator)** is a **control-plane load balancer** for LLM serving.

| DIO does | DIO does **not** |
|----------|------------------|
| Sit in front of engines as OpenAI-compatible gateway | Replace vLLM / SGLang / kernels |
| Learn per-backend latency online (dual-timescale NLMS) | Quantize model weights |
| Route with joint cost (latency + queue + tier + VRAM) | Train models |
| Admit / reject under overload (SLO) | Own GPU memory layouts |
| Work with **any model** the backend serves (Llama, Mistral, Qwen, …) | Speak proprietary cloud-only APIs without an adapter |

```text
  App / OpenAI SDK / LangChain
            │  POST /v1/chat/completions
            ▼
     ┌──────────────┐
     │  DIO :8085   │  pick → forward → feedback (NLMS)
     └──────┬───────┘
        ┌───┴────┐
        ▼        ▼
   Engine A   Engine B     ← real vLLM / HF / TGI / Ollama
```

---

## 2. Prerequisites

### Software

| Component | Requirement |
|-----------|-------------|
| Python | **3.9+** (3.11 tested) |
| Package | `pip install -e dio-serve` (or install from repo) |
| Engines | At least one OpenAI-compatible HTTP server |
| OS | Linux / Windows / macOS |

```bash
cd dio-serve
python -m pip install -e .
dio version
```

### Hardware

| Role | Minimum | Notes |
|------|---------|--------|
| **DIO gateway** | CPU only | ~tens of µs per schedule decision |
| **Engines** | GPU recommended | DIO does not need a GPU itself |
| Multi-backend demo | 1–2 GPUs or 2 engine processes | Heterogeneous routing needs ≥2 backends |

### Supported engines (production)

| Engine | How to expose | `Backend.api_style` |
|--------|---------------|---------------------|
| **vLLM** | `python -m vllm.entrypoints.openai.api_server --port 8000` | `openai` (default) |
| **SGLang** | OpenAI-compatible server | `openai` |
| **Ollama** | `ollama serve` + `/v1` | `openai` |
| **TGI** | OpenAI mode or native | `openai` or `tgi_generate` |
| **HF transformers** (dev) | `scripts/real_engine_server.py` | `openai` |
| **LM Studio** | Enable local server (often `:1234`) | `openai` |

**Models:** whatever the engine loads (not limited to OpenAI-hosted models).  
**Wire format:** OpenAI-compatible HTTP is the industry standard for self-hosted LLMs.

---

## 3. Quick start

### A. Production wrap (real engines)

```bash
# Terminal 1–2: engines (example vLLM)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct --port 8000
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct --port 8001

# Terminal 3: DIO
dio serve \
  -b gpu0=http://127.0.0.1:8000 \
  -b gpu1=http://127.0.0.1:8001 \
  --strategy nlms --nlms-mode dual \
  --slo-ms 60000 --port 8085
```

### B. Python library

```python
from dio import DIOGateway, Backend

gw = DIOGateway(
    backends=[
        Backend(id="gpu0", base_url="http://127.0.0.1:8000", api_style="openai"),
        Backend(id="gpu1", base_url="http://127.0.0.1:8001", api_style="openai"),
    ],
    strategy="nlms",
    nlms_mode="dual",
    slo_ms=30_000,
    admission_off=False,
    port=8085,
)
gw.run()
```

### C. Client (any OpenAI client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8085/v1", api_key="unused")
r = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",  # model name known to engine
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=64,
)
print(r.choices[0].message.content)
```

```bash
curl http://127.0.0.1:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

Response headers:

| Header | Meaning |
|--------|---------|
| `X-DIO-Backend` | Chosen backend id |
| `X-DIO-E2E-Ms` | Gateway-measured end-to-end ms |
| `Retry-After` | Present on **503** admission reject |

---

## 4. Core library API (functions you’ll call)

### 4.1 High-level gateway

| Call | Purpose |
|------|---------|
| `DIOGateway(backends=[...], strategy=..., slo_ms=...)` | Build gateway + scheduler |
| `gw.run(host=..., port=...)` | Serve forever (uvicorn) |
| `gw.add_backend(Backend(...))` | Hot-register a new engine |
| `gw.app` | Mount FastAPI app in your process |
| `gw.scheduler.metrics()` | Slopes, MAPE, admission counters |

### 4.2 Backend

```python
Backend(
    id="gpu0",
    base_url="http://10.0.0.5:8000",
    tier="small",           # or "large"
    model=None,             # optional override of request model
    api_style="openai",     # or "tgi_generate"
    api_key=None,           # Bearer token if engine needs it
    total_vram_mb=24000,
    free_vram_mb=24000,
)
```

| Method | Returns |
|--------|---------|
| `chat_url()` | Full chat endpoint URL |
| `completions_url()` | Completions URL |
| `auth_headers()` | Auth headers for engine |

### 4.3 Scheduler (advanced / research)

| Call | Purpose |
|------|---------|
| `Scheduler(strategy="nlms", dual=True, ...)` | Create router |
| `sched.register(id, tier=..., total_vram_mb=...)` | Add worker state |
| `sched.pick(prompt, tier=..., tokens=...)` | Choose backend (may raise `AdmissionError`) |
| `sched.feedback(id, e2e_ms, tokens)` | NLMS update after completion |
| `sched.release(id)` | Drop pending without learning (error path) |
| `sched.set_healthy(id, bool)` | Drain / restore |
| `sched.set_vram(id, free_mb)` | Update VRAM telemetry |
| `sched.metrics()` | Full observability dict |
| `sched.reset_stats()` | Clear counters |

### 4.4 Strategies

| `strategy` | Behavior |
|------------|----------|
| `nlms` | Dual/single NLMS + joint cost (**default, production**) |
| `rls` | 2×2 RLS baseline |
| `static` | Frozen offline slopes |
| `round_robin` | Classic RR |
| `least_loaded` | Min in-flight count |

### 4.5 CLI

| Command | Purpose |
|---------|---------|
| `dio serve -b URL ...` | Production gateway |
| `dio demo` | Zero-GPU smoke (mock engines) |
| `dio bench-smoke` | Quick NLMS vs RR on mocks |
| `dio version` | Package version |

### 4.6 HTTP ops endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/healthz` | Liveness |
| GET | `/v1/models` | Aggregate models |
| POST | `/v1/chat/completions` | Chat (main path) |
| POST | `/v1/completions` | Completions |
| GET | `/debug/metrics` | Workers + NLMS + admission |
| GET | `/debug/admission` | Goodput / reject counters |
| GET | `/debug/predictions` | MAPE samples |
| POST | `/debug/backends` | Hot-add backend |
| POST | `/debug/reset_stats` | Clear counters |

---

## 5. Measured performance (this repo)

### 5.1 Control-plane microbench (library only)

| Metric | Measured |
|--------|----------|
| `pick` + `feedback` | **~20–40 µs** per request (Python, 8–16 backends) |
| Memory per backend | ~KB (slopes + counters) |

Inference still dominates (ms–s). DIO is not the bottleneck.

### 5.2 Real model end-to-end (validated on author machine)

**Setup (2026-07-14 local run):**

| Item | Value |
|------|--------|
| GPU hardware present | NVIDIA GeForce **RTX 4050 Laptop** (6 GB) via `nvidia-smi` |
| PyTorch in env | `2.11.0+cpu` (**CUDA build not installed** → engines ran on **CPU**) |
| Model | **Qwen/Qwen2.5-0.5B-Instruct** (real HF weights, real `generate`) |
| Engines | 2× `scripts/real_engine_server.py` (OpenAI-compatible) |
| Slow engine | Same model + **2× post-decode mult** (still real tokens) |
| Gateway | `dio serve` NLMS / RoundRobin |

**Results:**

| Path | Status | Notes |
|------|--------|--------|
| Direct engine chat | **HTTP 200** | ~2.1–2.4 s (CPU 0.5B, max_tokens≈16–20) |
| DIO → engine (NLMS) | **HTTP 200** | All requests OK |
| NLMS routing (n=8) | **e0: 5, e1: 3** | Prefers faster backend |
| Round-robin (n=8) | **e0: 4, e1: 4** | Even split (baseline) |
| NLMS p50 | **~2.07 s** | Dominated by model decode, not DIO |

Artifact: `results_validation/local_realtime.json`

**Reproduce:**

```bash
cd dio-serve
pip install -e .
# Real HF engines + DIO (no mock sleep-only backends)
python scripts/validate_local_realtime.py --n 8 --model Qwen/Qwen2.5-0.5B-Instruct
# Single engine / lower RAM:
python scripts/validate_local_realtime.py --single-engine --n 6
```

### 5.3 Enabling the RTX 4050 for engines

`nvidia-smi` already sees the 4050. This environment had **CPU-only PyTorch**, so validation used **real models on CPU**. For GPU engines:

```bash
# Install CUDA PyTorch matching your driver (example cu124)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Or use vLLM / LM Studio (GPU) and only wrap with DIO:
dio serve -b http://127.0.0.1:1234   # LM Studio local server
```

DIO itself stays on CPU either way.

### 5.4 Paper algorithmic suite

```bash
python scripts/run_paper_experiments.py --quick
```

Includes dual vs single NLMS, admission, tiers, ablations, scale, gateway HTTP (see `results_paper/`).

---

## 6. Algorithm summary (what “performance” means)

### Dual-timescale NLMS

\[
\hat{y} = s^{\mathrm{eff}} N + b,\quad
s^{\mathrm{eff}} = 0.8\,s_f + 0.2\,s_s
\]

Updates each completion with \(\mu_f=0.1\), \(\mu_s=0.01\) — **O(1)** time.

### Joint cost

\[
S_w = \mathrm{wait}_w + \hat{y}_w + \mathrm{tier} + \mathrm{vram} - \mathrm{cache}
\]

Route to \(\arg\min_w S_w\).

### Admission

Reject with **503** if \(\min_w S_w > \mathrm{SLO}\) (or hard VRAM/tier block).

---

## 7. Configuration reference

| Env / flag | Default | Meaning |
|------------|---------|---------|
| `DIO_STRATEGY` / `--strategy` | `nlms` | Router policy |
| `DIO_NLMS_MODE` / `--nlms-mode` | `dual` | Dual vs single µ |
| `DIO_SLO_MS` / `--slo-ms` | `5000` | Admission threshold (raise for long decode) |
| `DIO_ADMISSION_OFF` / `--admission-off` | false | Disable 503 rejects |
| `DIO_ABLATION` / `--ablation` | `full` | Paper ablations |
| `--port` | `8085` | Gateway listen port |

---

## 8. Limits & best practices

1. **One learning state per gateway process** — multi-replica DIO = independent learners (OK for most deploys).  
2. **Protect `/debug/*`** — admin-only on private networks.  
3. **Auth** — put API keys / mTLS in front of DIO; DIO is not an IAM product.  
4. **SLO units** — `slo_ms` is on **predicted wait+exec**, not TTFT alone; set high enough for your model (e.g. 30–60 s for large models).  
5. **Mocks** — `dio demo` / `MockBackendServer` are for CI only, not production.  
6. **VRAM telemetry** — optional; set `free_vram_mb` via API or future NVML scraper for best admission accuracy.

---

## 9. Troubleshooting

| Symptom | Check |
|---------|--------|
| 502 from DIO | Engine URL / health; `curl engine:8000/v1/models` |
| 503 always | Raise `--slo-ms` or use `--admission-off` for debugging |
| Always one backend | Only one registered, or others unhealthy (`/debug/workers`) |
| Slow responses | Engine/model bound — check `X-DIO-E2E-Ms` vs engine logs |
| CUDA false | Install CUDA PyTorch or use vLLM/LM Studio for GPU engines |

---

## 10. Related docs

| Doc | Content |
|-----|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design & scale-out |
| [API.md](API.md) | Full class/method reference |
| [PRODUCTION.md](PRODUCTION.md) | Production LB checklist |
| [USE_CASES.md](USE_CASES.md) | Who uses DIO and how |
| [../README.md](../README.md) | Install & overview |

---

## 11. Validation checklist (green on this machine)

- [x] Library import `dio`  
- [x] Unit tests (`pytest tests/`)  
- [x] Paper suite (`run_paper_experiments.py --quick`)  
- [x] **Real model** OpenAI server (`real_engine_server.py` + Qwen-0.5B)  
- [x] **DIO NLMS** returns HTTP 200 with `X-DIO-Backend`  
- [x] Dual backends: NLMS skews to faster engine; RR splits evenly  
- [x] Metrics endpoint reports worker slopes / prediction stats  

For GPU-accelerated engines on RTX 4050: install CUDA PyTorch or start LM Studio/vLLM on GPU, then:

```bash
dio serve -b http://127.0.0.1:<engine-port>
```
