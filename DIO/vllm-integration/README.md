# DIO v3 — vLLM Integration

> **DIO as the Driver, vLLM as the Engine.**

This module integrates DIO's predictive NLMS scheduler with [vLLM](https://github.com/vllm-project/vllm),
the industry-standard LLM inference engine. Instead of modifying vLLM's C++/CUDA kernel code, DIO
operates as a non-invasive control plane that wraps vanilla vLLM instances.

## Architecture

```
┌──────────────┐     HTTP /v1/completions    ┌──────────────┐
│  DIO Manager │ ←──── gRPC Predict ────→    │  vLLM Proxy  │ ──────→ │  vLLM Engine │
│  (Go, NLMS)  │                             │  (Sidecar)   │ ←────── │  (PagedAttn) │
│  :50055/:8085│                             │  :50060      │         │  :8000       │
└──────────────┘                             └──────────────┘         └──────────────┘
       ↑                                            │
       └────── Real telemetry: latency_ms, ─────────┘
               tokens_used, freeVRAM (NVML)
```

## Quick Start (Single GPU)

```bash
# Terminal 1: Start vLLM
python vllm_launcher.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --port 8000 --gpu-index 0

# Terminal 2: Start DIO Manager
cd ../  # DIO root
go run cmd/manager/main.go

# Terminal 3: Start the Proxy
python worker_proxy.py \
  --worker-id vllm-gpu0 \
  --port 50060 \
  --vllm-url http://localhost:8000 \
  --manager-addr localhost:50055 \
  --gpu-index 0 --tier large

# Terminal 4: Send a request through DIO
curl -X POST http://localhost:8085/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "model_id": "llama", "tier": "large"}'
```

## Quick Start (Docker — Heterogeneous Demo)

```bash
# Requires 2 NVIDIA GPUs
docker compose -f docker-compose.yaml up

# The demo starts:
#   - vLLM "A100" (fast, 85% mem) on GPU 0
#   - vLLM "T4" (slow, 60% mem, 2.5x penalty) on GPU 1
#   - DIO Manager with NLMS scheduler
#   - Two proxy sidecars
#
# DIO will automatically detect the speed differential
# and route ~75% of traffic to the fast worker.
```

## How It Works

1. **vLLM Engine** serves the model with PagedAttention and continuous batching
2. **Worker Proxy** (this module) translates DIO's gRPC `Predict` → vLLM's HTTP `/v1/completions`
3. **Real Telemetry** is extracted:
   - `latency_ms` from wall-clock timing
   - `ttft_ms` from streaming SSE (Time-To-First-Token)
   - `tokens_used` from vLLM's `usage.total_tokens` (not estimated)
   - `free_vram_mb` from NVML (`pynvml`)
4. **DIO's NLMS** learns the real ms/token slope of each vLLM instance
5. **Roofline Admission** uses real VRAM pressure to prevent OOM

## GPU Requirements

| Setup | Minimum GPU | Notes |
|-------|------------|-------|
| Single worker (3B model) | 1× GPU, 16GB VRAM | L4, T4 (16GB), A10 |
| Single worker (8B model) | 1× GPU, 24GB VRAM | L4 (24GB), A100 |
| Heterogeneous demo | 2× GPUs | Any mix works |
| Full benchmark suite | 1× A100 (80GB) | 4 workers fit in 80GB |

## File Reference

| File | Purpose |
|------|---------|
| `worker_proxy.py` | Core sidecar: gRPC server → vLLM HTTP client, NVML telemetry |
| `vllm_launcher.py` | Starts vLLM as a managed subprocess with correct config |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container image for the proxy sidecar |
| `docker-compose.yaml` | Full demo stack (Manager + 2 engines + 2 proxies) |
| `benchmarks/` | Head-to-head comparison scripts |

## Lightning.ai Deployment

For running on Lightning.ai studios:

```bash
# 1. Create a studio with GPU (A100 recommended)
# 2. Clone the repo
git clone https://github.com/nisaral/DIO.git && cd DIO

# 3. Install vLLM
pip install vllm pynvml grpcio grpcio-tools

# 4. Build the Go manager
go build -o dio-manager cmd/manager/main.go

# 5. Follow the Quick Start above
```
