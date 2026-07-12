# DIO Architecture

How **dio-serve** works as a **scalable wrap-around** control plane for LLM inference engines (vLLM, SGLang, TGI, Ollama, …).

---

## 1. One-sentence design

> **DIO never replaces the engine.** It sits in front of one or more already-running OpenAI-compatible servers, learns each backend’s latency online, and routes (or rejects) every request using a joint cost + admission policy.

---

## 2. Layered architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  Clients                                                        │
│  OpenAI SDK · LangChain · curl · custom apps                    │
│  base_url = http://dio-host:8085/v1                             │
└────────────────────────────┬────────────────────────────────────┘
                             │  HTTPS / HTTP
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  DIO Gateway  (this package: FastAPI + uvicorn)                 │
│  ─────────────────────────────────────────────────────────────  │
│  OpenAI surface                                                 │
│    POST /v1/chat/completions                                    │
│    POST /v1/completions                                         │
│    GET  /v1/models                                              │
│  Ops / research surface                                         │
│    GET  /healthz                                                │
│    GET  /debug/metrics · /debug/admission · /debug/predictions  │
│    POST /debug/backends · /debug/chaos/vram · /debug/reset_stats│
│                                                                 │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────────┐  │
│  │ Token approx │→ │ Scheduler      │→ │ BackendPool         │  │
│  │ N ≈ |p|/4    │  │ dual NLMS      │  │ httpx forward       │  │
│  │ + max_tokens │  │ joint cost     │  │ health / timeouts   │  │
│  └──────────────┘  │ admission      │  └──────────┬──────────┘  │
│                    └────────────────┘             │             │
└───────────────────────────────────────────────────┼─────────────┘
                                                    │ HTTP OpenAI API
                    ┌───────────────────────────────┼────────────────┐
                    ▼                               ▼                ▼
             ┌────────────┐                  ┌────────────┐   ┌────────────┐
             │ Engine A   │                  │ Engine B   │   │ Engine N   │
             │ vLLM GPU0  │                  │ vLLM GPU1  │   │ SGLang /…  │
             │ :8000      │                  │ :8001      │   │ :800N      │
             └────────────┘                  └────────────┘   └────────────┘
                    ▲                               ▲
                    │  telemetry on response path   │
                    │  (latency, usage tokens)      │
                    └─────────── feedback ──────────┘
                               (NLMS update)
```

### Separation of concerns

| Layer | Owns | Does **not** own |
|-------|------|------------------|
| **Client** | prompts, app logic | GPUs, routing |
| **DIO Gateway** | routing, admission, learning | model weights, kernels |
| **Engine (vLLM…)** | batching, KV cache, decode | cluster-wide placement |

That split is what makes DIO **portable**: any engine that speaks OpenAI HTTP is a backend.

---

## 3. Request path (happy path)

```text
1. Client POST /v1/chat/completions
2. Gateway extracts prompt (+ optional X-DIO-Tier header)
3. Token estimate: N = len(prompt)//4 + max_tokens
4. Scheduler.pick(prompt, tier, N)
      for each healthy backend w:
          ŷ_w = s_eff_w · N + b_w          # dual-timescale NLMS
          wait_w = (pending_w / B) · avg_w
          S_w = wait_w + ŷ_w + tier + vram − cache
      w* = argmin S_w  (feasible only)
5. If min S_w > SLO → HTTP 503 + Retry-After   (admission)
6. Else forward HTTP to backend w* /v1/chat/completions
7. On completion: measure e2e_ms, read usage.total_tokens if present
8. Scheduler.feedback(w*, e2e_ms, tokens) → NLMS update; pending--
9. Response returned with X-DIO-Backend / X-DIO-E2E-Ms (+ dio.decision JSON)
```

### Failure path (admission)

```text
min_w S_w > SLO  OR  all workers hard-blocked (VRAM / tier)
        →  AdmissionError
        →  503 Service Unavailable
        →  Retry-After: estimated seconds
        →  counters: rejected_slo | rejected_vram
```

This is intentional: **shedding load preserves goodput** of admitted traffic.

---

## 4. Core algorithms

### 4.1 Dual-timescale NLMS (per backend)

Model:

\[
\hat{y} = s^{\mathrm{eff}} \cdot N + b
\]

Updates on each completion (error \(e = y - \hat{y}\)):

\[
s^{\mathrm{fast}} \leftarrow s^{\mathrm{fast}} + \mu_f \frac{e}{N}, \quad
s^{\mathrm{slow}} \leftarrow s^{\mathrm{slow}} + \mu_s \frac{e}{N}, \quad
b \leftarrow b + \mu_b e
\]

Effective slope:

\[
s^{\mathrm{eff}} = \alpha\, s^{\mathrm{fast}} + (1-\alpha)\, s^{\mathrm{slow}}
\qquad (\alpha=0.8,\; \mu_f=0.1,\; \mu_s=0.01)
\]

| Mode | Behavior | Use |
|------|----------|-----|
| `dual` | fast + slow | production / paper default |
| `single` | only fast | ablation: dual-µ claim |
| `static` | frozen slopes | offline profile baseline |

**Complexity:** \(O(1)\) arithmetic per completion; no matrix invert (unlike multi-feature RLS).

### 4.2 Joint cost

\[
S_w = \underbrace{\frac{Q_w}{B}\bar{t}_w}_{\text{wait}}
    + \underbrace{\hat{y}_w}_{\text{exec}}
    + \text{tierCost}
    + \text{vramCost}
    - \text{cacheBonus}
\]

| Term | Purpose |
|------|---------|
| wait | continuous-batching style queue amortization |
| exec | learned latency |
| tier | multi-model capability (hard block large→small) |
| vram | soft pressure &lt; 4 GB free; hard block &lt; 2.4 GB + long prompt |
| cache | prefix affinity (first 100 chars hash) |

### 4.3 Admission (goodput optimizer)

\[
\textbf{admit } w^\star \iff
  w^\star \text{ feasible }
  \;\wedge\;
  S_{w^\star} \le \mathrm{SLO}
\]

Otherwise **reject** (503). Ablatable via `admission_off=True` for pure routing A/B.

---

## 5. Scalability model

DIO scales along three axes:

### 5.1 Horizontal backends (data plane)

```text
        ┌── vLLM replica 1
DIO ────┼── vLLM replica 2
        ├── vLLM replica … N
        └── other engine
```

- Each backend is an independent process/GPU/node.
- Register with `Backend(id, base_url, …)` or `POST /debug/backends`.
- Hot registration supported; no restart required for new workers.
- Practical N: tens to low hundreds of backends on one gateway process (scan is \(O(N)\) per request, pure Python arithmetic).

### 5.2 Gateway process (control plane)

| Scale | Recommendation |
|-------|----------------|
| Lab / paper (2–8 GPUs) | single `dio serve` process |
| Production (many tenants) | multiple DIO replicas behind L4/NLB + sticky sessions optional |
| Very large N | shard backends by model or region; one DIO per shard |

Control plane is **stateless w.r.t. model weights** (state = NLMS slopes + pending counts). For multi-replica DIO, either:

- **shared nothing** (each replica learns independently — simplest), or  
- future: Redis-backed slope sync (not required for paper / typical deploys).

### 5.3 Why wrap-around scales better than engine forks

| Approach | Upgrade vLLM | Multi-engine | Ops |
|----------|--------------|--------------|-----|
| Fork vLLM scheduler | painful | hard | high |
| **DIO wrap** | drop-in new vLLM image | mix vLLM+SGLang | low |

Engines keep continuous batching, PagedAttention, etc. DIO only does **placement + admission**.

### 5.4 Throughput notes

- Per-request work in DIO is microseconds–low milliseconds of Python + one HTTP hop.
- End-to-end latency remains **engine-bound** (decode).
- For extreme QPS, run gateway with multiple uvicorn workers **only if** you accept split learning state, or stick to 1 worker + async I/O (default).

---

## 6. Deployment patterns

### Pattern A — Laptop / CI (mock)

```text
dio demo  →  in-process mock backends (fast + slow) + gateway
```

No GPU, no vLLM.

### Pattern B — Single node, multi-GPU (most common)

```text
GPU0: vLLM :8000
GPU1: vLLM :8001
CPU:  dio serve -b :8000 -b :8001 → :8085
```

### Pattern C — Multi-node

```text
node-a:8000  ─┐
node-b:8000  ─┼→  dio serve (any node or small VM) → clients
node-c:8000  ─┘
```

Backends are just URLs. Network RTT becomes part of learned \(b_w\) / slope.

### Pattern D — Kubernetes (conceptual)

```text
Service/dio  →  Deployment/dio-gateway
Backend Service/vllm-a, Service/vllm-b  (ClusterIP)
DIO backends: http://vllm-a:8000, http://vllm-b:8000
```

Engines use GPU node pools; DIO uses CPU pool.

---

## 7. Comparison: Python package vs Go plane

| | **dio-serve (Python)** | **DIO Go manager** |
|--|------------------------|--------------------|
| Install | `pip install` | Go build + binaries |
| Integration | OpenAI HTTP wrap | gRPC workers + gateway |
| Best for | research on any cloud, product demos | ultra-low overhead control plane studies |
| Location | `dio-serve/` | `DIO/` |

Both implement the **same scheduling ideas**. Prefer **dio-serve** for wrap-around adoption.

---

## 8. Security & multi-tenant notes

- DIO does not authenticate by default — put it **behind** your API gateway / mTLS / VPC.
- Treat `/debug/*` as **admin-only** in production (network policy or auth middleware).
- Do not expose debug chaos endpoints publicly.

---

## 9. Extensibility hooks

| Hook | How |
|------|-----|
| New engine | Run OpenAI server; add `Backend(base_url=…)` |
| Custom cost | Fork `Scheduler._score` or open an issue for plugins |
| VRAM telemetry | Update `free_vram_mb` via `POST /debug/chaos/vram` or future NVML scraper |
| Ablations | `ablation=` / `DIO_ABLATION=` |

---

## 10. Mental model for contributors

```text
backends.py   → what we wrap (URLs)
scheduler.py  → how we choose (NLMS + cost + admit)
gateway.py    → OpenAI HTTP surface + forward + feedback
cli.py        → dio serve | demo | bench-smoke
config.py     → env knobs
```

If you understand **pick → forward → feedback**, you understand DIO.
