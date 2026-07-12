# Production load balancing with DIO

## Is this mock code?

**No.** Production path:

1. You run **real** engines (vLLM, SGLang, TGI, Ollama, …) on GPUs.  
2. You register their HTTP URLs as `Backend(...)`.  
3. DIO **only** does: pick worker → HTTP forward → measure latency → NLMS update → admit/reject.

`MockBackendServer` exists **only** for CI and `dio demo`. Do not use mocks in production.

```python
# PRODUCTION
from dio import DIOGateway, Backend

gw = DIOGateway(
    backends=[
        Backend(id="gpu0", base_url="http://10.0.0.1:8000", tier="small"),
        Backend(id="gpu1", base_url="http://10.0.0.2:8000", tier="large"),
    ],
    strategy="nlms",
    nlms_mode="dual",
    slo_ms=30_000,
    admission_off=False,  # shed overload
)
gw.run(host="0.0.0.0", port=8085)
```

```bash
# PRODUCTION CLI
dio serve \
  -b gpu0=http://10.0.0.1:8000 \
  -b gpu1=http://10.0.0.2:8000 \
  --strategy nlms --slo-ms 30000 --port 8085
```

### Production features already in the library

| Feature | Behavior |
|---------|----------|
| Dual-timescale NLMS | Learns real e2e latency per backend under live traffic |
| Joint cost | Latency + queue + tier + VRAM |
| Admission | HTTP 503 + Retry-After when predicted cost > SLO |
| Health loop | Probes `/health` or `/v1/models`; marks dead backends unhealthy |
| 5xx handling | Marks backend unhealthy after server errors |
| Connection pool | httpx keepalive (200 max connections) |
| Hot register | `POST /debug/backends` or `gw.add_backend(...)` |
| Streaming | Proxies SSE when `stream: true` |
| Multi-engine styles | `api_style=openai` (default) or `tgi_generate` |

---

## Why “OpenAI API” — not only “OpenAI models”

**OpenAI-compatible HTTP is the industry wire format for self-hosted LLMs**, not a lock-in to GPT models.

| Engine | Typical models | Wire API DIO uses |
|--------|----------------|-------------------|
| vLLM | Llama, Mistral, Qwen, Phi, Gemma, … | `/v1/chat/completions` |
| SGLang | same | OpenAI-compatible |
| TGI | HF models | OpenAI mode or `/generate` (`api_style="tgi_generate"`) |
| Ollama | local models | `/v1/chat/completions` |
| LocalAI / LiteLLM | many | OpenAI-compatible |

**Clients** (LangChain, OpenAI Python SDK, Cursor, custom apps) already speak this API.

**Models** are chosen by the engine:

```json
{ "model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [...] }
```

DIO is **model-agnostic**. Llama vs Mistral is a backend config issue, not a DIO limitation.

### What about Anthropic / Gemini / Azure OpenAI cloud APIs?

Those are **different public HTTP schemas**. Options:

1. Put **LiteLLM** or a small adapter in front of them so they look OpenAI-compatible, then point DIO at that adapter.  
2. Extend `Backend.api_style` with more adapters (PRs welcome).  

Self-hosted research and most on-prem production fleets already use OpenAI-compatible engines — that is DIO’s primary market.

### TGI native example

```python
Backend(
    id="tgi0",
    base_url="http://10.0.0.3:8080",
    api_style="tgi_generate",  # maps OpenAI chat → /generate
)
```

Clients still call DIO with OpenAI chat format; DIO translates.

---

## Production checklist

- [ ] Real engines listening; smoke `curl http://engine:8000/v1/models`  
- [ ] `dio serve -b ...` with **no** mocks  
- [ ] `admission_off=False` and realistic `slo_ms`  
- [ ] Put DIO behind your auth gateway (DIO does not replace IAM)  
- [ ] Restrict `/debug/*` to private network  
- [ ] Multiple DIO replicas OK (independent learners) or one DIO per model pool  
- [ ] Optional: NVML sidecar to update free VRAM via `POST /debug/chaos/vram` or custom telemetry  

---

## Load path (production)

```text
App QPS
  → DIO (CPU, µs–ms)
  → chosen engine (GPU, 10ms–10s)
  → feedback into NLMS
```

Bottleneck remains **GPU decode**, not DIO.
