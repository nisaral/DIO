# Who uses DIO, and for what?

## One-line product definition

**DIO is a smart front door for LLM APIs:** you keep running **vLLM / SGLang / TGI / Ollama** (or any OpenAI-compatible server), and DIO sits in front as a **library or process** that **learns which GPU is fast**, **routes requests there**, and **rejects overload** before the fleet melts.

It is **not** a model, **not** a quantizer, and **not** a replacement for vLLM.

---

## What problem it solves

| Pain | Without DIO | With DIO |
|------|-------------|----------|
| 2+ GPUs / nodes | Round-robin or hand-written LB | Online ms/token learning + cost ranking |
| Mixed GPUs (T4 + A100) | Slow GPU becomes straggler | Traffic shifts to faster workers |
| Long prompts / VRAM | OOM / thrash | Soft VRAM penalty + hard admission |
| Traffic spikes | Queue forever → everyone slow | 503 when predicted cost > SLO |
| Multi-model | Manual ports / paths | Tier constraints in one router |
| Engine upgrades | Re-fork / re-patch | Change only the backend URL |

---

## Who uses it

### 1. ML / platform engineers (primary industry user)

- Run **several vLLM pods** on a cluster  
- Want **one OpenAI `base_url`** for the app team  
- Need **heterogeneous hardware** (cheap T4 + fast A100) without custom C++  

**How:**

```bash
# engines already up on :8000 :8001
dio serve -b http://gpu0:8000 -b http://gpu1:8001 --port 8085
# apps point OpenAI base_url → http://dio:8085/v1
```

### 2. Application / backend developers

- Building chatbots, agents, internal copilots  
- Use **OpenAI SDK / LangChain / LiteLLM**  
- Do not want to know which GPU served the request  

**How:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://dio:8085/v1", api_key="unused")
```

### 3. Researchers / paper authors (you)

- Need **reproducible ablations**: dual vs single NLMS, −queue, −VRAM, RR, RLS  
- Need to run on **any vendor box** without rewriting orchestration  

**How:**

```python
from dio import Scheduler, AblationFlags
# or
python scripts/run_publishable_suite.py --quick
```

### 4. Students / hackathon / indie builders

- One machine, two GPUs, want “something better than RR” today  
- `pip install` + wrap local vLLM  

---

## Who does **not** need DIO (alone)

| Need | Better tool |
|------|-------------|
| Faster kernels / batching | vLLM, TensorRT-LLM, SGLang |
| 4-bit / AWQ / GPTQ weights | engine quant (vLLM args) — **not DIO** |
| Train models | trainers / Hugging Face |
| Single GPU, low traffic | plain vLLM is enough |
| Global multi-region CDN | cloud LB + multiple DIO regions |

**“Quantized / efficient” for DIO means:** the **router** is tiny (tens of µs per pick), not that it quantizes model weights. **Model quant stays in the engine.** DIO stays a thin control plane so you can still use quantized vLLM freely.

---

## Real use-case stories

### Use case A — Startup chat product on 2× GPU cloud

- GPU0: Llama-3.1-8B  
- GPU1: same model  
- App has spikes at demo day  

**DIO role:** one URL, NLMS learns if one GPU is hotter/slower, admission returns 503 with Retry-After instead of multi-minute queues.

### Use case B — Lab with A100 + leftover T4

- Long research jobs on A100, short chat on either  
- RR pins half the chat to T4 → bad p99  

**DIO role:** higher ms/token on T4 → cost pushes interactive traffic to A100; optional tier=`large` for heavy jobs only on A100.

### Use case C — Multi-model “small router + big reasoner”

- Backend A: 3B / small tier  
- Backend B: 70B / large tier  

**DIO role:** hard block `large` requests off small workers; soft prefer small models for `small` tier.

### Use case D — Paper evaluation

- Same suite on laptop (mocks) and on 2×T4 (real vLLM)  
- Report MAPE dual vs single, goodput with admission, ablations  

**DIO role:** library API + `run_publishable_suite.py` without reimplementing the control plane.

---

## How people integrate it

```text
┌─────────────┐     OpenAI HTTP      ┌─────────────┐     OpenAI HTTP     ┌──────────┐
│ Your app    │ ───────────────────► │ dio-serve   │ ─────────────────► │ vLLM …   │
│ LangChain   │   base_url=/v1       │ (library or │   /v1/chat/...     │ engines  │
└─────────────┘                      │  `dio serve`)│                    └──────────┘
                                     └─────────────┘
```

1. **Process mode:** `dio serve -b …` (sidecar / small VM)  
2. **Library mode:** `DIOGateway(...).run()` inside your Python service  
3. **Scheduler-only mode:** embed `Scheduler.pick/feedback` in a custom proxy  

---

## Efficiency claims (what “optimized” means here)

| Property | Typical value | Why it matters |
|----------|---------------|----------------|
| Pick + feedback | ~tens of µs (pure Python) | Negligible vs LLM decode (10–10⁵ ms) |
| Memory | few KB per backend (slopes + counters) | Millions of backends theoretically; hundreds practical |
| Update complexity | O(1) NLMS | Not O(d²) RLS per feature |
| Hot path allocs | bounded deques for logs | Stable under load |
| Engine work | unchanged | Quant/tensor-parallel stay in vLLM |

DIO is **optimized as a control plane**, not as a GPU runtime.

---

## Summary table

| Question | Answer |
|----------|--------|
| What is it? | Predictive, admitting load balancer for LLM HTTP APIs |
| Who ships it? | Platform eng / ML ops / researchers |
| Who consumes it? | Any app that already speaks OpenAI API |
| What do you wrap? | vLLM and friends (URLs) |
| What do you not wrap? | Training, weight quant, CUDA kernels |
| Why novel? | Dual NLMS + joint cost + SLO admission, non-invasive |
| Why practical? | `pip install` + `base_url` change |
