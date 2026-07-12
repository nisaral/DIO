# DIO Serve — API Reference

Complete reference for the **`dio`** package after:

```bash
pip install -e dio-serve   # or: pip install dio-serve
```

```python
import dio
from dio import (
    Backend,
    BackendPool,
    DIOConfig,
    DIOGateway,
    Scheduler,
    DualTimescaleNLMS,
    RoutingDecision,
    AdmissionStats,
    AblationFlags,
)
```

---

## Package exports (`dio`)

| Symbol | Type | Description |
|--------|------|-------------|
| `Backend` | class | One OpenAI-compatible engine endpoint |
| `BackendPool` | class | Registry of backends + HTTP forward helpers |
| `DIOConfig` | class | Settings (env prefix `DIO_`) |
| `DIOGateway` | class | FastAPI gateway + scheduler + pool |
| `Scheduler` | class | NLMS / cost / admission router |
| `DualTimescaleNLMS` | class | Per-backend latency learner |
| `RoutingDecision` | dataclass | Cost breakdown for one pick |
| `AdmissionStats` | dataclass | Admit / reject / goodput counters |
| `AblationFlags` | dataclass | Paper ablation switches |
| `__version__` | str | Package version |

---

## 1. `Backend`

```python
Backend(
    id: str,
    base_url: str,
    tier: str = "small",
    model: Optional[str] = None,
    total_vram_mb: float = 24000.0,
    free_vram_mb: float = 24000.0,
    weight: float = 1.0,
    prior_slope: Optional[float] = None,
    prior_intercept: Optional[float] = None,
    labels: Dict[str, str] = {},
)
```

Describes one inference server (typically one vLLM process on one GPU).

| Field | Description |
|-------|-------------|
| `id` | Stable name used in routing logs (`X-DIO-Backend`) |
| `base_url` | Root URL, e.g. `http://127.0.0.1:8000` (no `/v1` suffix) |
| `tier` | Capability label: `"small"` or `"large"` (joint cost) |
| `model` | If set, overrides `model` field when forwarding |
| `total_vram_mb` / `free_vram_mb` | Memory headroom for VRAM cost / hard block |
| `weight` | Reserved for future weighted policies |
| `prior_slope` / `prior_intercept` | Optional STATIC warm-start hints |
| `labels` | Free-form metadata |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `chat_url()` | `str` | `{base}/v1/chat/completions` |
| `completions_url()` | `str` | `{base}/v1/completions` |
| `models_url()` | `str` | `{base}/v1/models` |
| `health_url()` | `str` | `{base}/health` |

---

## 2. `BackendPool`

```python
pool = BackendPool(backends: Optional[List[Backend]] = None)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `(backend: Backend) -> None` | Register backend |
| `get` | `(backend_id: str) -> Backend` | Lookup (raises `KeyError`) |
| `list` | `() -> List[Backend]` | All backends |
| `probe_health` | `async (client, backend_id) -> bool` | GET health/models |
| `forward_chat` | `async (client, backend_id, body, timeout) -> Response` | POST chat |
| `forward_completions` | `async (client, backend_id, body, timeout) -> Response` | POST completions |

---

## 3. `MockBackendServer`

In-process fake OpenAI server for CI / demos (no GPU).

```python
from dio.backends import MockBackendServer

mock = MockBackendServer(
    host="127.0.0.1",
    port=9001,
    latency_mult=1.0,
    decode_ms_per_token=12.0,
    name="mock",
)
await mock.start()
print(mock.base_url)   # http://127.0.0.1:9001
await mock.stop()
```

| Attribute / method | Description |
|--------------------|-------------|
| `base_url` | URL for `Backend(..., base_url=mock.base_url)` |
| `async start()` | Bind FastAPI/uvicorn mock |
| `async stop()` | Shutdown |

---

## 4. `DIOConfig`

Pydantic settings; env vars use prefix **`DIO_`**.

```python
cfg = DIOConfig(
    strategy="nlms",          # nlms|rls|static|round_robin|least_loaded
    nlms_mode="dual",         # dual|single
    ablation="full",
    slo_ms=5000.0,
    admission_off=False,
    host="0.0.0.0",
    port=8085,
    request_timeout_s=300.0,
    # NLMS hyperparams, VRAM limits, etc. — see config.py
)
```

| Field | Env | Default | Meaning |
|-------|-----|---------|---------|
| `strategy` | `DIO_STRATEGY` | `nlms` | Router policy |
| `nlms_mode` | `DIO_NLMS_MODE` | `dual` | Dual vs single µ |
| `ablation` | `DIO_ABLATION` | `full` | Cost-term ablation |
| `slo_ms` | `DIO_SLO_MS` | `5000` | Admission threshold |
| `admission_off` | `DIO_ADMISSION_OFF` | `False` | Disable 503 rejects |
| `host` / `port` | `DIO_HOST` / `DIO_PORT` | `0.0.0.0` / `8085` | Bind address |
| `request_timeout_s` | | `300` | Upstream HTTP timeout |
| `mu_fast` / `mu_slow` / `mu_bias` | | `0.1` / `0.01` / `0.005` | NLMS rates |
| `fast_slow_blend` | | `0.8` | α for \(s^{eff}\) |
| `vram_soft_limit_mb` | | `4096` | Soft VRAM penalty |
| `vram_hard_limit_mb` | | `2400` | Hard block threshold |
| `batch_size` | | `8` | Wait amortization B |
| `tier_mismatch_ms` | | `500` | small→large soft cost |
| `cache_bonus_ms` | | `200` | Prefix affinity bonus |

Helper:

```python
from dio.config import ablation_from_name
flags = ablation_from_name("no_queue")  # → AblationFlags
```

---

## 5. `AblationFlags`

```python
AblationFlags(
    name="full",
    disable_queue=False,
    disable_vram_soft=False,
    disable_vram_hard=False,
    disable_tier=False,
    disable_cache=False,
    single_timescale=False,
)
```

Used by `Scheduler` to zero out cost terms for paper tables.

---

## 6. `DualTimescaleNLMS`

Per-backend online latency model.

```python
pred = DualTimescaleNLMS(
    mu_fast=0.1,
    mu_slow=0.01,
    mu_bias=0.005,
    blend=0.8,
    initial_slope=0.1,
    initial_intercept=50.0,
    dual=True,
    frozen=False,
    tier="small",
    total_vram_mb=24000.0,
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `effective_slope` | `() -> float` | \(s^{eff}\) dual or single |
| `mode` | `() -> str` | `"DUAL"` \| `"SINGLE"` \| `"STATIC"` |
| `estimate` | `(tokens: int) -> (exec_ms, avg_ms)` | Predict latency |
| `update` | `(actual_ms, tokens) -> dict` | NLMS update; returns err stats |
| `snapshot` | `() -> dict` | slopes, MAE, MAPE, pending, VRAM |

Mutable fields (thread-safe via internal lock): `pending`, `free_vram_mb`, `healthy`, `tier`.

---

## 7. `SimpleRLS` (internal baseline)

2×2 recursive least squares predictor used when `strategy="rls"`.

| Method | Description |
|--------|-------------|
| `estimate(tokens)` | Predict |
| `update(actual_ms, tokens)` | Covariance update |

---

## 8. `RoutingDecision`

```python
@dataclass
class RoutingDecision:
    worker_id: str
    exec_ms: float
    wait_ms: float
    tier_cost_ms: float
    vram_cost_ms: float
    cache_bonus_ms: float
    total_ms: float
    tokens: int
    strategy: str = "nlms"

    def as_dict(self) -> dict: ...
```

---

## 9. `AdmissionStats`

Counters (mutated by `Scheduler`):

| Field | Meaning |
|-------|---------|
| `admitted` | Successful picks |
| `rejected_slo` | Rejected because min score > SLO |
| `rejected_vram` | Hard VRAM / feasibility blocks |
| `rejected_no_worker` | No healthy backends |
| `completed_under_slo` / `completed_over_slo` | Completions vs SLO |
| `completed_total` | Finished requests with feedback |
| `sum_e2e_ms` | Sum of e2e latencies |

```python
stats.snapshot(slo_ms, admission_enabled) -> dict
```

---

## 10. `AdmissionError`

Raised by `Scheduler.pick` when request must be rejected.

```python
except AdmissionError as e:
    e.retry_after_sec   # int, for Retry-After header
    str(e)              # human message
```

Gateway maps this to **HTTP 503**.

---

## 11. `Scheduler`

Core router. You normally use it **via `DIOGateway`**, but it is fully usable standalone (unit tests, custom servers).

```python
sched = Scheduler(
    strategy="nlms",
    dual=True,
    ablation=None,
    slo_ms=5000.0,
    admission_off=False,
    batch_size=8.0,
    tier_mismatch_ms=500.0,
    cache_bonus_ms=200.0,
    vram_soft_mb=4096.0,
    vram_hard_mb=2400.0,
    mu_fast=0.1,
    mu_slow=0.01,
    mu_bias=0.005,
    blend=0.8,
    initial_slope=0.1,
    initial_intercept=50.0,
    static_slope=1.0,
    static_intercept=50.0,
    decision_log_size=200,
    pred_history_size=5000,
)
```

### Lifecycle methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `register` | `(worker_id, *, tier="small", total_vram_mb=24000, free_vram_mb=None)` | Add worker + predictor |
| `unregister` | `(worker_id)` | Remove worker |
| `set_vram` | `(worker_id, free_mb)` | Update free VRAM (telemetry / chaos) |
| `set_healthy` | `(worker_id, healthy: bool)` | Drain / restore worker |

### Routing methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `pick` | `(prompt, *, tier="small", tokens=None) -> (worker_id, RoutingDecision)` | Choose backend; may raise `AdmissionError` |
| `release` | `(worker_id)` | Decrement pending without learning (error path) |
| `feedback` | `(worker_id, e2e_ms, tokens, *, predicted_ms=None)` | NLMS update + goodput counters + pending-- |

### Observability

| Method | Returns | Description |
|--------|---------|-------------|
| `metrics` | `dict` | workers, admission, decisions, prediction MAPE |
| `reset_stats` | `None` | Clear admission + pred history + decision log |

### Strategies (`strategy=`)

| Value | Behavior |
|-------|----------|
| `nlms` | Dual/single NLMS + joint cost |
| `rls` | 2×2 RLS + joint cost |
| `static` | Frozen slopes + joint cost |
| `round_robin` | Cyclic healthy workers |
| `least_loaded` | Min `pending` |

---

## 12. `DIOGateway`

High-level product object: **scheduler + backend pool + FastAPI app**.

```python
gw = DIOGateway(
    backends=[Backend(id="gpu0", base_url="http://127.0.0.1:8000")],
    config=None,           # or DIOConfig(...)
    # or kwargs passed into DIOConfig:
    strategy="nlms",
    nlms_mode="dual",
    slo_ms=30_000,
    port=8085,
)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `DIOConfig` | Active settings |
| `pool` | `BackendPool` | Backends |
| `scheduler` | `Scheduler` | Router |
| `app` | `FastAPI` | ASGI app (mountable) |

| Method | Description |
|--------|-------------|
| `add_backend(backend)` | Hot-register backend + scheduler worker |
| `run(host=None, port=None, **uvicorn_kwargs)` | Blocking uvicorn serve |
| `_proxy_json` / `_proxy_stream` | Internal forward paths |

### Embed in your own ASGI stack

```python
from dio import DIOGateway, Backend
gw = DIOGateway(backends=[...])
# mount gw.app in another FastAPI, or:
# uvicorn.run(gw.app, host="0.0.0.0", port=8085)
```

---

## 13. HTTP API (after `dio serve` / `gw.run()`)

### OpenAI-compatible

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz`, `/health` | Liveness |
| GET | `/v1/models` | Aggregated models from backends |
| POST | `/v1/chat/completions` | Chat (JSON or SSE if `stream`) |
| POST | `/v1/completions` | Completions |

**Optional headers**

| Header | Meaning |
|--------|---------|
| `X-DIO-Tier` | `small` \| `large` capability hint |

**Response headers**

| Header | Meaning |
|--------|---------|
| `X-DIO-Backend` | Chosen backend id |
| `X-DIO-E2E-Ms` | Gateway-measured e2e latency |
| `Retry-After` | On 503 admission reject |

**Response body** (non-stream) may include:

```json
{
  "choices": [...],
  "usage": {...},
  "dio": {
    "backend_id": "gpu0",
    "e2e_ms": 412.3,
    "decision": {
      "exec_ms": 380.1,
      "wait_ms": 20.0,
      "tier_cost_ms": 0,
      "vram_cost_ms": 12.2,
      "cache_bonus_ms": 0,
      "total_ms": 412.3
    }
  }
}
```

### Debug / ops

| Method | Path | Description |
|--------|------|-------------|
| GET | `/debug/metrics` | Full scheduler snapshot |
| GET | `/debug/admission` | Goodput / reject counters |
| GET | `/debug/predictions?limit=N` | MAPE sample ring buffer |
| GET | `/debug/workers` | Worker list + slopes |
| POST | `/debug/reset_stats` | Clear counters |
| POST | `/debug/chaos/vram` | `{"worker_id","free_vram_mb"}` |
| POST | `/debug/backends` | Hot-add `{"id","base_url","tier?",...}` |

---

## 14. CLI (`dio` / `dio-serve`)

Installed console scripts: **`dio`**, **`dio-serve`**.

### `dio serve`

Wrap existing engines.

```bash
dio serve \
  -b gpu0=http://127.0.0.1:8000 \
  -b gpu1=http://127.0.0.1:8001;tier=large \
  --strategy nlms \
  --nlms-mode dual \
  --slo-ms 60000 \
  --port 8085
```

| Option | Description |
|--------|-------------|
| `-b / --backend` | Repeatable URL or `id=URL` or `id=URL;tier=large` |
| `--host` / `--port` | Bind |
| `--strategy` | Router policy |
| `--nlms-mode` | `dual` \| `single` |
| `--slo-ms` | Admission threshold |
| `--admission-off` | Never 503 for SLO |
| `--ablation` | Paper ablation name |
| `--tier` | Per-backend tiers (order matches `-b`) |
| `--vram` | Per-backend total VRAM MB |

### `dio demo`

Zero-GPU end-to-end: two mocks + gateway + traffic + metrics printout.

```bash
dio demo --port 8085 --duration 20
```

### `dio bench-smoke`

Quick NLMS vs round_robin on mocks.

```bash
dio bench-smoke -n 40 -c 4
```

### `dio version`

Print package version.

### Module form

```bash
python -m dio serve -b http://127.0.0.1:8000
```

---

## 15. Minimal end-to-end examples

### Library

```python
from dio import DIOGateway, Backend

gw = DIOGateway(
    backends=[
        Backend(id="gpu0", base_url="http://127.0.0.1:8000"),
        Backend(id="gpu1", base_url="http://127.0.0.1:8001"),
    ],
    strategy="nlms",
    nlms_mode="dual",
    slo_ms=30_000,
)
gw.run()
```

### Client

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8085/v1", api_key="unused")
r = client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Scheduler only (unit / research)

```python
from dio import Scheduler

s = Scheduler(strategy="nlms", dual=True, admission_off=True, slo_ms=1e9)
s.register("w0", tier="small")
wid, dec = s.pick("hello world", tokens=16)
s.feedback(wid, e2e_ms=120.0, tokens=16)
print(s.metrics()["prediction"]["mape_pct"])
```

---

## 16. Thread safety

- `Scheduler` and `DualTimescaleNLMS` use locks for concurrent requests (uvicorn async + thread pool).
- Prefer **one gateway process** with async I/O for shared learning state.
- Multiple gateway replicas = independent learners (still correct, less shared knowledge).

---

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md) — system design & scalability  
- [../README.md](../README.md) — install & quick start  
