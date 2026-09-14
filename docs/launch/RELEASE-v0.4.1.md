## What this is

DIO is a non-invasive LLM gateway. It learns each backend's latency online
(dual-timescale NLMS, `y = s*tokens + b`, no GPU profiling and no engine patches)
and uses that model to route across heterogeneous vLLM / SGLang / TGI / Ollama
engines, with SLO-aware admission and session affinity on top.

Point any OpenAI or Ollama client at it and keep the rest of your stack:

```bash
pip install -e dio-serve
dio-serve --config dio-serve/dio.example.yaml
```

## Highlights

- **Learns latency online.** Per-backend `slope*tokens + intercept`, updated from
  real request outcomes; optional hybrid inputs scraped from vLLM `/metrics`
  (KV-cache pressure, queue depth, prefix hit rate) with zero engine patches.
- **OpenAI + Ollama on the same port.** `/v1/chat/completions`, `/v1/completions`,
  `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/v1/models`, with SSE
  and ndjson streaming, tool calls, and usage/token parity across both paths.
- **SLO-aware admission** with an observable, honest policy: `empirical` (a
  ranking gate, default) or `strict` (a shedding contract) — and
  `rejected_slo_suppressed` so you can tell which one you are getting.
- **MCP server included**, so an AI IDE can query routing state directly.
- **Apache-2.0, 112 tests, ruff clean**, CI on Python 3.9-3.12.

## Fixed in this release

Six gaps that a dogfooding swarm hit on a live gateway, closed rather than
documented (details and the seeding tests in `tests/test_gap_fixes.py`):

- `admission.mode: strict` — sheds on the observed tail even when one backend is
  still the clear best option (#23).
- `/debug/affinity` reports `evictions` / `cache_size` / `capacity`, so a
  collapsing hit rate under prefix churn is attributable (#21).
- **Robust learner.** A contended sample is clipped against the low quartile of
  recent latencies; an *uncontended* one is taken at face value, because the
  scheduler knows how many requests were in flight when the sample was admitted.
  A contended burst used to blow the slope from 2.6 to 168 and predict 14.8 s for
  a 0.25 s request; it now stays at 2.6 / 0.44 s. `mae`/`mape` still count the raw
  error and `clipped_updates` reports when the guard fires.
- One model table for `/v1/models`, `/api/tags` and routing, so advertised ==
  routable; the Ollama digest is stable across restarts (#20).
- `body_size_cap_bytes` (8 MiB) and `prompt_chars_cap` (1M) return 413 in each
  dialect's envelope (#22).
- `security.data_plane_auth: true` extends `DIO_API_KEY` to `/v1/*` and `/api/*`
  (health probes stay open). Off by default: DIO is a control plane.

Also: routing scored every backend twice per request and re-hashed the prompt per
backend — it is now one pure pass with the prefix key computed once.

## Honest demo number

Offline three-way comparison (`examples/agent_swarm_demo.py --agents 8`), eight
concurrent agents, one engine throttled 8x mid-run, mock engines:

| router | session switches | mean (post-throttle) |
|---|---|---|
| round-robin | 10.9 | 1994 ms |
| sticky | 0.0 | 1453 ms |
| **DIO** | **1.0** | **1328 ms** |

Behaviour demo on mock engines, not a hardware benchmark. The README's
"Hardening round" section lists what is still open, honestly.
