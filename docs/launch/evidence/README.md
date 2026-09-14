# Dogfooding evidence

Two rounds of agents driving **one live gateway** (`examples/swarm_live_stack.py`,
three mock engines, no GPU). Nothing here is a hardware benchmark; the point is
behaviour, correctness and honesty of the reported numbers.

## Round 1 - four agents, `dio-serve` v0.4.0 (pre-hardening)

Surfaces, one agent each: OpenAI JSON, OpenAI SSE, Ollama ndjson, and
adversarial load/edge cases. ~1000 requests total: helper sessions of 6-20 turns,
plus 80-, 200- and 500-way concurrent bursts.

What came back:

| observation | number |
|---|---|
| requests | 987 admitted, 1 failed (`failure_rate` 0.001) |
| latency under the 500-way burst | mean 13.2 s, p95 28.7 s, **against a 5 s SLO** |
| 503 / `Retry-After` ever returned | **0** |
| `rejected_slo_suppressed` (admitted although the tail was over budget) | 314 |
| `goodput_fraction` | 0.49 |
| session affinity `hit_rate` under churn | 0.34 |
| prediction `mae_ms` after one `max_tokens: 10**9` request | **12,446,080 ms** |
| prediction `mape_pct` | **5,676,503 %** |
| `X-DIO-Backend` vs the engine's own echo | 18/18, 22/22, 380/380 |
| SSE framing audit | compliant (11 frames, one `finish_reason`, `[DONE]` last) |

The raw snapshot of that run is [live-swarm-metrics-prefix.json](live-swarm-metrics-prefix.json).
The poisoned `mae_ms`/`mape_pct` are not a measurement artefact: one absurd
completion budget entered the routing feature unclamped and never decayed out of
the window. Affinity at 0.34 is the documented consequence of many distinct
prompts churning the 2048-entry prefix LRU, not a routing failure.

Bugs found and fixed (each with a regression test, see `CHANGELOG.md`):

- streaming ignored the engine's API key -> every stream 401'd on token-gated engines
- failures counted as successes -> `goodput_fraction` read 1.00 on a half-failed run
- `POST /debug/backends` accepted any `base_url` (SSRF primitive)
- `max_tokens: 10**9` forwarded unbounded; empty `messages` / negative budgets accepted
- Ollama streaming reported estimates while the JSON path reported the engine's usage
- absurd `max_tokens` poisoned the learned model (above)
- streamed tool calls were dropped; `done_reason` was hardcoded `stop`
- unknown `model` returned 200 and was served anyway
- unparseable JSON returned FastAPI's 422 instead of OpenAI's 400 envelope
- nothing signalled overload: no 503, no header, at p95 29 s

## Round 2 - verification agents, post-fix build

Same stack, restarted on the fixed build; two agents re-tested the specific
claims (token parity, error envelopes, budget headers, no metric poisoning,
header trust under a 60-way burst, disconnect recovery). Snapshot:
[live-swarm-metrics-postfix.json](live-swarm-metrics-postfix.json).

## Reproducing

```powershell
cd dio-serve
.\.venv\Scripts\python.exe examples\swarm_live_stack.py --port 18088 --base-port 18100 --duration 600
# other terminal
.\.venv\Scripts\python.exe examples\swarm_client.py --url http://127.0.0.1:18088 --agent-id 1 --persona coding --turns 12 --api openai --stream
curl.exe -s http://127.0.0.1:18088/debug/metrics
```

The offline, three-way comparison of the same workload (round-robin vs sticky
round-robin vs DIO) is `examples/agent_swarm_demo.py`; the numbers are in the
root `README.md`.
