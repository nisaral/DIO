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

Same stack, restarted on the fixed build; two fresh agents re-tested the claims
independently rather than trusting the changelog. Snapshot:
[live-swarm-metrics-postfix.json](live-swarm-metrics-postfix.json).

| claim | verdict | evidence |
|---|---|---|
| Ollama `prompt_eval_count` matches between `stream: true` and `false` | confirmed | 4/4, 168/168, 1552/1552, 25/25 across four prompt sizes; the estimator would have said 6 and 105 |
| Ollama `eval_count` matches | confirmed after a follow-up | the mock's JSON path still reported `max_tokens`; fixed, now 3 == 3 |
| malformed JSON is a 400 with the OpenAI envelope | confirmed | `{"error":{"message":"15: JSON decode error","type":"invalid_request_error","code":"invalid_body"}}` |
| `max_tokens: 1e9` (float) says *integer*, not *positive* | confirmed | `'max_tokens' must be an integer (got 1000000000.0)` |
| budget headers on JSON and streams | confirmed | `x-dio-budget-ms: 5000`, `x-dio-over-budget: 0/1`, `x-dio-predicted-ms: 109.2` |
| tool calls + `done_reason` follow the engine | confirmed | `pytest tests/test_gateway_contract.py` 7 passed; live `num_predict: 2` -> `done_reason: "length"`, `eval_count: 2` |
| one absurd budget no longer poisons the model | confirmed | `prediction.mae_ms` stayed at 3578 ms (was 12,446,080 ms), `mape_pct` 842 % (was 5,676,503 %), `dio.decision.tokens` = 32768 while the forwarded body kept `max_tokens` |
| 72-way burst stays honest | confirmed | 72/72 HTTP 200, p50 1251 ms / p95 3242 ms, `admitted` +72 exactly, `failed_total 0` |
| `X-DIO-Backend` is trustworthy | confirmed | 10/10 headers matched the engine's own `[gpu-x]` echo and `dio.backend_id` |
| a client abort mid-stream does not hurt the next request | confirmed | `curl --max-time 0.4` cut the stream; next request 200 in 1.21 s, `/health` ok, 3/3 backends healthy |

Honest asterisks from this round, left in the open on purpose:

- the empirical admission gate did not reject anything in the burst because the
  burst never approached the SLO (p99 3.3 s vs 5 s), so `rejected_slo: 0` there is
  a weak signal rather than a passing test (issue #23 asks for a strict mode);
- one backend's fit degraded under churn (`intercept` collapsed to ~0.1 ms while
  its MAE exceeded its own mean latency) even though no statistic was poisoned --
  the learner is still noisy under heavy concurrency;
- `slo` admission remains a *reporting* feature in practice: overload is now
  visible via `X-DIO-Over-Budget`, but the default mode still answers 200s.

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
