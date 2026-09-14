# Changelog

All notable changes to DIO and `dio-serve` will be documented in this file.

## [Unreleased] - production hardening

Every change below came out of auditing the gateway and then dogfooding it; each
fix ships with a regression test. Test suite: 35 -> 101.

### Fixed - round 2 (a live gateway driven by a swarm of agents)
- **Streamed token counts disagreed with JSON ones.** `/api/chat` and
  `/api/generate` reported estimates on the streaming path (prompt chars/4, one per
  delta chunk) and the engine's `usage` on the JSON path, so the same prompt gave
  two different answers depending on `stream`. The gateway now asks OpenAI-style
  engines for a `stream_options.include_usage` chunk, and retries once without the
  field when an engine 400s on it. Engines that never send usage still fall back to
  the estimate (covered by a test).
- **One absurd `max_tokens` poisoned the model.** The routing feature is
  prompt-tokens + completion budget, so `max_tokens: 10**9` produced a ~10^9 ms
  prediction and left a permanent crater in `mae`/`mape` (one sample, never
  decays). The feature is now clamped by `token_feature_cap` (default 32768, 0
  disables) -- the forwarded request is still untouched, the engine's
  `max_model_len` stays authoritative.
- **Streamed tool calls were dropped.** The Ollama streaming translation only read
  `delta.content`, so a function-calling agent got a 200 with an empty answer. Tool
  call fragments (name, then piecewise JSON arguments) are accumulated and returned
  in `message.tool_calls`, including in the final frame.
- **Streamed `done_reason` was hardcoded `stop`** even when the engine stopped for
  `length`. It now follows the engine's `finish_reason` (same mapping as the JSON
  path).
- **An unknown `model` was served as a 200** by whichever backend was cheapest.
  With a `model_map` configured, an unserved name is now a 404 `model_not_found`
  (Ollama shape on `/api/*`) instead of an empty allow-list that surfaced as a
  confusing 503 "no healthy backends".
- **Unparseable JSON returned FastAPI's 422** with a `{"detail": ...}` body; OpenAI
  clients expect 400 with the `{"error": {...}}` envelope. Same for the message:
  `max_tokens: 1e9` is a *float*, and "must be a positive integer" sent callers
  after the wrong bug.
- **Nothing signalled overload.** In a 500-wide storm at p95 ~29 s against a 5 s
  budget, no response ever told the client it was over budget (the empirical gate
  suppresses rejects while some backend is still clearly best). Responses now carry
  `X-DIO-Budget-Ms` and `X-DIO-Over-Budget` (plus `X-DIO-Predicted-Ms` on streams,
  which are flushed before latency is known), and the `dio` block reports
  `slo_ms` / `over_budget`.
- **`MockBackendServer` lied about usage**: it reported `max_tokens` as
  `completion_tokens`, always `finish_reason: "stop"`, and slept
  `decode_ms * max_tokens` even for absurd budgets. It now reports what it
  generated, honours `max_tokens` (truncating with `finish_reason: "length"`) and
  bounds the simulated decode work.

### Fixed - correctness
- **Streaming ignored the backend's API key.** Both SSE paths built the upstream
  request without `Backend.auth_headers()`, so every stream 401'd against a
  token-gated engine while the JSON path worked. They now also honour
  `Backend.timeout_s`.
- **Streaming admission rejections crashed** (`NameError: cannot access free
  variable 'e'`) and emitted a non-JSON body: the generator closed over an
  `except ... as e` binding that Python unbinds when the block exits.
- **An engine dying mid-stream** silently truncated the client with `200 OK`.
  DIO now emits an in-band error frame plus `[DONE]` (SSE) or an error object
  (NDJSON) instead of ending the response as if it had finished.
- **Failed requests were counted as successes.** `feedback()` ran for 4xx/5xx and
  for mid-stream aborts, so `goodput_fraction` read `1.00` on a run where half the
  requests failed. Failures no longer train the learner and are counted in
  `attempts` / `failed_total`; goodput is computed over attempts.
- **NLMS train/serve skew.** The learner was fed the backend's
  `usage.total_tokens`, while `estimate()` had predicted on the router's own token
  count. The gradient is `err/tokens`, so a mismatch evaluates the update at the
  wrong point and drifts the slope. Backend usage is now surfaced separately as
  `reported_tokens` and is informational only.
- **`batch_size=0`** (env-overridable via `DIO_BATCH_SIZE`) raised
  `ZeroDivisionError` in the NLMS hot path.
- **TGI translation** treated `temperature=0` / `max_tokens=0` as "unset"
  (silently sampling when the client asked for greedy), dropped `stop`, `top_p`
  and the penalties, and always reported `prompt_tokens: 0`.
- **`_resolve_model_backends`** substring-matched, so `model="3B"` or `"a"` could
  hijack routing; matching is now anchored to a whole path segment.
- **`POST /debug/backends`** accepted an arbitrary `base_url`, which made the
  admin plane an SSRF primitive (`http://169.254.169.254/...`). Backend URLs are
  validated (`http`/`https`, real host, no metadata hosts); bad input is a 400.
- **One upstream 5xx evicted an otherwise healthy engine** from the whole pool for
  up to `health_interval_s` (5 s), during which every request 503'd. Eviction now
  requires 3 consecutive failures, and `Retry-After` reports the real recovery
  horizon instead of a hardcoded 1 s.
- **`/health` reported `ok` while every request was being rejected.** It now
  returns `status: degraded` with `backends_unhealthy` / `backends_registered`.
- **Ollama non-streaming translations dropped** DIO's `X-DIO-Backend` /
  `X-DIO-E2E-Ms` routing headers.
- **`logging.basicConfig` at import time** in `dio.mcp` reconfigured the *root*
  logger, so embedding DIO hijacked the host application's logging. Logging is now
  configured explicitly by the CLI (`configure_stdio_logging`).
- **MCP JSON-RPC compliance:** notifications are no longer answered, batch
  requests dispatch correctly, non-object messages get `-32600`, and
  `protocolVersion` is negotiated instead of assumed.
- **`ConfigError`** replaces raw `AttributeError` / `ValueError` tracebacks for a
  malformed `dio.yaml`; `dio serve` exits 2 with one diagnostic line.

### Added
- `DIO_API_KEY` gates every `/debug/*` route (`Authorization: Bearer` or
  `X-DIO-API-Key`), with a startup warning when the gateway is bound to a
  non-loopback address and no key is set.
- Bounded LRU (2048 entries) for the session/prefix affinity cache.
- `dio-serve/Dockerfile`, `docker-compose.yml` (Ollama engine + gateway), and
  `.dockerignore`.
- `examples/agent_swarm_demo.py`: offline round-robin / sticky / DIO comparison
  under a mid-run engine throttle; `examples/swarm_live_stack.py` and
  `examples/swarm_client.py` to drive a live gateway from many concurrent agents.
- GitHub Actions CI: lint, test matrix (3.9-3.12), zero-GPU CLI smoke test, and a
  package build that asserts the sdist contains the license.
- Community files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  issue/PR templates, Dependabot, and the full Apache-2.0 `LICENSE` text.

### Changed
- `dio.__version__` and the package metadata now come from a single source
  (`dio/_version.py`); CI asserts they cannot drift.
- `ruff` is clean across `src/` and `tests/`; `pyproject.toml` declares the
  pytest and ruff configuration the repository actually uses.
- `dio-serve/LICENSE` carries the full Apache-2.0 text (was a 13-line notice).

---

## [v0.4.0] - 2026-09-14

### Highlights
- **MCP Server for AI-IDE Integration (`dio mcp`)**: Packages DIO as a Model Context Protocol (MCP) server over JSON-RPC 2.0 stdio. AI-powered IDEs (Cursor, Claude Desktop, VS Code, Windsurf) can query cluster health, inspect model availability, obtain real-time latency/cost predictions using learned NLMS slopes, and route prompts intelligently.
- **Ollama Native Adapter**: DIO now acts as a drop-in gateway for both OpenAI and Ollama clients (`POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `GET /api/version`, `POST /api/show`). Point tools like OpenWebUI, Continue.dev, or Ollama CLI to DIO and route transparently across vLLM and Ollama engines!
- **Config-as-Code (`dio.yaml`)**: Define multi-backend pools, model routing maps, and dual-timescale NLMS scheduler knobs in a clean YAML file.
- **Auto-Discovery CLI (`dio init -d`)**: Scans localhost for running Ollama (11434), vLLM (8000/8001), and SGLang (30000) instances, discovers loaded models, and auto-generates a ready-to-run `dio.yaml`.
- **Multi-Model Routing**: Intelligently dispatch requests to backends serving the requested model, with exact and partial matching, and safety rejection for unmatched models.
- **SSE & NDJSON Streaming**: Full streaming support for OpenAI Server-Sent Events (SSE) and Ollama newline-delimited JSON (NDJSON) with zero buffering and accurate latency tracking.
- **Windows Terminal Polish**: Clean ASCII formatting for CLI banners, tables, and panels across Windows PowerShell and CMD.

### Added
- `dio.mcp`: Model Context Protocol server exposing `dio_get_models`, `dio_predict_latency`, `dio_route_prompt`, and `dio_cluster_status` tools.
- `dio mcp`: CLI command to launch the MCP server over stdio with customizable `--gateway-url`.
- `dio.config_file`: YAML and JSON config parser with upward directory traversal discovery.
- `detect_local_backends`: Async probing of local ports to discover active engines and models.
- `dio init`: CLI command with `--detect` / `--no-detect` and `--output` options.
- `dio config`: CLI command to display resolved backend tables and model routing rules.
- `_proxy_ollama_stream`: Dynamic on-the-fly translation from upstream OpenAI SSE to client-facing Ollama NDJSON streams.
- `MockBackendServer` streaming: In-process test and demo server now yields realistic SSE chunks with simulated TTFT and decode latency.
- Comprehensive test suite: Added `test_config_file.py`, `test_multi_model_routing.py`, `test_streaming.py`, `test_ollama_adapter.py`, and `test_mcp.py` (35/35 passing).

### Changed
- Refactored `_resolve_model_backends` to safely reject unknown models when explicit model bindings exist.
- Updated `/v1/models` to aggregate configured models from `model_map` alongside live backend probe results.
- `dio serve` banner now displays both OpenAI and Ollama base URLs.
- Dependencies: Added `pyyaml>=6.0` to `dio-serve`.

---

## [v0.3.0] - 2026-08-24

### Added
- Calibration-robust journal revision updates for Cluster Computing.
- EWMA routing strategy for baseline comparisons.
- Kaggle dual-T4 validation runbooks and scripts.
- Multi-instance vLLM Prometheus metrics scraping (`/metrics`).
- Decoupled admission control with empirical percentile gating.
