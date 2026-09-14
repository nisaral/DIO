# Contributing to DIO

Thanks for taking the time to contribute. DIO (Distributed Inference Orchestrator) is an
Apache-2.0, predictive NLMS orchestrator and universal LLM gateway. It wraps engines you
already run (vLLM, Ollama, SGLang, TGI, anything OpenAI-compatible) and routes traffic
across them without patching the engines themselves.

This document is the practical guide to getting a change merged. By participating you agree
to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 1. Know which half of the repo you are in

This repository contains two implementations of the same scheduling ideas. They are not
equals, and one of them is frozen.

| Path | What it is | New feature work? |
|------|------------|-------------------|
| `dio-serve/` | **The shipping product.** Python package `dio-serve` (v0.4.0) providing the `dio` CLI, the gateway, the scheduler, the MCP server, and the test suite. | **Yes - all new features go here.** |
| `DIO/` | Legacy Go control plane (gRPC workers + gateway). Used for the original low-overhead control-plane research. | **No.** Frozen. Bug fixes only, and only if they cannot be done in `dio-serve/`. |

If you are adding an engine adapter, a route, a config option, a metric, or a CLI flag, it
belongs in `dio-serve/`. See `dio-serve/docs/ARCHITECTURE.md` section 7 for the full
comparison.

---

## 2. Development setup

Requires Python 3.9+ (`requires-python` in `pyproject.toml`); the maintainers develop and
test on 3.11. No GPU is needed to develop or to run the full test suite.

```bash
git clone https://github.com/nisaral/DIO.git
cd DIO/dio-serve
```

Create the virtual environment and install in editable mode:

```bash
python3.11 -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Then install the package with the dev extra:

```bash
pip install -e ".[dev]"
```

The `dev` extra brings `pytest`, `pytest-asyncio` (the suite uses `@pytest.mark.asyncio`),
`ruff`, `build` and `twine`; `httpx` is a runtime dependency, so it comes in automatically.

Optional extras:

```bash
pip install -e ".[bench]"   # matplotlib + numpy, for the scripts under dio-serve/scripts/
pip install -e ".[nvml]"    # pynvml, for real VRAM telemetry instead of reported values
```

---

## 3. Run the tests

From inside `dio-serve/`:

```bash
python -m pytest tests/ -q
```

The suite is 89 tests and finishes in a couple of minutes on a laptop, entirely offline - no
GPU and no network access required. It covers config-file parsing, validation and discovery,
multi-model routing, SSE + ndjson streaming (including mid-stream failure and
admission-rejection regressions), the Ollama native adapter, engine-metrics scraping, admin
authentication, request-shape validation, the scheduler, and the MCP server.

While iterating, run only the file you touched:

```bash
python -m pytest tests/test_streaming.py -q
```

The suite must stay green, and it must keep passing with no network access. If you add a feature, add a test next to the existing ones in
`dio-serve/tests/`. Do not add a GPU or network dependency to the default suite - use the
existing `MockBackendServer` pattern instead.

---

## 4. Try the change on a real gateway (zero GPU)

Before and after your change, exercise the product end to end:

```bash
dio demo --duration 15
```

This starts a real gateway on `127.0.0.1` with mock backends and generates traffic against
it, so you can watch routing decisions, NLMS prediction updates, and admission behaviour
without any hardware. `dio bench-smoke` does the same thing as a head-to-head NLMS vs
round-robin comparison on synthetic heterogeneous workers.

When your change touches the HTTP surface, add a real config file and use the normal path:

```bash
dio init                 # auto-discovers running local engines, writes dio.yaml
dio config               # prints the parsed backends and the model->backend routing map
dio serve -c dio.yaml    # start the gateway
```

---

## 5. Code style

We use [ruff](https://docs.astral.sh/ruff/) for both linting and formatting. From
`dio-serve/`:

```bash
ruff check .
ruff format .
```

There is currently no `[tool.ruff]` section in `pyproject.toml`, so ruff's defaults apply
(line length 88). A few existing files exceed that; do not reformat files you are not
changing - keep your diff scoped to your own work.

Beyond the linter:

- Keep the type annotations that are already there. Public functions and dataclasses in
  `src/dio/` are annotated; match that.
- Prefer dataclasses and plain functions over new abstraction layers.
- Keep new dependencies out of the core `dependencies` list unless there is no alternative.
  Optional functionality belongs in an extra.
- Do not add inline comments that restate the code. Comments should explain *why*.
- Keep source files ASCII. The repo has some pre-existing encoding damage; do not add more.

---

## 6. Adding a new engine adapter

Most engines are already OpenAI-compatible, so try the cheap path first: point a `Backend`
at the server and, if the paths differ, use `api_style="custom"` with `chat_path` /
`completions_path`. No code change needed.

If the wire format itself differs, extend `Backend.api_style`:

1. Add the literal to the `ApiStyle` type in `dio-serve/src/dio/backends.py`.
2. Map it to a URL in `Backend.chat_url()` and `Backend.completions_url()`.
3. If the request/response shape differs, add a translation function next to the existing
   `openai_chat_to_tgi_generate` / `tgi_generate_to_openai_chat` pair in `backends.py`, and
   call it from the forward path in `dio-serve/src/dio/gateway.py`.
4. Add a test in `dio-serve/tests/` following `test_ollama_adapter.py`, using the mock
   backend server. Translation functions should be tested directly as pure functions.
5. Document the new style in `dio-serve/docs/PRODUCTION.md`, in the engine support table
   and the "What about Anthropic / Gemini / Azure OpenAI cloud APIs?" section.

`api_style="tgi_generate"` is the reference implementation for a non-OpenAI path. Note that
`config_file.py` auto-selects an `api_style` during `dio init` discovery - update it too if
your engine should be detected automatically.

---

## 7. Commit and pull request conventions

We use [Conventional Commits](https://www.conventionalcommits.org/). Real examples from
this repository:

```text
feat(dio-serve): v0.4.0 release - config-as-code, multi-model routing, Ollama native adapter, and SSE streaming
feat(mcp): add Model Context Protocol server for AI-IDE integration
fix: skip T7 by default, disable admission in benchmarks, never abort on locust exit
fix: register worker before load; harden preflight checks
feat: calibrated heterogeneity emulation (intercept/slope, jitter, thermal)
```

Format: `type(scope): summary`, imperative mood, no trailing period.

- Types in use: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `perf`, `ci`, `build`.
- Scopes in use: `dio-serve` for the Python product, `mcp` for the MCP server, or the
  module you touched (`scheduler`, `gateway`, `config`, ...). Scope is optional.
- If your change is breaking, say so in the commit body and the PR description.

For pull requests:

- Use the PR template and fill in the checklist honestly.
- Keep one logical change per PR. Unrelated cleanups belong in their own PR.
- Say what you tested and how, including the exact commands.
- Update documentation in the same PR: `README.md`, `docs/PRODUCTION.md`,
  `docs/API.md`, or `CHANGELOG.md` as appropriate.
- New behaviour needs a test. Bug fixes need a test that fails before the fix.
- Sign off your commits (`git commit -s`) to certify the
  [Developer Certificate of Origin](https://developercertificate.org/). We do not require a
  CLA.

---

## 8. Security ground rules for contributions

Read [SECURITY.md](SECURITY.md) for the full picture. The essentials for reviewers:

- **DIO ships with no authentication and binds `0.0.0.0` by default.** It is designed to sit
  behind an authenticating reverse proxy on a private network, not on the public internet.
- **Every `/debug/*` route is unauthenticated**, including the state-mutating ones:
  `POST /debug/reset_stats`, `POST /debug/chaos/vram`, and `POST /debug/backends`. These
  wipe statistics, forge VRAM values, and hot-register arbitrary backends.
- Therefore: **never add a new `/debug/*` route, or extend an existing one, without
  treating it as an admin-only endpoint.** Do not document, demo, or screenshot a
  publicly reachable `/debug` path. Do not add a feature whose happy path requires exposing
  `/debug/*` to untrusted clients.
- Never commit credentials, GPU host names, API keys, or private endpoints. Review config
  excerpts and logs pasted into issues for the same.
- If you believe you have found a vulnerability, do not open a public issue. Follow
  [SECURITY.md](SECURITY.md).

---

## 9. Good first issues

These are scoped, real, and known to be needed. Comment on the GitHub issue to claim one.

1. **Add an Anthropic or Gemini `api_style` adapter** using the recipe in section 6. Good
   way to learn the adapter path end to end.
2. **Add an OpenTelemetry or Prometheus `/metrics` endpoint** alongside `/debug/metrics`, so
   operators do not have to scrape a debug route for production telemetry.
3. **Publish to PyPI** with a trusted-publishing release workflow. The version already has a
   single source (`dio-serve/src/dio/_version.py`) and CI asserts the distribution metadata
   matches it.
4. **Add `mypy` checking** for `src/dio/scheduler.py` and `src/dio/backends.py`, and wire it
   into the CI workflow.
5. **Document a real deployment** end to end: reverse proxy with TLS, two vLLM backends, and
   the config file to match. `DIO_API_KEY` already guards `/debug/*`, so the proxy only has to
   terminate TLS.
6. **Model-name consistency.** `/api/tags` falls back to the backend *id* as a model name when
   a backend declares no `model`, while `/v1/models` reports what the engine advertises, and a
   gateway with no `model_map` forwards any model name it is given. Make the three agree, and
   make routing accept whatever is listed.
7. **Request-size guards.** There is no cap on body size or prompt length (`max_tokens_cap`
   guards the completion budget only). Add a configurable `max_body_bytes` / prompt bound,
   reject with the OpenAI envelope, and test it.
8. **A `strict` admission mode.** The empirical gate deliberately does not reject while one
   backend is still clearly the best option (see the counter semantics in
   `dio-serve/README.md`), so a saturated pool keeps answering 200s with
   `X-DIO-Over-Budget: 1`. Add an opt-in mode that sheds load on the observed tail alone.
9. **SSE keep-alive frames.** A long queue wait produces no bytes until the first token
   (observed TTFT climbing to ~2 s under load), which trips short client read timeouts.
   Emit `: keep-alive` comments while waiting for the upstream. Requires care: the OpenAI SSE
   passthrough is currently byte-exact.
10. **Affinity hit-rate under churn.** With many distinct prefixes the 2048-entry LRU evicts
   sessions mid-conversation and `affinity.hit_rate` collapses. Add eviction counters to
   `/debug/metrics` and consider a size-aware policy.

---

## 10. Getting help

- Open a [discussion](https://github.com/nisaral/DIO/discussions) for design questions and
  "how do I" questions.
- Open an issue for a confirmed bug or a concrete feature request, using the templates.
- Keep discussions technical and kind. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
