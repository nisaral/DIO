<p align="center">
  <img src="docs/assets/logo.jpg" alt="DIO logo" width="150"/>
</p>

<h1 align="center">DIO Serve (v0.4.0)</h1>

<p align="center">
  <strong>Predictive NLMS Orchestrator & Universal LLM Gateway</strong><br/>
  Wraps vLLM, Ollama, SGLang & TGI · Dual OpenAI + Ollama API · Zero Engine Patches
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.4.0-blue.svg" alt="Version 0.4.0" />
  <img src="https://img.shields.io/badge/python-3.9+-brightgreen.svg" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/tests-35%20passing-success.svg" alt="Tests" />
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License" />
</p>

<p align="center">
  <a href="#quick-start-in-60-seconds">Quick Start</a> ·
  <a href="#config-as-code-dioyaml">Config-as-Code</a> ·
  <a href="#universal-api-openai--ollama">Ollama & OpenAI Support</a> ·
  <a href="#multi-model-routing">Multi-Model Routing</a> ·
  <a href="#mcp-server-for-ai-ides">MCP Server</a> ·
  <a href="#cli-reference">CLI Reference</a> ·
  <a href="docs/ARCHITECTURE.md">Architecture</a>
</p>

---

## What is DIO?

**DIO (Distributed Inference Orchestrator)** is a production-grade, non-invasive control plane that load-balances requests across heterogeneous LLM instances (vLLM, Ollama, SGLang, TGI).

Instead of naive Round-Robin (Nginx/Envoy) that ignores GPU divergence, thermal throttling, and KV-cache pressure, DIO:
1. **Learns** each backend's latency in real time using an online **Dual-Timescale Normalized Least Mean Squares (NLMS)** adaptive filter ($O(1)$ updates, zero background training).
2. **Fuses** non-invasive telemetry (vLLM `/metrics` for KV-cache utilization, queue depth, and prefix-cache hits) into a joint cost function.
3. **Routes** requests intelligently based on target model, session prefix affinity, and hardware tier.
4. **Admits or rejects** under burst overload using **empirical percentile gating**, guaranteeing strict latency SLOs while maximizing goodput.

```text
       OpenAI SDK / LangChain / Curl / Continue.dev / OpenWebUI
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
          OpenAI API (:8085/v1)       Ollama API (:8085/api)
         ┌──────────────────────────────────────────────────┐
         │                    DIO GATEWAY                   │
         │  • Dual-Timescale NLMS   • Multi-Model Routing   │
         │  • Empirical Admission   • KV-Cache Fusion       │
         └─────────────────────────┬────────────────────────┘
                   ┌───────────────┼───────────────┐
                   ▼               ▼               ▼
              vLLM (GPU 0)    vLLM (GPU 1)    Ollama (CPU/GPU)
               Port :8000      Port :8001       Port :11434
```

---

## Highlights in v0.4.0

- 🤖 **MCP Server for AI-IDEs (`dio mcp`)**: Seamlessly connects DIO to Claude Desktop, Cursor, VS Code, and Windsurf over JSON-RPC 2.0 stdio. AI assistants can inspect running models, query cluster health, preview latency and queue delays via learned NLMS filters, and route inferences.
- ⚙️ **Config-as-Code (`dio.yaml`)**: Define multi-backend pools, model routing rules, and scheduler knobs in a single YAML file.
- 🔍 **Local Engine Auto-Discovery (`dio init -d`)**: Automatically scans `localhost` for active Ollama (11434), vLLM (8000/8001), and SGLang (30000) instances, detects loaded models, and generates your customized `dio.yaml`.
- 🦙 **Universal Drop-In API (OpenAI + Ollama)**: Exposes both OpenAI endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) and Ollama native endpoints (`/api/chat`, `/api/generate`, `/api/tags`, `/api/version`, `/api/show`). Point your favorite client to DIO without changing code!
- 🔀 **Multi-Model Routing**: Directs requests to specific backends based on the requested model name (exact or partial matching) with safety rejection for unserved models.
- ⚡ **Full Real-Time Streaming**: Real-time Server-Sent Events (SSE) for OpenAI clients and Newline-Delimited JSON (NDJSON) for Ollama clients with non-buffering chunk passthrough and accurate end-to-end latency accounting.
- 🧪 **Offline Mock Streaming**: Built-in `MockBackendServer` with streaming SSE support for unit tests, offline development, and CI environments without GPU dependencies.

---

## Quick Start in 60 Seconds

### 1. Installation

```bash
git clone https://github.com/nisaral/DIO.git
cd DIO/dio-serve
pip install -e .
```

### 2. Auto-Discover & Initialize

Run `dio init` to scan your running local engines and generate `dio.yaml`:

```bash
dio init
```
*Output:*
```text
Scanning localhost for running inference engines...
Discovered 2 active inference engine(s):
  * ollama-local (ollama) at http://127.0.0.1:11434 [models: llama3:latest, mistral:latest]
  * vllm-primary (vllm) at http://127.0.0.1:8000 [models: meta-llama/Llama-3.2-3B-Instruct]
[OK] Created dio.yaml
```

Inspect resolved routing anytime with:
```bash
dio config
```

### 3. Start DIO Gateway

```bash
dio serve
```
DIO starts listening on `http://0.0.0.0:8085`!

### 4. Zero-GPU Demo Mode

Test DIO instantly on any laptop without GPU or external servers:

```bash
dio demo --duration 15
```

---

## Universal API: OpenAI & Ollama

### Calling via OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8085/v1", api_key="unused")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Calling via Ollama CLI or SDK

Simply point `OLLAMA_HOST` to DIO:

```bash
export OLLAMA_HOST=http://127.0.0.1:8085

# Ollama CLI commands work directly through DIO
ollama list
ollama run llama3 "Explain quantum computing in one sentence."
```

Or make a native HTTP call:

```bash
curl http://127.0.0.1:8085/api/chat -d '{
  "model": "llama3",
  "messages": [{"role": "user", "content": "Hello DIO!"}],
  "stream": false
}'
```

---

## Config-as-Code (`dio.yaml`)

Define your infrastructure declaratively:

```yaml
# dio.yaml - Distributed Inference Orchestrator
backends:
  # Primary GPU (vLLM instance)
  - id: gpu-primary
    url: http://localhost:8000
    tier: large
    engine: vllm
    models:
      - meta-llama/Llama-3.2-3B-Instruct

  # Local CPU / secondary GPU (Ollama instance)
  - id: ollama-local
    url: http://localhost:11434
    tier: small
    engine: ollama
    models:
      - llama3
      - codellama
      - mistral

scheduler:
  strategy: nlms        # Options: nlms | rls | ewma | static | round_robin | least_loaded
  nlms_mode: dual       # dual: fast µ (0.1) for bursts + slow µ (0.01) for thermal drift
  engine_metrics: true  # Non-invasive scraping of vLLM /metrics for KV-cache pressure
  cache_bonus_ms: 200   # Affinity bonus for session/prefix reuse in multi-turn chat

admission:
  mode: empirical       # Options: empirical (percentile gate) | rank_only | absolute
  slo_ms: 30000         # Maximum acceptable latency threshold (ms)
  percentile: 95        # Percentile gate for empirical mode

server:
  host: 0.0.0.0
  port: 8085
  timeout: 300
```

---

## Multi-Model Routing

When requests arrive at DIO, the target model parameter is evaluated:
1. **Exact Match**: Request for `meta-llama/Llama-3.2-3B-Instruct` routes strictly to backends serving that exact ID.
2. **Partial / Case-Insensitive Match**: Request for `llama3` or `mistral` matches substring names on compatible backends.
3. **Safety Protection**: If model constraints are configured and an unserved model (e.g. `gpt-4`) is requested, DIO returns a controlled `503 Service Unavailable` rather than misrouting to an arbitrary engine.
4. **Synthetic Fallback**: If no model constraints are defined, traffic is load-balanced across all healthy backends according to NLMS cost scores.

---

## MCP Server for AI-IDEs

Package DIO as a **Model Context Protocol (MCP)** server so AI assistants (Cursor, Claude Desktop, VS Code Continue/Cline, Windsurf) can interact with your cluster:

### Available Tools

| Tool | Purpose |
|------|---------|
| `dio_get_models` | Query DIO for available models, active backend bindings, and health |
| `dio_predict_latency` | Get latency and cost predictions before sending requests using learned NLMS slopes |
| `dio_route_prompt` | Route prompts through DIO's smart scheduler to the optimal backend |
| `dio_cluster_status` | Query live cluster telemetry, learned slopes, intercepts, and KV pressure |

### IDE Integration Setup

#### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "dio": {
      "command": "dio",
      "args": ["mcp", "--gateway-url", "http://127.0.0.1:8085"]
    }
  }
}
```

#### Cursor (`.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "dio": {
      "command": "dio",
      "args": ["mcp", "--gateway-url", "http://127.0.0.1:8085"]
    }
  }
}
```

---

## CLI Reference

| Command | Usage | Description |
|---------|-------|-------------|
| `dio serve` | `dio serve [-c dio.yaml] [-p 8085]` | Start DIO gateway using config file or CLI backend flags |
| `dio init` | `dio init [-d / -n] [-o dio.yaml]` | Generate `dio.yaml` with auto-discovery of running engines |
| `dio config` | `dio config [-c dio.yaml]` | Display parsed backends table and model-to-backend routing map |
| `dio mcp` | `dio mcp [-g http://127.0.0.1:8085]` | Run DIO as an MCP server over stdio for AI-IDE integration |
| `dio demo` | `dio demo [-t 20] [-p 8085]` | Zero-GPU live demo with mock backends and traffic generation |
| `dio bench-smoke` | `dio bench-smoke [-n 40] [-c 4]` | Compare NLMS vs Round-Robin on synthetic heterogeneous workers |
| `dio version` | `dio version` | Show current package version (`0.4.0`) |

---

## Observability & Debug Endpoints

DIO provides rich observability without requiring external agents:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Service liveness, active backend count, and strategy |
| `/v1/models` | `GET` | Aggregated OpenAI-compatible model list |
| `/api/tags` | `GET` | Aggregated Ollama-compatible model list |
| `/debug/metrics` | `GET` | Live NLMS learned slopes, intercepts, MAPE, and decision counters |
| `/debug/engine` | `GET` | Scraped vLLM Prometheus metrics snapshots (KV cache, waiting queue) |
| `/debug/admission` | `GET` | Admission rejection stats, goodput metrics, and SLO tracking |
| `/debug/workers` | `GET` | Worker health status, tier classifications, and error counts |

Inspect metrics in real time:
```bash
curl -s http://localhost:8085/debug/metrics | jq '.prediction.mape_pct, .admission'
```

---

## Python Library Usage

You can also embed DIO directly in your Python applications:

```python
from dio import Backend, DIOGateway

gw = DIOGateway(
    backends=[
        Backend(id="gpu0", base_url="http://127.0.0.1:8000", tier="large"),
        Backend(id="ollama", base_url="http://127.0.0.1:11434", tier="small"),
    ],
    strategy="nlms",
    nlms_mode="dual",
    slo_ms=30000,
    port=8085,
)

gw.run()
```

---

## Verification & Testing

DIO is thoroughly tested with comprehensive unit and integration suites:

```bash
pytest tests/ -v
```
*All 35 tests passing across config parsing, multi-model routing, SSE streaming, Ollama adapter, and MCP server.*

---

## Citation

If you use DIO in your research or production systems, please cite:

```bibtex
@software{dio2026,
  title  = {DIO: Calibration-Robust Routing and Admission for Multi-Instance LLM Serving},
  author = {Nisar, Keyush and Parikh, Krishil and Maisheri, Krisha and Gawade, Aruna and Rathod, Nilesh T. and Florence, Angelin A.},
  year   = {2026},
  url    = {https://github.com/nisaral/DIO},
  doi    = {10.5281/zenodo.22085398}
}
```

## License

Apache-2.0
