# Changelog

All notable changes to DIO and `dio-serve` will be documented in this file.

## [v0.4.0] - 2026-09-14

### Highlights
- **Ollama Native Adapter**: DIO now acts as a drop-in gateway for both OpenAI and Ollama clients (`POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `GET /api/version`, `POST /api/show`). Point tools like OpenWebUI, Continue.dev, or Ollama CLI to DIO and route transparently across vLLM and Ollama engines!
- **Config-as-Code (`dio.yaml`)**: Define multi-backend pools, model routing maps, and dual-timescale NLMS scheduler knobs in a clean YAML file.
- **Auto-Discovery CLI (`dio init -d`)**: Scans localhost for running Ollama (11434), vLLM (8000/8001), and SGLang (30000) instances, discovers loaded models, and auto-generates a ready-to-run `dio.yaml`.
- **Multi-Model Routing**: Intelligently dispatch requests to backends serving the requested model, with exact and partial matching, and safety rejection for unmatched models.
- **SSE & NDJSON Streaming**: Full streaming support for OpenAI Server-Sent Events (SSE) and Ollama newline-delimited JSON (NDJSON) with zero buffering and accurate latency tracking.
- **Windows Terminal Polish**: Clean ASCII formatting for CLI banners, tables, and panels across Windows PowerShell and CMD.

### Added
- `dio.config_file`: YAML and JSON config parser with upward directory traversal discovery.
- `detect_local_backends`: Async probing of local ports to discover active engines and models.
- `dio init`: CLI command with `--detect` / `--no-detect` and `--output` options.
- `dio config`: CLI command to display resolved backend tables and model routing rules.
- `_proxy_ollama_stream`: Dynamic on-the-fly translation from upstream OpenAI SSE to client-facing Ollama NDJSON streams.
- `MockBackendServer` streaming: In-process test and demo server now yields realistic SSE chunks with simulated TTFT and decode latency.
- Comprehensive test suite: Added `test_config_file.py`, `test_multi_model_routing.py`, `test_streaming.py`, and `test_ollama_adapter.py` (28/28 passing).

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
