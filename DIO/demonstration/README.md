# DIO Demonstration System

Interactive demo for presenting DIO's distributed inference orchestration system.

## Quick Start

```bash
# 1. Build the demo server
cd DIO/demonstration
go build -o demo.exe ./cmd/demo/

# 2. Set DIO root (if not auto-detected)
set DIO_ROOT=C:\Users\nisar\OneDrive\Desktop\Go-serve\DIO

# 3. Build the DIO Manager (if not already built)
cd ..
go build -o dio-manager.exe ./cmd/manager/main.go
cd demonstration

# 4. (Optional) Pull Ollama model for chat
ollama pull phi3:mini

# 5. Run the demo
demo.exe

# 6. Open browser
# → http://localhost:9090
```

## Features

| Feature | What It Shows |
|---------|--------------|
| **T1–T11 Test Suite** | Run all validation tests with live streaming results |
| **NLMS Convergence Chart** | Watch the adaptive filter learn in real-time |
| **Interactive Injection** | Type custom prompts, see routing + latency metrics |
| **Burst Mode** | Fire 5–100 requests at once, watch throughput |
| **Ollama Chat** | Ask questions about DIO using a local LLM (RTX 4050) |

## Architecture

```
Browser ←WebSocket→ Demo Go Server (:9090) ←gRPC/HTTP→ DIO Manager (:8085) ←gRPC→ Workers
                                              ↓
                                        Ollama (:11434)
```

## Presentation Script

See [PRESENTATION_SCRIPT.md](./PRESENTATION_SCRIPT.md) for a step-by-step demo flow.
