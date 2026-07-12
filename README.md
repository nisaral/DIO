<p align="center">
  <img src="dio-serve/docs/assets/logo.jpg" alt="DIO" width="140"/>
</p>

<h1 align="center">DIO — Distributed Inference Orchestrator</h1>

<p align="center">
  <strong>Wrap vLLM. Learn latency online. Route smart. Admit under SLO.</strong><br/>
  Research system + industry-ready <code>pip</code> gateway
</p>

<p align="center">
  <a href="dio-serve/README.md"><strong>dio-serve package</strong></a> ·
  <a href="dio-serve/docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="dio-serve/docs/API.md">API docs</a> ·
  <a href="https://github.com/nisaral/DIO">GitHub</a>
</p>

---

## Start here (recommended)

```bash
git clone https://github.com/nisaral/DIO.git
cd DIO/dio-serve
pip install -e .
dio demo                 # no GPU
dio serve -b http://127.0.0.1:8000 -b http://127.0.0.1:8001
```

Point any OpenAI client at `http://localhost:8085/v1`.

Full product docs → **[dio-serve/README.md](dio-serve/README.md)**  
How it scales as a wrap-around → **[dio-serve/docs/ARCHITECTURE.md](dio-serve/docs/ARCHITECTURE.md)**  
Every method after pip install → **[dio-serve/docs/API.md](dio-serve/docs/API.md)**

---

## Architecture (wrap-around)

```text
 Clients (OpenAI SDK / curl / LangChain)
                 │
                 ▼
        ┌────────────────┐
        │  DIO Gateway   │  dual-timescale NLMS
        │  :8085 /v1/*   │  joint cost · admission
        └───────┬────────┘
           HTTP OpenAI API (unmodified engines)
        ┌───────┴────────┬────────────┐
        ▼                ▼            ▼
   vLLM GPU0        vLLM GPU1     SGLang / TGI / Ollama
```

DIO **does not** own kernels, KV caches, or continuous batching.  
Those stay in vLLM (etc.). DIO owns **placement + learning + admission**.

That is what makes it **scalable and portable**: add GPUs by adding URLs.

---

## What is novel / better

| Feature | Benefit |
|---------|---------|
| Dual-timescale NLMS | Tracks burst jitter *and* slow thermal drift, \(O(1)\) |
| Joint cost | Latency + queue + tier + VRAM + prefix cache in one score |
| SLO admission | `reject if min S_w > SLO` → better goodput under overload |
| Zero engine patches | Upgrade vLLM freely; mix engines |
| `pip` + OpenAI API | Works on Kaggle, Lightning, RunPod, bare metal |

---

## Repo map

```text
DIO/                          # GitHub root
├── dio-serve/                # ★ pip package (default path)
│   ├── docs/ARCHITECTURE.md
│   ├── docs/API.md
│   ├── docs/assets/logo.jpg
│   └── src/dio/
├── DIO/                      # Go control plane + Locust paper suite (optional)
│   └── benchmarks/camera_ready_suite.py
├── paper_drafts_latex/       # Academic paper
└── figs/                     # Paper figures
```

| Path | Use when |
|------|----------|
| **dio-serve** | Experiments on any cloud, demos, product integration |
| **DIO/** (Go) | Systems microbenchmarks, gRPC workers, dashboard |

---

## Quick commands

```bash
# Package
cd dio-serve && pip install -e .
dio demo
dio bench-smoke
dio serve -b http://127.0.0.1:8000 -b http://127.0.0.1:8001

# Optional full paper suite (Go path)
cd DIO
python benchmarks/camera_ready_suite.py --mode mock --quick
```

---

## Citation

```bibtex
@software{dio2026,
  title  = {DIO: Predictive Orchestration for Heterogeneous LLM Inference},
  author = {Nisar, Keyush and Parikh, Krishil and Maisheri, Krisha},
  year   = {2026},
  url    = {https://github.com/nisaral/DIO}
}
```

## License

Apache-2.0 (see `dio-serve/LICENSE`)
