#!/usr/bin/env python3
"""
Production-style DIO wrap around real vLLM (or any OpenAI-compatible) servers.

Prereq: engines already running, e.g.

  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \\
      --model meta-llama/Llama-3.2-3B-Instruct --port 8000
  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \\
      --model meta-llama/Llama-3.2-3B-Instruct --port 8001

Then:

  python examples/production_vllm.py
  # clients → http://127.0.0.1:8085/v1
"""

from dio import Backend, DIOGateway

if __name__ == "__main__":
    gw = DIOGateway(
        backends=[
            # Real engines — change host/ports to your fleet
            Backend(
                id="gpu0",
                base_url="http://127.0.0.1:8000",
                tier="small",
                api_style="openai",  # vLLM / SGLang / Ollama
            ),
            Backend(
                id="gpu1",
                base_url="http://127.0.0.1:8001",
                tier="small",
                api_style="openai",
            ),
            # Example TGI native (uncomment if you use TGI /generate):
            # Backend(
            #     id="tgi0",
            #     base_url="http://127.0.0.1:8080",
            #     api_style="tgi_generate",
            # ),
        ],
        strategy="nlms",
        nlms_mode="dual",
        slo_ms=60_000,
        admission_off=False,  # production: shed overload
        host="0.0.0.0",
        port=8085,
    )
    print("DIO production gateway on :8085 — backends:", [b.id for b in gw.pool.list()])
    gw.run()
