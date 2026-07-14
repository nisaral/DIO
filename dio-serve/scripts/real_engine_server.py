#!/usr/bin/env python3
"""
Minimal OpenAI-compatible inference server for REAL local tests.

Uses HuggingFace transformers (not mock latency). Tries CUDA, falls back to CPU.

  python scripts/real_engine_server.py --port 8000 --model Qwen/Qwen2.5-0.5B-Instruct
  python scripts/real_engine_server.py --port 8001 --model Qwen/Qwen2.5-0.5B-Instruct --latency-mult 2.0
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"Ready: {model_id} on {device}", flush=True)
    return tok, model


def messages_to_text(messages: List[Dict[str, Any]], tok) -> str:
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    parts = []
    for m in messages:
        parts.append(f"{m.get('role','user')}: {m.get('content','')}")
    parts.append("assistant:")
    return "\n".join(parts)


def build_app(model_id: str, latency_mult: float = 1.0) -> FastAPI:
    device = pick_device()
    tok, model = load_model(model_id, device)
    app = FastAPI(title=f"DIO Real Engine ({model_id})")
    app.state.model_id = model_id
    app.state.device = device
    app.state.latency_mult = latency_mult

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": model_id,
            "device": device,
            "cuda": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    def chat(body: Dict[str, Any] = Body(...)):
        t0 = time.perf_counter()
        messages = body.get("messages") or [{"role": "user", "content": "hi"}]
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 32)
        max_new = max(1, min(max_new, 128))
        prompt = messages_to_text(messages, tok)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        text = tok.decode(gen, skip_special_tokens=True)
        e2e = (time.perf_counter() - t0) * 1000.0
        if latency_mult > 1.0:
            # optional artificial slowdown for hetero demo (still real tokens)
            extra = e2e * (latency_mult - 1.0)
            time.sleep(extra / 1000.0)
            e2e *= latency_mult
        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_tokens = int(gen.shape[0]) if gen.numel() else 1
        return JSONResponse(
            {
                "id": f"chatcmpl-real-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "dio_engine_e2e_ms": round(e2e, 2),
            }
        )

    @app.post("/v1/completions")
    def completions(body: Dict[str, Any] = Body(...)):
        prompt = str(body.get("prompt") or "Hello")
        return chat(
            {
                "model": body.get("model"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": body.get("max_tokens", 32),
            }
        )

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--latency-mult", type=float, default=1.0)
    args = ap.parse_args()
    app = build_app(args.model, args.latency_mult)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
