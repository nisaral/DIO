#!/usr/bin/env python3
"""
Mock vLLM backend exposing the *real* Prometheus gauges DIO scrapes.

This exists so the gateway's hybrid path can be validated without GPUs. It is
NOT a latency model for the paper -- it is a harness self-test fixture. It
reproduces the three signals that were dead in the n=10 dual-T4 campaign:

  vllm:num_requests_running     -- real in-flight count (concurrency witness)
  vllm:num_requests_waiting     -- real queue depth beyond max_concurrency
  vllm:prefix_cache_hits/queries-- real prefix-cache accounting
  vllm:gpu_cache_usage_perc     -- KV occupancy proportional to in-flight work

Usage:
  python scripts/mock_vllm_server.py --port 8000 --latency-ms 120
  python scripts/mock_vllm_server.py --port 8001 --latency-ms 240
"""
from __future__ import annotations

import argparse
import hashlib
import threading
import time
from typing import Any, Dict, List

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import PlainTextResponse


class EngineState:
    """Tracks in-flight/queued work the way a real engine scheduler would."""

    def __init__(self, max_concurrency: int, kv_per_request: float) -> None:
        self.lock = threading.Lock()
        self.max_concurrency = max_concurrency
        self.kv_per_request = kv_per_request
        self.running = 0
        self.waiting = 0
        self.prefix_hits = 0
        self.prefix_queries = 0
        self.seen_prefixes: set[str] = set()
        self.peak_running = 0
        self.peak_waiting = 0

    def admit(self, prefix_key: str) -> bool:
        """Register arrival. Returns True if this prompt prefix was cached."""
        with self.lock:
            self.prefix_queries += 1
            hit = prefix_key in self.seen_prefixes
            if hit:
                self.prefix_hits += 1
            else:
                self.seen_prefixes.add(prefix_key)
            if self.running >= self.max_concurrency:
                self.waiting += 1
                self.peak_waiting = max(self.peak_waiting, self.waiting)
            else:
                self.running += 1
                self.peak_running = max(self.peak_running, self.running)
            return hit

    def start_service(self) -> None:
        """Block until a running slot frees up, then occupy it."""
        while True:
            with self.lock:
                if self.running < self.max_concurrency and self.waiting > 0:
                    self.waiting -= 1
                    self.running += 1
                    self.peak_running = max(self.peak_running, self.running)
                    return
                if self.running <= self.max_concurrency and self.waiting == 0:
                    return
            time.sleep(0.005)

    def release(self) -> None:
        with self.lock:
            self.running = max(0, self.running - 1)

    def snapshot(self) -> Dict[str, float]:
        with self.lock:
            return {
                "running": float(self.running),
                "waiting": float(self.waiting),
                "hits": float(self.prefix_hits),
                "queries": float(self.prefix_queries),
                "kv": min(1.0, self.running * self.kv_per_request),
                "peak_running": float(self.peak_running),
                "peak_waiting": float(self.peak_waiting),
            }


def build_app(model: str, latency_ms: float, state: EngineState) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    def models() -> Dict[str, Any]:
        return {"object": "list", "data": [{"id": model, "object": "model"}]}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        s = state.snapshot()
        return "\n".join(
            [
                "# TYPE vllm:num_requests_running gauge",
                f'vllm:num_requests_running{{model_name="{model}"}} {s["running"]}',
                "# TYPE vllm:num_requests_waiting gauge",
                f'vllm:num_requests_waiting{{model_name="{model}"}} {s["waiting"]}',
                "# TYPE vllm:gpu_cache_usage_perc gauge",
                f'vllm:gpu_cache_usage_perc{{model_name="{model}"}} {s["kv"]:.6f}',
                "# TYPE vllm:prefix_cache_hits counter",
                f'vllm:prefix_cache_hits{{model_name="{model}"}} {s["hits"]}',
                "# TYPE vllm:prefix_cache_queries counter",
                f'vllm:prefix_cache_queries{{model_name="{model}"}} {s["queries"]}',
                "",
            ]
        )

    @app.get("/debug/peaks")
    def peaks() -> Dict[str, float]:
        return state.snapshot()

    @app.post("/debug/reset")
    def reset() -> Dict[str, str]:
        with state.lock:
            state.prefix_hits = 0
            state.prefix_queries = 0
            state.seen_prefixes.clear()
            state.peak_running = 0
            state.peak_waiting = 0
        return {"status": "reset"}

    @app.post("/v1/chat/completions")
    def chat(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        msgs: List[Dict[str, Any]] = body.get("messages") or []
        prompt = "".join(str(m.get("content", "")) for m in msgs)
        key = hashlib.sha1(prompt[:256].encode("utf-8")).hexdigest()

        hit = state.admit(key)
        state.start_service()
        try:
            max_tokens = int(body.get("max_tokens") or 16)
            # Cache hits skip prefill, mirroring real prefix-cache behaviour.
            prefill = 0.0 if hit else latency_ms * 0.5
            decode = latency_ms * 0.01 * max_tokens
            time.sleep((prefill + decode) / 1000.0)
            return {
                "id": "mock",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": max(1, len(prompt) // 4),
                    "completion_tokens": max_tokens,
                    "total_tokens": max(1, len(prompt) // 4) + max_tokens,
                },
            }
        finally:
            state.release()

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--latency-ms", type=float, default=120.0)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--kv-per-request", type=float, default=0.12)
    args = ap.parse_args()

    state = EngineState(args.max_concurrency, args.kv_per_request)
    app = build_app(args.model, args.latency_ms, state)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
