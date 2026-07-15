#!/usr/bin/env python3
"""
OpenAI-compatible delay proxy for real-GPU heterogeneity tests.

Forwards /v1/* (and health) to an upstream engine, then sleeps so wall-clock
end-to-end latency seen by DIO is approximately ``latency_mult ×`` the upstream
time. Tokens still come from the real engine — only the observed service time
is inflated (models a slower peer / throttled GPU without a second SKU).

Usage:
  python scripts/latency_delay_proxy.py \\
      --upstream http://127.0.0.1:18001 \\
      --port 18101 \\
      --latency-mult 2.0
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse


def build_app(upstream: str, latency_mult: float) -> FastAPI:
    upstream = upstream.rstrip("/")
    mult = max(1.0, float(latency_mult))
    app = FastAPI(title="dio-latency-delay-proxy")
    client = httpx.Client(timeout=300.0)

    def _proxy(path: str, request: Request, body: bytes) -> Response:
        t0 = time.perf_counter()
        url = f"{upstream}{path}"
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        try:
            r = client.request(
                request.method,
                url,
                content=body,
                headers=headers,
                params=dict(request.query_params),
            )
        except Exception as e:
            return JSONResponse({"error": f"upstream: {e}"}, status_code=502)
        elapsed = time.perf_counter() - t0
        if mult > 1.0:
            # Sleep so total wall time ≈ mult * upstream_time
            time.sleep(elapsed * (mult - 1.0))
        # Drop hop-by-hop headers
        out_headers = {
            k: v
            for k, v in r.headers.items()
            if k.lower()
            not in (
                "content-encoding",
                "transfer-encoding",
                "content-length",
                "connection",
            )
        }
        out_headers["X-DIO-Delay-Mult"] = str(mult)
        out_headers["X-DIO-Upstream-Ms"] = f"{elapsed * 1000.0:.2f}"
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=out_headers,
            media_type=r.headers.get("content-type"),
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def catch_all(path: str, request: Request):
        body = await request.body()
        return _proxy("/" + path if path else "/", request, body)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "upstream": upstream, "latency_mult": mult}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Latency-multiplying OpenAI proxy")
    ap.add_argument("--upstream", required=True, help="Real engine base URL")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18101)
    ap.add_argument("--latency-mult", type=float, default=2.0)
    args = ap.parse_args()
    app = build_app(args.upstream, args.latency_mult)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
