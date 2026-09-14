"""
Model Context Protocol (MCP) Server for DIO.

Exposes DIO intelligence to AI-powered IDEs (Cursor, Claude Desktop, Antigravity,
VS Code, Windsurf, Zed) over JSON-RPC 2.0 stdio:

  1. Query DIO for available models, backends, and cluster health
  2. Route prompts through DIO's predictive NLMS scheduler
  3. Get latency/cost predictions before sending requests
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

import httpx

# Logging must go strictly to stderr so stdout remains pure JSON-RPC
log = logging.getLogger("dio.mcp")
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s [dio.mcp] %(message)s")


class DIOMCPServer:
    """
    Model Context Protocol server for DIO.

    Supports stdio transport (newline-delimited JSON-RPC 2.0).
    """

    def __init__(self, gateway_url: str = "http://127.0.0.1:8085") -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # --- Tool Implementations ---

    async def get_models(self) -> Dict[str, Any]:
        """Query DIO for available models, backend bindings, and health."""
        client = self._http()
        try:
            r = await client.get(f"{self.gateway_url}/v1/models")
            health_r = await client.get(f"{self.gateway_url}/health")
            models_data = r.json().get("data", []) if r.status_code == 200 else []
            health_data = health_r.json() if health_r.status_code == 200 else {}
            return {
                "status": "online",
                "gateway_url": self.gateway_url,
                "models": [m.get("id") for m in models_data],
                "model_details": models_data,
                "cluster_health": health_data,
            }
        except Exception as e:
            # Fall back to local config discovery if gateway is not currently reachable
            from dio.config_file import discover_config, load_config_file

            cfg_path = discover_config()
            if cfg_path:
                try:
                    backends, cfg, model_map = load_config_file(str(cfg_path))
                    return {
                        "status": "gateway_offline (loaded from local dio.yaml)",
                        "gateway_url": self.gateway_url,
                        "config_file": str(cfg_path),
                        "models": list(model_map.keys()) or [b.model for b in backends if b.model],
                        "backends": [{"id": b.id, "url": b.base_url, "tier": b.tier} for b in backends],
                        "strategy": cfg.strategy,
                    }
                except Exception:
                    pass
            return {
                "status": "offline",
                "gateway_url": self.gateway_url,
                "error": f"Could not connect to DIO gateway at {self.gateway_url}: {e}",
                "hint": "Start the gateway with: dio serve",
            }

    async def predict_latency(
        self, prompt: str, model: Optional[str] = None, tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Predict latency and scheduling scores before sending requests.
        Uses DIO's dual-timescale NLMS parameters and queue delays.
        """
        client = self._http()
        est_tokens = tokens if tokens is not None else max(1, len(prompt) // 4)
        try:
            r = await client.get(f"{self.gateway_url}/debug/metrics")
            if r.status_code != 200:
                return {"error": f"Gateway returned status {r.status_code}"}

            metrics = r.json()
            workers = metrics.get("workers", {})
            slo_ms = float(metrics.get("admission", {}).get("slo_ms", 30000.0))

            predictions = []
            for wid, w in workers.items():
                fast_s = float(w.get("fast_slope") or 4.0)
                slow_s = float(w.get("slow_slope") or 4.0)
                alpha = float(w.get("alpha") or 0.8)
                eff_slope = alpha * fast_s + (1.0 - alpha) * slow_s
                intercept = float(w.get("intercept") or 100.0)
                pending = int(w.get("pending") or 0)
                y_hat = eff_slope * est_tokens + intercept
                wait_ms = pending * max(10.0, y_hat * 0.25)
                tier_cost = float(w.get("tier_cost") or 0.0)
                total_cost = wait_ms + y_hat + tier_cost

                predictions.append({
                    "backend_id": wid,
                    "predicted_execution_ms": round(y_hat, 1),
                    "queue_wait_ms": round(wait_ms, 1),
                    "total_score": round(total_cost, 1),
                    "pending_requests": pending,
                    "learned_slope_ms_per_token": round(eff_slope, 2),
                    "healthy": w.get("healthy", True),
                })

            predictions.sort(key=lambda x: x["total_score"])
            best = predictions[0] if predictions else None
            return {
                "status": "online",
                "estimated_tokens": est_tokens,
                "model": model or "default",
                "recommended_backend": best["backend_id"] if best else None,
                "predicted_latency_ms": best["total_score"] if best else None,
                "slo_ms": slo_ms,
                "admissible": (best["total_score"] <= slo_ms) if best else False,
                "candidate_backends": predictions,
            }
        except Exception as e:
            return {
                "status": "gateway_offline",
                "estimated_tokens": est_tokens,
                "model": model or "default",
                "predicted_latency_ms": round(est_tokens * 15.0 + 100.0, 1),
                "note": f"Estimated via heuristic (DIO gateway at {self.gateway_url} is offline: {e})",
            }

    async def route_prompt(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Route an inference request through DIO's smart scheduler.
        DIO automatically dispatches to the optimal backend with lowest joint cost.
        """
        client = self._http()
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model:
            payload["model"] = model

        try:
            resp = await client.post(
                f"{self.gateway_url}/v1/chat/completions",
                json=payload,
            )
            if resp.status_code != 200:
                return {
                    "error": f"Gateway error {resp.status_code}",
                    "details": resp.text,
                }
            data = resp.json()
            choices = data.get("choices") or []
            content = choices[0]["message"]["content"] if choices else ""
            backend_used = getattr(resp, "headers", {}).get("X-DIO-Backend", "unknown")
            e2e_ms = getattr(resp, "headers", {}).get("X-DIO-E2E-Ms", "0.0")

            return {
                "content": content,
                "backend_used": backend_used,
                "latency_ms": float(e2e_ms),
                "model": data.get("model", model),
                "usage": data.get("usage", {}),
            }
        except Exception as e:
            return {"error": f"Failed to route prompt: {e}"}

    async def cluster_status(self) -> Dict[str, Any]:
        """Get live DIO cluster telemetry, learned slopes, and KV pressure."""
        client = self._http()
        try:
            m_resp = await client.get(f"{self.gateway_url}/debug/metrics")
            eng_resp = await client.get(f"{self.gateway_url}/debug/engine")
            h_resp = await client.get(f"{self.gateway_url}/health")

            return {
                "status": "online",
                "gateway": h_resp.json() if h_resp.status_code == 200 else {},
                "scheduler": m_resp.json() if m_resp.status_code == 200 else {},
                "engine_metrics": eng_resp.json() if eng_resp.status_code == 200 else {},
            }
        except Exception as e:
            return {"status": "offline", "error": f"Could not fetch cluster status: {e}"}

    # --- JSON-RPC 2.0 Handler ---

    async def handle_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = req.get("method")
        msg_id = req.get("id")
        params = req.get("params") or {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "dio-mcp",
                        "version": "0.4.0",
                    },
                },
            }

        elif method in ("notifications/initialized", "initialized"):
            return None

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        {
                            "name": "dio_get_models",
                            "description": "Query DIO gateway for available LLM models, active backends, and cluster health.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                        {
                            "name": "dio_predict_latency",
                            "description": "Get latency and cost predictions before sending requests. Predicts queue delay, execution latency, and optimal backend using DIO's dual-timescale NLMS filter.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "prompt": {
                                        "type": "string",
                                        "description": "The prompt text or query to estimate",
                                    },
                                    "model": {
                                        "type": "string",
                                        "description": "Target model name (optional)",
                                    },
                                    "tokens": {
                                        "type": "integer",
                                        "description": "Estimated token count (optional)",
                                    },
                                },
                                "required": ["prompt"],
                            },
                        },
                        {
                            "name": "dio_route_prompt",
                            "description": "Route a prompt through DIO's intelligent NLMS scheduler to the optimal backend and return the completion.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "prompt": {
                                        "type": "string",
                                        "description": "Prompt text to send",
                                    },
                                    "model": {
                                        "type": "string",
                                        "description": "Target model name (optional)",
                                    },
                                    "max_tokens": {
                                        "type": "integer",
                                        "description": "Maximum tokens to generate (default: 256)",
                                    },
                                    "temperature": {
                                        "type": "number",
                                        "description": "Sampling temperature (default: 0.7)",
                                    },
                                },
                                "required": ["prompt"],
                            },
                        },
                        {
                            "name": "dio_cluster_status",
                            "description": "Get live telemetry from the DIO cluster, including learned worker slopes, KV-cache pressure, and admission goodput statistics.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    ]
                },
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments") or {}

            try:
                if tool_name == "dio_get_models":
                    res = await self.get_models()
                elif tool_name == "dio_predict_latency":
                    res = await self.predict_latency(
                        prompt=str(args.get("prompt", "")),
                        model=args.get("model"),
                        tokens=args.get("tokens"),
                    )
                elif tool_name == "dio_route_prompt":
                    res = await self.route_prompt(
                        prompt=str(args.get("prompt", "")),
                        model=args.get("model"),
                        max_tokens=int(args.get("max_tokens", 256)),
                        temperature=float(args.get("temperature", 0.7)),
                    )
                elif tool_name == "dio_cluster_status":
                    res = await self.cluster_status()
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32601, "message": f"Unknown tool '{tool_name}'"},
                    }

                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(res, indent=2)}
                        ],
                        "isError": "error" in res,
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Tool execution failed: {e}"}],
                        "isError": True,
                    },
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method '{method}' not supported"},
            }

    async def run_stdio(self) -> None:
        """Main stdio loop reading JSON-RPC from stdin and writing to stdout."""
        log.info("DIO MCP server started (gateway: %s)", self.gateway_url)

        while True:
            try:
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                req = json.loads(line)
                resp = await self.handle_request(req)
                if resp is not None:
                    out = json.dumps(resp) + "\n"
                    sys.stdout.write(out)
                    sys.stdout.flush()
            except json.JSONDecodeError:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                log.exception("Error in stdio loop: %s", e)


def main() -> None:
    """CLI entry point for dio mcp."""
    import argparse

    parser = argparse.ArgumentParser(description="DIO Model Context Protocol (MCP) server")
    parser.add_argument(
        "--gateway-url",
        "-g",
        default="http://127.0.0.1:8085",
        help="Base URL of the running DIO gateway (default: http://127.0.0.1:8085)",
    )
    args = parser.parse_args()

    server = DIOMCPServer(gateway_url=args.gateway_url)
    try:
        asyncio.run(server.run_stdio())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
