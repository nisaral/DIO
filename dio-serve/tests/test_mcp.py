"""Tests for Model Context Protocol (MCP) server in dio.mcp."""

from __future__ import annotations

import json
from typing import ClassVar, Dict
from unittest.mock import patch

import pytest

from dio.mcp import DIOMCPServer


@pytest.mark.asyncio
async def test_mcp_initialize():
    server = DIOMCPServer(gateway_url="http://127.0.0.1:8085")
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "cursor", "version": "0.40.0"},
        },
    }
    resp = await server.handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "result" in resp
    assert resp["result"]["serverInfo"]["name"] == "dio-mcp"
    assert "capabilities" in resp["result"]
    assert "tools" in resp["result"]["capabilities"]
    await server.close()


@pytest.mark.asyncio
async def test_mcp_tools_list():
    server = DIOMCPServer(gateway_url="http://127.0.0.1:8085")
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    resp = await server.handle_request(req)
    assert resp["id"] == 2
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "dio_get_models" in tool_names
    assert "dio_predict_latency" in tool_names
    assert "dio_route_prompt" in tool_names
    assert "dio_cluster_status" in tool_names
    await server.close()


@pytest.mark.asyncio
async def test_mcp_unknown_method():
    server = DIOMCPServer(gateway_url="http://127.0.0.1:8085")
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "unknown_rpc_method",
        "params": {},
    }
    resp = await server.handle_request(req)
    assert "error" in resp
    assert resp["error"]["code"] == -32601
    await server.close()


@pytest.mark.asyncio
async def test_mcp_unknown_tool():
    server = DIOMCPServer(gateway_url="http://127.0.0.1:8085")
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "non_existent_tool", "arguments": {}},
    }
    resp = await server.handle_request(req)
    assert "error" in resp
    assert resp["error"]["code"] == -32601
    await server.close()


@pytest.mark.asyncio
async def test_mcp_get_models_offline():
    server = DIOMCPServer(gateway_url="http://127.0.0.1:9999")  # Unreachable port
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "dio_get_models", "arguments": {}},
    }
    resp = await server.handle_request(req)
    assert resp["id"] == 5
    assert "result" in resp
    content = json.loads(resp["result"]["content"][0]["text"])
    # Should report gateway status (either offline or fallback config)
    assert "status" in content
    assert "gateway_url" in content
    await server.close()


@pytest.mark.asyncio
async def test_mcp_predict_latency_offline():
    server = DIOMCPServer(gateway_url="http://127.0.0.1:9999")
    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "dio_predict_latency",
            "arguments": {
                "prompt": "Write a python function to compute fibonacci numbers",
                "tokens": 100,
            },
        },
    }
    resp = await server.handle_request(req)
    assert resp["id"] == 6
    assert "result" in resp
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["status"] == "gateway_offline"
    assert content["estimated_tokens"] == 100
    assert content["predicted_latency_ms"] > 0
    await server.close()


@pytest.mark.asyncio
async def test_mcp_tools_with_mocked_gateway():
    server = DIOMCPServer(gateway_url="http://127.0.0.1:8085")

    # Mock gateway responses
    async def mock_get(url, *args, **kwargs):
        class MockResp:
            def __init__(self, status_code, data):
                self.status_code = status_code
                self._data = data

            def json(self):
                return self._data

        if "/v1/models" in url:
            return MockResp(200, {"data": [{"id": "meta-llama/Llama-3-8B-Instruct"}]})
        elif "/health" in url:
            return MockResp(200, {"status": "healthy", "backends_total": 2, "backends_healthy": 2})
        elif "/debug/metrics" in url:
            return MockResp(200, {
                "workers": {
                    "b0": {
                        "fast_slope": 2.0,
                        "slow_slope": 2.0,
                        "alpha": 1.0,
                        "intercept": 10.0,
                        "pending": 0,
                        "tier_cost": 0.0,
                        "healthy": True,
                    }
                },
                "admission": {"slo_ms": 5000.0},
            })
        elif "/debug/engine" in url:
            return MockResp(200, {"enabled": False})
        return MockResp(404, {})

    async def mock_post(url, *args, **kwargs):
        class MockPostResp:
            status_code = 200
            headers: ClassVar[Dict[str, str]] = {
                "X-DIO-Backend": "b0",
                "X-DIO-E2E-Ms": "42.5",
            }
            text = ""

            def json(self):
                return {
                    "choices": [{"message": {"role": "assistant", "content": "Hello from DIO!"}}],
                    "usage": {"total_tokens": 15},
                }

        return MockPostResp()

    with patch.object(server._http(), "get", side_effect=mock_get), \
         patch.object(server._http(), "post", side_effect=mock_post):

        # 1. Test dio_get_models
        req1 = {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "dio_get_models", "arguments": {}},
        }
        resp1 = await server.handle_request(req1)
        data1 = json.loads(resp1["result"]["content"][0]["text"])
        assert data1["status"] == "online"
        assert "meta-llama/Llama-3-8B-Instruct" in data1["models"]

        # 2. Test dio_predict_latency
        req2 = {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "dio_predict_latency", "arguments": {"prompt": "Hello world"}},
        }
        resp2 = await server.handle_request(req2)
        data2 = json.loads(resp2["result"]["content"][0]["text"])
        assert data2["status"] == "online"
        assert data2["recommended_backend"] == "b0"
        # y_hat = 2.0 * 2 + 10.0 = 14.0 ms
        assert data2["predicted_latency_ms"] == 14.0
        assert data2["admissible"] is True

        # 3. Test dio_cluster_status
        req3 = {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {"name": "dio_cluster_status", "arguments": {}},
        }
        resp3 = await server.handle_request(req3)
        data3 = json.loads(resp3["result"]["content"][0]["text"])
        assert data3["status"] == "online"
        assert "gateway" in data3
        assert "scheduler" in data3

        # 4. Test dio_route_prompt
        req4 = {
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "dio_route_prompt",
                "arguments": {"prompt": "Say hi", "max_tokens": 50},
            },
        }
        resp4 = await server.handle_request(req4)
        data4 = json.loads(resp4["result"]["content"][0]["text"])
        assert data4["content"] == "Hello from DIO!"

    await server.close()
