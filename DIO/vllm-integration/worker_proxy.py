"""
DIO v3 — vLLM Worker Proxy (Sidecar)

This process bridges DIO's gRPC-based Control Plane with vLLM's
OpenAI-compatible HTTP API. It:

  1. Starts a gRPC server implementing DIO's InferenceWorker.Predict
  2. Translates each Predict call into an HTTP request to vLLM's
     /v1/completions endpoint (with streaming for real TTFT)
  3. Collects real VRAM telemetry via NVML (pynvml)
  4. Registers with the DIO Manager on startup
  5. Reports real token counts, TTFT, and latency back to DIO

Architecture:
  [DIO Go Manager] --gRPC Predict--> [This Proxy] --HTTP /v1/completions--> [vLLM Engine]
                  <--gRPC Response--              <--JSON/SSE Response--

Usage:
  python worker_proxy.py \
    --worker-id vllm-a100-0 \
    --port 50060 \
    --vllm-url http://localhost:8000 \
    --manager-addr localhost:50055 \
    --gpu-index 0 \
    --tier large \
    --model-id meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import threading
from concurrent import futures

import grpc
import requests

# --- NVML for real VRAM telemetry ---
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlDeviceGetTemperature,
        nvmlDeviceGetUtilizationRates,
        NVML_TEMPERATURE_GPU,
        nvmlShutdown,
    )
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# --- Proto imports ---
# Add parent directories to path for proto imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, DIO_ROOT)
sys.path.insert(0, os.path.join(DIO_ROOT, "benchmarks"))

try:
    import api.proto.dio_pb2 as pb
    import api.proto.dio_pb2_grpc as pb_grpc
except ImportError:
    # Fallback: try direct import (when running from benchmarks dir)
    try:
        import dio_pb2 as pb
        import dio_pb2_grpc as pb_grpc
    except ImportError:
        print("ERROR: Cannot import DIO protobuf files.")
        print("  Ensure api/proto/dio_pb2.py exists or run from the DIO root.")
        sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DIO-vLLM-Proxy")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU Telemetry Collector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GPUTelemetry:
    """Collects real GPU metrics via NVIDIA Management Library (NVML)."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.handle = None
        self.gpu_name = "Unknown"
        self.total_vram_mb = 0

        if NVML_AVAILABLE:
            try:
                nvmlInit()
                self.handle = nvmlDeviceGetHandleByIndex(gpu_index)
                # nvmlDeviceGetName may return bytes or str depending on version
                name = nvmlDeviceGetName(self.handle)
                self.gpu_name = name.decode("utf-8") if isinstance(name, bytes) else name
                mem_info = nvmlDeviceGetMemoryInfo(self.handle)
                self.total_vram_mb = mem_info.total // (1024 * 1024)
                logger.info(
                    f"🖥️  GPU {gpu_index}: {self.gpu_name} "
                    f"({self.total_vram_mb} MB VRAM)"
                )
            except Exception as e:
                logger.warning(f"NVML init failed for GPU {gpu_index}: {e}")
                self.handle = None
        else:
            logger.warning(
                "pynvml not installed — VRAM telemetry will be estimated. "
                "Install with: pip install pynvml"
            )

    def get_free_vram_mb(self) -> int:
        """Returns free VRAM in MB from NVML, or estimate if unavailable."""
        if self.handle:
            try:
                mem_info = nvmlDeviceGetMemoryInfo(self.handle)
                return int(mem_info.free // (1024 * 1024))
            except Exception:
                pass
        # Fallback: assume 80% free of a 24GB card
        return 19200

    def get_total_vram_mb(self) -> int:
        """Returns total VRAM in MB."""
        if self.total_vram_mb > 0:
            return self.total_vram_mb
        return 24576  # Default 24GB

    def get_utilization(self) -> float:
        """Returns GPU utilization percentage (0-100)."""
        if self.handle:
            try:
                util = nvmlDeviceGetUtilizationRates(self.handle)
                return float(util.gpu)
            except Exception:
                pass
        return 0.0

    def get_temperature(self) -> float:
        """Returns GPU temperature in Celsius."""
        if self.handle:
            try:
                return float(
                    nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
                )
            except Exception:
                pass
        return 0.0

    def shutdown(self):
        """Clean up NVML."""
        if NVML_AVAILABLE and self.handle:
            try:
                nvmlShutdown()
            except Exception:
                pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# vLLM HTTP Client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class VLLMClient:
    """
    Communicates with a vLLM OpenAI-compatible API server.
    Supports both streaming (for real TTFT) and non-streaming modes.
    """

    def __init__(self, base_url: str, model_id: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self.session = requests.Session()
        # Keep-alive for reduced connection overhead
        self.session.headers.update({"Connection": "keep-alive"})

    def health_check(self) -> bool:
        """Check if the vLLM server is ready."""
        try:
            resp = self.session.get(
                f"{self.base_url}/health", timeout=5
            )
            return resp.status_code == 200
        except Exception:
            # Some vLLM versions don't have /health, try /v1/models
            try:
                resp = self.session.get(
                    f"{self.base_url}/v1/models", timeout=5
                )
                return resp.status_code == 200
            except Exception:
                return False

    def generate_streaming(
        self, prompt: str, max_tokens: int = 200, temperature: float = 0.7
    ) -> dict:
        """
        Send a streaming completion request to vLLM.
        Returns dict with: output, latency_ms, ttft_ms, prompt_tokens,
                          completion_tokens, total_tokens
        """
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        start_time = time.perf_counter()
        ttft_ms = 0.0
        output_chunks = []
        first_token_received = False
        completion_tokens = 0

        try:
            resp = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue

                data_str = line_str[6:]  # Strip "data: " prefix
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Record TTFT on first chunk with content
                if not first_token_received:
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("text", ""):
                        ttft_ms = (time.perf_counter() - start_time) * 1000
                        first_token_received = True

                # Accumulate output text
                choices = chunk.get("choices", [])
                if choices:
                    text = choices[0].get("text", "")
                    if text:
                        output_chunks.append(text)
                        completion_tokens += 1  # Approximate per-chunk

                # vLLM streaming may include usage in final chunk
                usage = chunk.get("usage")
                if usage:
                    completion_tokens = usage.get(
                        "completion_tokens", completion_tokens
                    )

            total_latency_ms = (time.perf_counter() - start_time) * 1000

            # If we never got TTFT (empty response), use total latency
            if ttft_ms == 0.0:
                ttft_ms = total_latency_ms * 0.15

            output_text = "".join(output_chunks)

            # Estimate prompt tokens from input length (BPE ≈ 4 bytes/token)
            prompt_tokens = max(len(prompt) // 4, 1)

            return {
                "output": output_text,
                "latency_ms": total_latency_ms,
                "ttft_ms": ttft_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        except requests.exceptions.Timeout:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(f"vLLM request timed out after {elapsed:.0f}ms")
            return {
                "output": "ERROR: vLLM timeout",
                "latency_ms": elapsed,
                "ttft_ms": elapsed,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(f"vLLM request failed: {e}")
            return {
                "output": f"ERROR: {str(e)}",
                "latency_ms": elapsed,
                "ttft_ms": elapsed,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    def generate_sync(
        self, prompt: str, max_tokens: int = 200, temperature: float = 0.7
    ) -> dict:
        """
        Non-streaming completion (fallback).
        Returns same dict format as generate_streaming.
        """
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        start_time = time.perf_counter()

        try:
            resp = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            total_latency_ms = (time.perf_counter() - start_time) * 1000

            data = resp.json()
            usage = data.get("usage", {})
            choices = data.get("choices", [{}])

            return {
                "output": choices[0].get("text", "") if choices else "",
                "latency_ms": total_latency_ms,
                "ttft_ms": total_latency_ms * 0.15,  # Estimated
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(f"vLLM sync request failed: {e}")
            return {
                "output": f"ERROR: {str(e)}",
                "latency_ms": elapsed,
                "ttft_ms": elapsed,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIO gRPC Worker Implementation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class VLLMWorkerProxy(pb_grpc.InferenceWorkerServicer):
    """
    Implements DIO's InferenceWorker gRPC service by proxying
    requests to a vLLM instance.

    This is the "Sidecar" pattern: it sits next to vLLM and
    translates between DIO's protocol and vLLM's HTTP API.
    """

    def __init__(
        self,
        vllm_client: VLLMClient,
        gpu_telemetry: GPUTelemetry,
        worker_id: str,
        use_streaming: bool = True,
        max_tokens: int = 200,
        latency_mult: float = 1.0,
    ):
        self.vllm_client = vllm_client
        self.gpu = gpu_telemetry
        self.worker_id = worker_id
        self.use_streaming = use_streaming
        self.max_tokens = max_tokens
        self.latency_mult = latency_mult

        # Request counter for telemetry
        self.request_count = 0
        self.total_tokens_served = 0

    def Predict(self, request, context):
        """
        Core gRPC handler: receives DIO InferenceRequest,
        forwards to vLLM, returns DIO InferenceResponse with
        real telemetry.
        """
        self.request_count += 1
        req_id = self.request_count

        # 1. Extract prompt from gRPC request
        try:
            prompt = request.data.decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            prompt = str(request.data)

        input_chars = len(prompt)
        logger.info(
            f"📨 [{req_id}] Predict request: {input_chars} chars "
            f"(~{input_chars // 4} tokens) | tier={request.tier}"
        )

        # 2. Forward to vLLM (streaming for real TTFT, sync as fallback)
        if self.use_streaming:
            result = self.vllm_client.generate_streaming(
                prompt=prompt,
                max_tokens=self.max_tokens,
            )
        else:
            result = self.vllm_client.generate_sync(
                prompt=prompt,
                max_tokens=self.max_tokens,
            )

        # 3. Apply heterogeneity multiplier (for simulated slow hardware)
        actual_latency = result["latency_ms"] * self.latency_mult
        actual_ttft = result["ttft_ms"] * self.latency_mult

        # If latency multiplier > 1, inject extra sleep to simulate slow GPU
        if self.latency_mult > 1.0:
            extra_sleep_ms = actual_latency - result["latency_ms"]
            if extra_sleep_ms > 0:
                time.sleep(extra_sleep_ms / 1000.0)

        # 4. Update telemetry counters
        self.total_tokens_served += result["total_tokens"]

        # 5. Log with real metrics
        free_vram = self.gpu.get_free_vram_mb()
        logger.info(
            f"✅ [{req_id}] Complete: "
            f"latency={actual_latency:.1f}ms, "
            f"ttft={actual_ttft:.1f}ms, "
            f"tokens={result['total_tokens']} "
            f"(prompt={result['prompt_tokens']}, "
            f"completion={result['completion_tokens']}), "
            f"freeVRAM={free_vram}MB"
        )

        # 6. Build DIO-compatible gRPC response
        response = pb.InferenceResponse(
            output=result["output"].encode("utf-8"),
            latency_ms=float(actual_latency),
            tokens_used=int(result["total_tokens"]),
            ttft_ms=float(actual_ttft),
        )

        # Set extended fields if the proto supports them
        try:
            response.prompt_tokens = int(result["prompt_tokens"])
            response.completion_tokens = int(result["completion_tokens"])
        except AttributeError:
            # Proto doesn't have these fields yet — that's OK
            pass

        return response

    def CheckHealth(self, request, context):
        """Health check — verify vLLM is responsive."""
        from google.protobuf.empty_pb2 import Empty

        if self.vllm_client.health_check():
            return Empty()
        else:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("vLLM engine is not responding")
            return Empty()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Manager Registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def register_with_manager(args, worker_address: str, gpu: GPUTelemetry):
    """Register this vLLM proxy with the DIO Manager via gRPC."""
    logger.info(f"📡 Registering with DIO Manager at {args.manager_addr}...")

    max_retries = 10
    retry_delay = 3  # seconds

    for attempt in range(max_retries):
        try:
            channel = grpc.insecure_channel(args.manager_addr)
            stub = pb_grpc.OrchestratorStub(channel)

            # Build registration request
            reg_request = pb.RegisterRequest(
                worker_id=args.worker_id,
                address=worker_address,
                tier=args.tier,
                vram_gb=int(gpu.get_total_vram_mb()),  # Send as MB (field name is misleading)
            )

            # Set extended fields if available
            try:
                reg_request.engine_type = "vllm"
                reg_request.free_vram_mb = int(gpu.get_free_vram_mb())
            except AttributeError:
                pass  # Proto doesn't have these fields yet

            resp = stub.RegisterWorker(reg_request)
            channel.close()

            if resp.success:
                logger.info(
                    f"✅ Registered as '{args.worker_id}' "
                    f"(engine=vllm, tier={args.tier}, "
                    f"vram={gpu.get_total_vram_mb()}MB)"
                )
                return True
            else:
                logger.warning("Manager rejected registration.")
                return False

        except grpc.RpcError as e:
            logger.warning(
                f"Registration attempt {attempt + 1}/{max_retries} failed: "
                f"{e.code()} - {e.details()}"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            logger.warning(
                f"Registration attempt {attempt + 1}/{max_retries} failed: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    logger.error("❌ Failed to register with Manager after all retries.")
    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VRAM Telemetry Heartbeat
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def start_vram_heartbeat(
    args, worker_address: str, gpu: GPUTelemetry, interval: int = 30
):
    """
    Periodically re-register with updated VRAM info.
    This gives the DIO scheduler live memory pressure data.
    """

    def heartbeat_loop():
        while True:
            time.sleep(interval)
            try:
                channel = grpc.insecure_channel(args.manager_addr)
                stub = pb_grpc.OrchestratorStub(channel)

                reg = pb.RegisterRequest(
                    worker_id=args.worker_id,
                    address=worker_address,
                    tier=args.tier,
                    vram_gb=int(gpu.get_free_vram_mb()),  # Update with CURRENT free VRAM
                )

                try:
                    reg.engine_type = "vllm"
                    reg.free_vram_mb = int(gpu.get_free_vram_mb())
                except AttributeError:
                    pass

                stub.RegisterWorker(reg)
                channel.close()

                logger.debug(
                    f"💓 Heartbeat sent: freeVRAM={gpu.get_free_vram_mb()}MB, "
                    f"util={gpu.get_utilization():.0f}%, "
                    f"temp={gpu.get_temperature():.0f}°C"
                )
            except Exception as e:
                logger.debug(f"Heartbeat failed (non-fatal): {e}")

    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    logger.info(f"💓 VRAM heartbeat started (interval={interval}s)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Wait for vLLM readiness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def wait_for_vllm(vllm_client: VLLMClient, timeout: int = 300):
    """Block until the vLLM server is healthy."""
    logger.info(
        f"⏳ Waiting for vLLM at {vllm_client.base_url} "
        f"(timeout={timeout}s)..."
    )
    start = time.time()
    while time.time() - start < timeout:
        if vllm_client.health_check():
            elapsed = time.time() - start
            logger.info(f"✅ vLLM is ready! (took {elapsed:.1f}s)")
            return True
        time.sleep(2)

    logger.error(f"❌ vLLM not ready after {timeout}s. Is it running?")
    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(
        description="DIO vLLM Worker Proxy — Sidecar for vLLM integration"
    )
    parser.add_argument(
        "--worker-id",
        required=True,
        help="Unique worker identifier (e.g., 'vllm-a100-0')",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50060,
        help="gRPC listen port for DIO Predict calls",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM OpenAI-compatible API URL",
    )
    parser.add_argument(
        "--manager-addr",
        default="localhost:50055",
        help="DIO Manager gRPC address",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index for NVML telemetry",
    )
    parser.add_argument(
        "--tier",
        default="large",
        choices=["small", "large"],
        help="Worker tier for DIO scheduling",
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model ID (must match vLLM's loaded model)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Max output tokens per request",
    )
    parser.add_argument(
        "--latency-mult",
        type=float,
        default=1.0,
        help="Latency multiplier to simulate slow hardware (1.0=normal)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming (use sync requests, estimated TTFT)",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="VRAM heartbeat interval in seconds",
    )
    parser.add_argument(
        "--vllm-timeout",
        type=int,
        default=300,
        help="Timeout waiting for vLLM to become ready (seconds)",
    )
    args = parser.parse_args()

    # ── Initialize GPU telemetry ──
    gpu = GPUTelemetry(gpu_index=args.gpu_index)

    # ── Initialize vLLM client ──
    vllm_client = VLLMClient(
        base_url=args.vllm_url,
        model_id=args.model_id,
        timeout=args.max_tokens,  # Rough timeout based on expected gen time
    )

    # ── Wait for vLLM to be ready ──
    if not wait_for_vllm(vllm_client, timeout=args.vllm_timeout):
        logger.error(
            "vLLM is not running. Start it with:\n"
            f"  python -m vllm.entrypoints.openai.api_server "
            f"--model {args.model_id} --port 8000"
        )
        sys.exit(1)

    # ── Create the proxy service ──
    proxy = VLLMWorkerProxy(
        vllm_client=vllm_client,
        gpu_telemetry=gpu,
        worker_id=args.worker_id,
        use_streaming=not args.no_streaming,
        max_tokens=args.max_tokens,
        latency_mult=args.latency_mult,
    )

    # ── Start gRPC server ──
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=16),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )
    pb_grpc.add_InferenceWorkerServicer_to_server(proxy, server)
    listen_addr = f"0.0.0.0:{args.port}"
    server.add_insecure_port(listen_addr)
    server.start()

    worker_address = f"localhost:{args.port}"
    logger.info(
        f"🚀 vLLM Worker Proxy '{args.worker_id}' listening on {listen_addr}"
    )
    logger.info(
        f"   Engine: vLLM at {args.vllm_url}"
    )
    logger.info(
        f"   Model:  {args.model_id}"
    )
    logger.info(
        f"   Streaming: {'ON' if not args.no_streaming else 'OFF'}"
    )
    logger.info(
        f"   GPU: {gpu.gpu_name} "
        f"(free={gpu.get_free_vram_mb()}MB / total={gpu.get_total_vram_mb()}MB)"
    )

    # ── Register with DIO Manager ──
    if not register_with_manager(args, worker_address, gpu):
        logger.warning(
            "⚠️ Continuing without registration — "
            "Manager may be starting up. Heartbeats will retry."
        )

    # ── Start VRAM heartbeat ──
    start_vram_heartbeat(
        args, worker_address, gpu, interval=args.heartbeat_interval
    )

    # ── Graceful shutdown ──
    def shutdown_handler(signum, frame):
        logger.info("🛑 Shutting down...")
        server.stop(grace=5)
        gpu.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    logger.info("✅ Ready to serve predictions via vLLM!")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
