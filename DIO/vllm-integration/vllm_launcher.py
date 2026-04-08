"""
DIO v3 — vLLM Engine Launcher

Launches a vLLM OpenAI-compatible API server as a subprocess
with the correct GPU and model configuration.

This is a helper script — you can also start vLLM manually:
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --port 8000

Usage:
  python vllm_launcher.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --port 8000 \
    --gpu-index 0 \
    --gpu-memory-utilization 0.85

After launching, start the worker_proxy.py to connect DIO to this vLLM instance.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DIO-vLLM-Launcher")


def wait_for_vllm_ready(port: int, timeout: int = 600) -> bool:
    """Block until vLLM's HTTP API is responsive."""
    url = f"http://localhost:{port}/v1/models"
    logger.info(f"⏳ Waiting for vLLM to be ready on port {port}...")

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])]
                logger.info(f"✅ vLLM is ready! Models: {models}")
                return True
        except Exception:
            pass
        time.sleep(3)

    logger.error(f"❌ vLLM did not become ready within {timeout}s")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="DIO vLLM Launcher — Starts a vLLM API server"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model ID to serve",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="vLLM API server port"
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU device index (sets CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory for vLLM (0.0-1.0)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (number of GPUs per model)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", "squeezellm", "fp8", None],
        help="Quantization method (None for default precision)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences per iteration (batch size)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph (useful for debugging or small GPUs)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind vLLM server",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in model repo",
    )
    args = parser.parse_args()

    # Set GPU visibility
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    # Build vLLM command
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--host",
        args.host,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--dtype",
        args.dtype,
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]

    if args.quantization:
        cmd.extend(["--quantization", args.quantization])

    if args.enforce_eager:
        cmd.append("--enforce-eager")

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    logger.info("=" * 60)
    logger.info("DIO vLLM Launcher")
    logger.info("=" * 60)
    logger.info(f"  Model:    {args.model}")
    logger.info(f"  Port:     {args.port}")
    logger.info(f"  GPU:      {args.gpu_index}")
    logger.info(f"  Mem Util: {args.gpu_memory_utilization}")
    logger.info(f"  Max Len:  {args.max_model_len}")
    logger.info(f"  TP Size:  {args.tensor_parallel_size}")
    logger.info(f"  Dtype:    {args.dtype}")
    logger.info(f"  Quant:    {args.quantization or 'None'}")
    logger.info("=" * 60)
    logger.info(f"  Command:  {' '.join(cmd)}")
    logger.info("=" * 60)

    # Launch vLLM as a subprocess
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Forward vLLM output to our logger
    def stream_output():
        for line in iter(process.stdout.readline, b""):
            line_str = line.decode("utf-8", errors="replace").rstrip()
            if line_str:
                logger.info(f"[vLLM] {line_str}")

    import threading

    output_thread = threading.Thread(target=stream_output, daemon=True)
    output_thread.start()

    # Wait for vLLM to be ready
    if not wait_for_vllm_ready(args.port, timeout=600):
        logger.error("vLLM failed to start. Check logs above.")
        process.terminate()
        sys.exit(1)

    logger.info(
        f"\n🎉 vLLM is serving '{args.model}' on port {args.port}!\n"
        f"   Next: Start the DIO proxy:\n"
        f"   python worker_proxy.py \\\n"
        f"     --worker-id vllm-gpu{args.gpu_index} \\\n"
        f"     --port 50060 \\\n"
        f"     --vllm-url http://localhost:{args.port} \\\n"
        f"     --manager-addr localhost:50055 \\\n"
        f"     --gpu-index {args.gpu_index} \\\n"
        f"     --tier large\n"
    )

    # Handle graceful shutdown
    def shutdown(signum, frame):
        logger.info("🛑 Shutting down vLLM...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep alive
    try:
        process.wait()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
