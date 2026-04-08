"""
DIO v3 — Head-to-Head Benchmark: DIO+vLLM vs. vLLM (Round-Robin)

This script runs the same workload against two configurations:
  1. BASELINE: 2 vLLM engines behind simple Round-Robin (direct HTTP)
  2. DIO v3:   2 vLLM engines behind DIO's NLMS Control Plane

It produces comparison metrics:
  - p50, p95, p99 latency
  - SLO attainment (target ≤2s TTFT)
  - Failure rate
  - Request distribution (traffic split)

Usage:
  # Ensure DIO Manager, 2 vLLM engines, and 2 proxies are running
  python run_dio_vs_vllm.py

  # Or with custom config:
  python run_dio_vs_vllm.py \
    --dio-url http://localhost:8085 \
    --vllm-urls http://localhost:8000,http://localhost:8001 \
    --duration 120 --concurrency 20
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DIO-vs-vLLM")

# ── Workload Data ──

SYNTHETIC_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to implement quicksort with detailed comments.",
    "What are the main differences between TCP and UDP protocols?",
    "Summarize the plot of Shakespeare's Hamlet in 200 words.",
    "How does a neural network learn? Explain backpropagation step by step.",
    "Write a short story about a robot discovering emotions.",
    "What is the significance of the Turing test in artificial intelligence?",
    "Explain quantum computing to a 10-year-old.",
    "Compare and contrast supervised and unsupervised machine learning.",
    "Describe the architecture of a modern web application.",
    "What are the ethical implications of autonomous weapons?",
    "Explain how a compiler transforms source code into machine code.",
    "Write a detailed analysis of supply chain management in global trade.",
    "How does PagedAttention work in vLLM? Explain the memory management.",
    "Describe the evolution of programming languages from Fortran to Rust.",
    # Long prompts for VRAM stress
    "Write a comprehensive 2000-word essay on the history of artificial intelligence, "
    "covering its origins in the 1950s, the AI winters, the rise of machine learning, "
    "deep learning breakthroughs, and current trends in large language models. Include "
    "discussion of key figures like Alan Turing, John McCarthy, Geoffrey Hinton, and "
    "Yann LeCun. Analyze the societal impact of AI in healthcare, autonomous vehicles, "
    "and natural language processing. " * 3,
    "Provide a detailed technical analysis of distributed systems architecture, "
    "covering topics such as consensus algorithms (Paxos, Raft), distributed hash "
    "tables, CAP theorem implications, eventual consistency models, vector clocks, "
    "and conflict-free replicated data types (CRDTs). Include code examples in Go "
    "and discuss real-world implementations in systems like Cassandra, DynamoDB, "
    "and CockroachDB. " * 3,
]


def load_sharegpt_prompts(path: str, max_prompts: int = 500) -> List[str]:
    """Load real prompts from ShareGPT dataset if available."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = []
        for item in data:
            if "conversations" in item:
                for turn in item["conversations"]:
                    if turn["from"] == "human" and len(turn["value"]) > 20:
                        prompts.append(turn["value"])
                        if len(prompts) >= max_prompts:
                            return prompts
                        break
        return prompts
    except Exception as e:
        logger.warning(f"Failed to load ShareGPT: {e}")
        return []


@dataclass
class RequestResult:
    strategy: str
    latency_ms: float
    ttft_ms: float
    tokens: int
    success: bool
    error: str = ""
    timestamp: float = 0.0


@dataclass
class BenchmarkConfig:
    duration_seconds: int = 120
    concurrency: int = 20
    max_tokens: int = 200
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    slo_target_ms: float = 2000.0  # 2s TTFT target


# ── Benchmark Runners ──


def send_dio_request(
    dio_url: str, prompt: str, model_id: str, max_tokens: int
) -> RequestResult:
    """Send a request through DIO's HTTP gateway."""
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{dio_url}/api/generate",
            json={
                "prompt": prompt,
                "model_id": model_id,
                "tier": "large",
            },
            timeout=300,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            return RequestResult(
                strategy="DIO+vLLM",
                latency_ms=data.get("latency_ms", elapsed_ms),
                ttft_ms=data.get("ttft_ms", elapsed_ms * 0.15),
                tokens=data.get("tokens_used", 0),
                success=True,
                timestamp=time.time(),
            )
        else:
            return RequestResult(
                strategy="DIO+vLLM",
                latency_ms=elapsed_ms,
                ttft_ms=elapsed_ms,
                tokens=0,
                success=False,
                error=f"HTTP {resp.status_code}",
                timestamp=time.time(),
            )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            strategy="DIO+vLLM",
            latency_ms=elapsed_ms,
            ttft_ms=elapsed_ms,
            tokens=0,
            success=False,
            error=str(e),
            timestamp=time.time(),
        )


def send_vllm_direct_request(
    vllm_urls: List[str],
    prompt: str,
    model_id: str,
    max_tokens: int,
    rr_counter: list,
) -> RequestResult:
    """Send a request directly to vLLM with Round-Robin selection."""
    # Simple Round-Robin
    idx = rr_counter[0] % len(vllm_urls)
    rr_counter[0] += 1
    url = vllm_urls[idx]

    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/v1/completions",
            json={
                "model": model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=300,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage", {})
            return RequestResult(
                strategy="vLLM-RoundRobin",
                latency_ms=elapsed_ms,
                ttft_ms=elapsed_ms * 0.15,  # Estimated (no streaming)
                tokens=usage.get("total_tokens", 0),
                success=True,
                timestamp=time.time(),
            )
        else:
            return RequestResult(
                strategy="vLLM-RoundRobin",
                latency_ms=elapsed_ms,
                ttft_ms=elapsed_ms,
                tokens=0,
                success=False,
                error=f"HTTP {resp.status_code}",
                timestamp=time.time(),
            )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            strategy="vLLM-RoundRobin",
            latency_ms=elapsed_ms,
            ttft_ms=elapsed_ms,
            tokens=0,
            success=False,
            error=str(e),
            timestamp=time.time(),
        )


def run_benchmark(
    strategy: str,
    config: BenchmarkConfig,
    prompts: List[str],
    dio_url: str = None,
    vllm_urls: List[str] = None,
) -> List[RequestResult]:
    """Run a timed benchmark with concurrent requests."""
    results = []
    rr_counter = [0]
    start_time = time.time()
    request_count = 0

    logger.info(
        f"▶ Starting {strategy} benchmark "
        f"(duration={config.duration_seconds}s, "
        f"concurrency={config.concurrency})"
    )

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures_list = []

        while time.time() - start_time < config.duration_seconds:
            prompt = random.choice(prompts)

            if strategy == "DIO+vLLM":
                future = executor.submit(
                    send_dio_request,
                    dio_url,
                    prompt,
                    config.model_id,
                    config.max_tokens,
                )
            else:
                future = executor.submit(
                    send_vllm_direct_request,
                    vllm_urls,
                    prompt,
                    config.model_id,
                    config.max_tokens,
                    rr_counter,
                )

            futures_list.append(future)
            request_count += 1

            # Rate limiting: ~2 req/s per concurrent slot
            time.sleep(0.05)

            # Harvest completed futures
            done = [f for f in futures_list if f.done()]
            for f in done:
                try:
                    results.append(f.result())
                except Exception as e:
                    logger.error(f"Request error: {e}")
                futures_list.remove(f)

        # Wait for remaining
        for f in as_completed(futures_list, timeout=60):
            try:
                results.append(f.result())
            except Exception as e:
                logger.error(f"Timeout waiting for result: {e}")

    elapsed = time.time() - start_time
    successes = sum(1 for r in results if r.success)
    logger.info(
        f"✅ {strategy}: {successes}/{len(results)} succeeded "
        f"in {elapsed:.1f}s ({len(results)/elapsed:.1f} req/s)"
    )

    return results


def analyze_results(results: List[RequestResult], config: BenchmarkConfig) -> dict:
    """Compute summary statistics."""
    if not results:
        return {}

    latencies = [r.latency_ms for r in results if r.success]
    ttfts = [r.ttft_ms for r in results if r.success]
    total = len(results)
    successes = len(latencies)
    failures = total - successes

    if not latencies:
        return {
            "strategy": results[0].strategy,
            "total": total,
            "failures": failures,
            "failure_rate": 1.0,
        }

    latencies.sort()
    ttfts.sort()

    slo_met = sum(1 for t in ttfts if t <= config.slo_target_ms)

    return {
        "strategy": results[0].strategy,
        "total": total,
        "successes": successes,
        "failures": failures,
        "failure_rate": failures / total if total > 0 else 0,
        "p50_latency_ms": latencies[len(latencies) // 2],
        "p95_latency_ms": latencies[int(len(latencies) * 0.95)],
        "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
        "mean_latency_ms": sum(latencies) / len(latencies),
        "p50_ttft_ms": ttfts[len(ttfts) // 2],
        "p99_ttft_ms": ttfts[int(len(ttfts) * 0.99)],
        "slo_attainment": slo_met / len(ttfts) if ttfts else 0,
        "throughput_rps": successes / (config.duration_seconds),
    }


def main():
    parser = argparse.ArgumentParser(
        description="DIO v3 Head-to-Head Benchmark"
    )
    parser.add_argument(
        "--dio-url",
        default="http://localhost:8085",
        help="DIO Manager HTTP URL",
    )
    parser.add_argument(
        "--vllm-urls",
        default="http://localhost:8000,http://localhost:8001",
        help="Comma-separated vLLM direct URLs for RR baseline",
    )
    parser.add_argument(
        "--duration", type=int, default=120, help="Benchmark duration (seconds)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=20, help="Concurrent request count"
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model ID",
    )
    parser.add_argument(
        "--sharegpt-path",
        default=None,
        help="Path to ShareGPT JSON for real prompts",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results_vllm",
        help="Output directory for results",
    )
    args = parser.parse_args()

    vllm_urls = [u.strip() for u in args.vllm_urls.split(",")]

    config = BenchmarkConfig(
        duration_seconds=args.duration,
        concurrency=args.concurrency,
        model_id=args.model_id,
    )

    # Load prompts
    prompts = SYNTHETIC_PROMPTS
    if args.sharegpt_path:
        real_prompts = load_sharegpt_prompts(args.sharegpt_path)
        if real_prompts:
            prompts = real_prompts
            logger.info(f"Loaded {len(prompts)} ShareGPT prompts")

    # Create output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DIO v3 — Head-to-Head Benchmark")
    print("  vLLM (Round-Robin) vs. DIO+vLLM (NLMS Predictive)")
    print("=" * 70)
    print(f"  Duration:    {config.duration_seconds}s")
    print(f"  Concurrency: {config.concurrency}")
    print(f"  Model:       {config.model_id}")
    print(f"  Prompts:     {len(prompts)}")
    print(f"  DIO URL:     {args.dio_url}")
    print(f"  vLLM URLs:   {vllm_urls}")
    print("=" * 70)

    # ── Phase 1: vLLM Round-Robin Baseline ──
    print("\n📊 Phase 1: vLLM Round-Robin Baseline...")
    rr_results = run_benchmark(
        "vLLM-RoundRobin", config, prompts, vllm_urls=vllm_urls
    )

    # ── Phase 2: DIO + vLLM ──
    print("\n📊 Phase 2: DIO + vLLM (NLMS Predictive)...")
    dio_results = run_benchmark(
        "DIO+vLLM", config, prompts, dio_url=args.dio_url
    )

    # ── Analysis ──
    rr_stats = analyze_results(rr_results, config)
    dio_stats = analyze_results(dio_results, config)

    # Save raw results
    all_results = rr_results + dio_results
    df = pd.DataFrame([vars(r) for r in all_results])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    # Save summary
    summary_df = pd.DataFrame([rr_stats, dio_stats])
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    # Print comparison table
    print("\n" + "=" * 70)
    print("  RESULTS: Head-to-Head Comparison")
    print("=" * 70)
    print(f"{'Metric':<25} {'vLLM (RR)':<20} {'DIO+vLLM':<20} {'Improvement':<15}")
    print("-" * 70)

    metrics = [
        ("p50 Latency (ms)", "p50_latency_ms"),
        ("p95 Latency (ms)", "p95_latency_ms"),
        ("p99 Latency (ms)", "p99_latency_ms"),
        ("Mean Latency (ms)", "mean_latency_ms"),
        ("p99 TTFT (ms)", "p99_ttft_ms"),
        ("SLO Attainment (%)", "slo_attainment"),
        ("Failure Rate (%)", "failure_rate"),
        ("Throughput (req/s)", "throughput_rps"),
    ]

    for label, key in metrics:
        rr_val = rr_stats.get(key, 0)
        dio_val = dio_stats.get(key, 0)

        if key in ("slo_attainment",):
            rr_str = f"{rr_val * 100:.1f}%"
            dio_str = f"{dio_val * 100:.1f}%"
            if rr_val > 0:
                improvement = f"+{((dio_val - rr_val) / rr_val) * 100:.0f}%"
            else:
                improvement = "N/A"
        elif key in ("failure_rate",):
            rr_str = f"{rr_val * 100:.1f}%"
            dio_str = f"{dio_val * 100:.1f}%"
            improvement = f"{((rr_val - dio_val) / max(rr_val, 0.001)) * 100:.0f}% less"
        elif key in ("throughput_rps",):
            rr_str = f"{rr_val:.1f}"
            dio_str = f"{dio_val:.1f}"
            if rr_val > 0:
                improvement = f"+{((dio_val - rr_val) / rr_val) * 100:.0f}%"
            else:
                improvement = "N/A"
        else:
            rr_str = f"{rr_val:.1f}"
            dio_str = f"{dio_val:.1f}"
            if rr_val > 0:
                reduction = ((rr_val - dio_val) / rr_val) * 100
                improvement = f"-{reduction:.0f}%"
            else:
                improvement = "N/A"

        print(f"{label:<25} {rr_str:<20} {dio_str:<20} {improvement:<15}")

    print("=" * 70)
    print(f"\n📁 Results saved to: {out_dir}")
    print(f"   Raw data:  {out_dir / 'raw_results.csv'}")
    print(f"   Summary:   {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
