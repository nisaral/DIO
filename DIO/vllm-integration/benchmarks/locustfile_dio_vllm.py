"""
DIO v3 — Locust Load Test for DIO+vLLM

This Locust file sends requests to DIO's HTTP gateway,
which routes them to vLLM workers via the NLMS scheduler.

Usage:
  # Headless mode (for benchmarks):
  locust -f locustfile_dio_vllm.py --headless \
    --users 50 --spawn-rate 10 --run-time 2m \
    --host http://localhost:8085 \
    --csv results/dio_vllm

  # Web UI mode (interactive):
  locust -f locustfile_dio_vllm.py --host http://localhost:8085
  # Open http://localhost:8089
"""

import json
import os
import random

from locust import HttpUser, task, between, events

# ── Load Dataset ──

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Try loading ShareGPT
DATASET_PATHS = [
    os.path.join(DIO_ROOT, "benchmarks", "ShareGPT_V3_unfiltered_cleaned_split.json"),
    os.path.join(DIO_ROOT, "benchmarks", "data", "sharegpt.jsonl"),
]

prompts = []
for path in DATASET_PATHS:
    if os.path.exists(path):
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for item in raw:
                    if "conversations" in item:
                        for turn in item["conversations"]:
                            if turn["from"] == "human":
                                prompts.append(turn["value"])
                                break
            elif path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if "prompt" in data:
                            prompts.append(data["prompt"])
            if prompts:
                print(f"✅ Loaded {len(prompts)} prompts from {os.path.basename(path)}")
                break
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}")

if not prompts:
    print("ℹ️ Using synthetic prompts (no dataset found)")
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to implement quicksort.",
        "What are the main differences between TCP and UDP?",
        "Summarize the plot of Shakespeare's Hamlet.",
        "How does a neural network learn?",
        "Explain quantum computing to a 10-year-old.",
        "Write a short story about a robot discovering emotions.",
        "What is the significance of the Turing test?",
        "Compare supervised and unsupervised machine learning.",
        "Describe the architecture of a modern web application.",
    ] + [
        # Long prompts for VRAM stress testing
        "Write a comprehensive essay on the history of AI. " * 20,
        "Provide a detailed analysis of distributed systems. " * 20,
        "Explain the evolution of programming languages. " * 20,
    ]

MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")


class DIOVLLMUser(HttpUser):
    """Simulates a user sending requests through DIO to vLLM."""

    wait_time = between(0.1, 1.0)

    @task(8)
    def generate_small(self):
        """Standard short prompt (majority of traffic)."""
        prompt = random.choice(prompts[:10]) if len(prompts) > 10 else random.choice(prompts)

        payload = {
            "prompt": prompt[:500],  # Cap at ~125 tokens
            "model_id": MODEL_ID,
            "tier": "small",
        }

        with self.client.post(
            "/api/generate",
            json=payload,
            catch_response=True,
            name="/api/generate [small]",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    latency = data.get("latency_ms", 0)
                    tokens = data.get("tokens_used", 0)
                    if latency > 10000:  # 10s = very slow
                        response.failure(f"Slow: {latency:.0f}ms")
                    else:
                        response.success()
                except Exception:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)
    def generate_large(self):
        """Long prompt (VRAM stress test)."""
        if len(prompts) > 10:
            prompt = random.choice(prompts[10:])
        else:
            prompt = random.choice(prompts) * 5

        payload = {
            "prompt": prompt[:4000],  # Cap at ~1000 tokens
            "model_id": MODEL_ID,
            "tier": "large",
        }

        with self.client.post(
            "/api/generate",
            json=payload,
            catch_response=True,
            name="/api/generate [large]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def health_check(self):
        """Verify DIO Manager is healthy."""
        with self.client.get(
            "/api/test",
            catch_response=True,
            name="/api/test [health]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Manager unhealthy: {response.status_code}")
