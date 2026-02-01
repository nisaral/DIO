import json
import random
import os
import time
from locust import HttpUser, task, between, constant_pacing, events
from benchmarks.workload_generator import WorkloadGenerator

generator = WorkloadGenerator()

class DIOUser(HttpUser):
    # Arrival Pattern Logic
    arrival_mode = os.environ.get("ARRIVAL_MODE", "UNIFORM")
    
    if arrival_mode == "BURSTY":
        # Poisson-like arrivals (random wait)
        wait_time = between(0.1, 2.0)
    else:
        # Uniform QPS (Constant Pacing)
        # Default to 1 request every 1 second per user if not specified
        wait_time = constant_pacing(1)

    @task
    def chat_completion(self):
        workload = os.environ.get("WORKLOAD", "ShareGPT")
        prompt, output_len = generator.get_prompt(workload)
        
        # Truncate very long prompts to avoid huge payloads in test
        if len(prompt) > 12000: # Allow larger prompts for ArXiv
            prompt = prompt[:12000]
            
        start_time = time.time()
        with self.client.post("/api/generate", json={
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt": prompt
        }, catch_response=True) as response:
            latency = (time.time() - start_time) * 1000
            
            # Parse Server Metrics (TTFT, Tokens)
            ttft = 0
            tokens = output_len
            try:
                if response.status_code == 200:
                    data = response.json()
                    ttft = data.get("ttft_ms", 0)
                    if "tokens_used" in data:
                        tokens = data["tokens_used"]
            except Exception:
                pass

            # Custom Metric Logging (Simulated TPOT)
            # Since we don't have streaming, we approximate TPOT = Latency / OutputTokens
            # Use server-side tokens if available for accuracy
            tpot = latency / max(1, tokens)
            
            # Fallback: If server didn't report TTFT (e.g. non-streaming), assume TTFT = Latency
            # if ttft <= 0:
            #     ttft = latency
            
            # Fire custom event for aggregation
            events.request.fire(
                request_type="TPOT",
                name=workload,
                response_time=tpot,
                response_length=tokens,
                exception=None,
            )

            # Fire TTFT event (Critical for "Stress Test" defense)
            if ttft > 0:
                events.request.fire(
                    request_type="TTFT",
                    name=workload,
                    response_time=ttft,
                    response_length=0,
                    exception=None,
                )

    def on_start(self):
        pass

class AgentUser(HttpUser):
    """
    Simulates a multi-step agent workload where tasks must be executed in sequence.
    Reads from benchmarks/agent_workload.jsonl
    """
    wait_time = constant_pacing(2) # Slower pacing for complex agents
    workload_data = []

    def on_start(self):
        # Load workload once
        if not AgentUser.workload_data:
            try:
                with open("benchmarks/agent_workload.jsonl", "r") as f:
                    for line in f:
                        AgentUser.workload_data.append(json.loads(line))
            except Exception as e:
                print(f"⚠️ Failed to load agent workload: {e}")

    @task
    def run_agent_chain(self):
        if not AgentUser.workload_data:
            return

        # Pick a random agent chain (grouped by ID in a real scenario, here we just pick one step)
        # For simplicity in this benchmark, we treat each line as a step to be executed
        step = random.choice(AgentUser.workload_data)
        
        start_time = time.time()
        with self.client.post("/api/generate", json={
            "model_id": "agent-model",
            "prompt": step["prompt"],
            "tier": step.get("tier", "small")
        }, catch_response=True) as response:
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # Log Agent Step Metric
                events.request.fire(
                    request_type="AGENT_STEP",
                    name=f"Agent_{step['id']}_{step['tier']}",
                    response_time=latency,
                    response_length=step["output_len"],
                    exception=None,
                )
            else:
                response.failure(f"Agent step failed: {response.text}")