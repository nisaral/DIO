import json
import os
import time
import csv
import uuid
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner

# --- Configuration ---
WORKLOAD_FILE = os.environ.get("WORKLOAD_FILE", "benchmarks/agent_workload.jsonl")
LOG_FILE = os.environ.get("LOG_FILE", "benchmarks/real_world/results/raw_logs.csv")
MODE = os.environ.get("TEST_MODE", "ROUTING") # "ROUTING" or "AGENT_CHAIN"

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# --- Load Workload ---
workload_data = []
if os.path.exists(WORKLOAD_FILE):
    with open(WORKLOAD_FILE, 'r') as f:
        for line in f:
            if line.strip():
                workload_data.append(json.loads(line))
else:
    # Fallback data if file missing
    workload_data = [
        {"id": "chat_1", "prompt": "Hello", "output_len": 50, "tier": "small"},
        {"id": "reason_1", "prompt": "Analyze this", "output_len": 500, "tier": "large"}
    ]

# --- CSV Logger Setup ---
# We write headers only once
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "request_id", "type", "tier", "latency_ms", "ttft_ms", "tokens_used", "worker_id", "status"])

def log_request(request_id, req_type, tier, latency, ttft, tokens, worker, status):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), request_id, req_type, tier, latency, ttft, tokens, worker, status])

class DIOResearchUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def run_workload(self):
        if MODE == "AGENT_CHAIN":
            self.run_agent_chain()
        else:
            self.run_routing_task()

    def run_routing_task(self):
        """
        Picks a random task and sends it. Used for Baseline & Routing tests.
        """
        import random
        item = random.choice(workload_data)
        self._send_request(item, "SINGLE")

    def run_agent_chain(self):
        """
        Executes a multi-step chain for a specific Agent ID.
        """
        # Group by ID (simple implementation: filter unique IDs then pick one)
        # In a real high-perf scenario, pre-group these in __init__
        unique_ids = list(set(d['id'] for d in workload_data))
        import random
        agent_id = random.choice(unique_ids)
        chain_steps = sorted([d for d in workload_data if d['id'] == agent_id], key=lambda x: x.get('step', 0))

        for step in chain_steps:
            success = self._send_request(step, "CHAIN_STEP")
            if not success:
                break # Break chain on failure

    def _send_request(self, item, req_type):
        payload = {
            "model_id": os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            "prompt": item["prompt"],
            "tier": item.get("tier", "small")
        }
        
        start_time = time.time()
        # We pass tier in payload (Manager logic) AND header (if using Nginx/Gateway later)
        headers = {"X-DIO-Tier": item.get("tier", "small")}
        
        with self.client.post("/api/generate", json=payload, headers=headers, catch_response=True) as response:
            latency = (time.time() - start_time) * 1000
            
            # Extract Metrics
            worker_id = response.headers.get("X-DIO-Worker-ID", "unknown")
            ttft = 0
            tokens = item.get("output_len", 0)
            status = "FAILURE"

            if response.status_code == 200:
                status = "SUCCESS"
                try:
                    data = response.json()
                    ttft = data.get("ttft_ms", 0)
                    if "tokens_used" in data:
                        tokens = data["tokens_used"]
                except:
                    pass
            else:
                response.failure(f"Failed: {response.text}")

            # Log to CSV
            log_request(
                request_id=item["id"],
                req_type=req_type,
                tier=item.get("tier", "small"),
                latency=latency,
                ttft=ttft,
                tokens=tokens,
                worker=worker_id,
                status=status
            )
            
            # Fire Locust Events for UI
            events.request.fire(
                request_type=req_type,
                name=f"{item.get('source', 'general')}_{item.get('tier', 'small')}",
                response_time=latency,
                response_length=tokens,
                exception=None if status == "SUCCESS" else Exception("Failed")
            )

            return status == "SUCCESS"

# Hook to print start message
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"🚀 Starting DIO Research Test | Mode: {MODE} | Log: {LOG_FILE}")