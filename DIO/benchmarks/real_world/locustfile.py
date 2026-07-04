import json
import os
import time
import random
from locust import HttpUser, task, between, events

# --- Research Standards for 3B/7B Models ---
# TTFT < 0.5s (Instant feel)
# TPOT < 50ms (Faster than human reading)
TTFT_SLO_MS = float(os.environ.get("TTFT_SLO_MS", "2000"))  # match paper SLO
TPOT_SLO_MS = 50.0

workload_data = []

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    global workload_data
    raw_path = os.environ.get("WORKLOAD_FILE")
    
    if not raw_path:
        # Fallback for T7 scalability test (no dataset needed)
        workload_data = [{"prompt": "ignore", "tier": "small", "out": 10}]
        return

    workload_path = os.path.abspath(raw_path)
    print(f"📂 [LOCUST] Loading: {workload_path}")
    
    workload_data = []
    if os.path.exists(workload_path):
        with open(workload_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        # Normalize ShareGPT vs Arxiv formats
                        prompt = d.get("prompt")
                        if not prompt and "conversations" in d:
                            prompt = d["conversations"][0]["value"]
                        
                        if prompt:
                            workload_data.append({
                                "prompt": prompt, 
                                "tier": d.get("tier", "large"),
                                "output_len": d.get("output_len", 128)
                            })
                    except: pass
    print(f"✅ [LOCUST] Loaded {len(workload_data)} prompts.")

class DIOResearchUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def execute(self):
        if not workload_data: return
        item = random.choice(workload_data)
        
        payload = {
            "model_id": os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct"),
            "prompt": item["prompt"],
            "tier": item["tier"]
        }
        
        start = time.time()
        # "catch_response=True" allows us to mark requests as failures manually
        with self.client.post("/api/generate", json=payload, catch_response=True) as resp:
            lat = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    ttft = data.get("ttft_ms", 0)
                    tokens = data.get("tokens_used", 1)
                    
                    # Calculate Inter-Token Latency (TPOT)
                    # (Total Latency - Time To First Token) / (Tokens Generated - 1)
                    tpot = (lat - ttft) / max(1, tokens - 1)
                    
                    # --- SLO Logic ---
                    met_slo = (ttft <= TTFT_SLO_MS) and (tpot <= TPOT_SLO_MS)
                    
                    # Fire event for CSV logging
                    events.request.fire(
                        request_type="GEN",
                        name="SLO_Met" if met_slo else "SLO_Missed", # Splitting this helps charts later
                        response_time=lat,
                        response_length=tokens,
                        exception=None
                    )
                    resp.success()
                except Exception as e:
                    resp.failure(f"Bad JSON: {e}")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text}")