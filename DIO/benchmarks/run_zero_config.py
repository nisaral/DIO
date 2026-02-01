import time
import pandas as pd
import requests
import os
import json

# Config
MANAGER_URL = "http://localhost:8080"
# Note: You might need to adjust this ID based on what shows up in your logs/dashboard
# Usually it's the hostname of the container if not explicitly set.
# We will now fetch it dynamically.

def run_convergence_test():
    print("🚀 Starting Zero-Config Convergence Test...")
    
    # 0. Fetch Active Worker ID
    worker_id = "dio-worker-1" # Fallback
    try:
        resp = requests.get(f"{MANAGER_URL}/debug/workers")
        if resp.status_code == 200:
            try:
                workers = resp.json().get("workers", [])
                if workers:
                    worker_id = workers[0]
                    print(f"  🔎 Found active worker: {worker_id}")
            except json.JSONDecodeError:
                print(f"  ⚠️ Failed to parse workers JSON. Raw response: {resp.text[:100]}")
        else:
            print(f"  ⚠️ Failed to fetch workers: Status {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️ Error fetching workers: {e}")

    # 1. Reset RLS state (Simulate fresh worker)
    try:
        requests.post(f"{MANAGER_URL}/debug/reset_worker", json={"worker_id": worker_id})
        print(f"  ✅ Reset state for {worker_id}")
    except Exception as e:
        print(f"  ⚠️ Warning: Could not reset worker (Check ID?): {e}")
    
    results = []
    start_time = time.time()
    
    # 2. Blast requests for 60 seconds
    # We use a fixed prompt length to test convergence on a specific workload type
    prompt = "Test prompt " * 10 # Approx 20-30 tokens
    tokens_est = 50 # Estimate for debug endpoint

    print("  ⏳ Running probe requests...")
    for i in range(50):
        req_start = time.time()
        
        # Send a "Probe" request
        try:
            resp = requests.post(f"{MANAGER_URL}/api/generate", json={"model_id": "gpt-4", "prompt": prompt})
            latency = resp.json().get('latency_ms', 0)
            
            # Get DIO's internal prediction (Telemetry)
            pred_resp = requests.get(f"{MANAGER_URL}/debug/prediction?worker={worker_id}&tokens={tokens_est}")
            predicted = pred_resp.json().get('predicted_ms', 0)
            
            error = abs(predicted - latency)
            elapsed = time.time() - start_time
            
            results.append({
                "time_sec": elapsed,
                "actual": latency,
                "predicted": predicted,
                "error": error
            })
            
            print(f"    T={elapsed:.1f}s | Act: {latency:.0f}ms | Pred: {predicted:.0f}ms | Err: {error:.0f}ms")
        except Exception as e:
            print(f"    ❌ Request failed: {e}")
        
        time.sleep(0.5) # High QPS
        
    # 3. Save for Plotting
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/convergence_data.csv", index=False)
    print("✅ Convergence data saved. Plot 'error' vs 'time_sec'.")

if __name__ == "__main__":
    run_convergence_test()