import argparse
import time
import requests
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Config ---
MANAGER_URL = "http://localhost:8080" # Localhost is safe inside the Studio

def load_model(mock=False):
    if mock:
        return None, None
    
    print("   🚀 Loading Model (TinyLlama)...")
    try:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto" # Uses GPU if available
        )
        return tokenizer, model
    except Exception as e:
        print(f"   ⚠️ GPU Load Failed ({e}). Falling back to MOCK.")
        return None, None

def generate(tokenizer, model, prompt, output_len, latency_mult=1.0):
    start = time.perf_counter()
    
    if model:
        # REAL INFERENCE
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=output_len)
    else:
        # MOCK INFERENCE (approx 20ms/token for CPU sim)
        time.sleep(output_len * 0.02)

    raw_latency = (time.perf_counter() - start) * 1000
    
    # Simulate Heterogeneity (e.g. T4 is 2x slower than A100)
    final_latency = raw_latency * latency_mult
    
    # Sleep the difference if we are simulating a slower card
    if final_latency > raw_latency:
        time.sleep((final_latency - raw_latency) / 1000)
        
    return final_latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--tier", default="large")
    parser.add_argument("--vram", type=int, default=24000)
    parser.add_argument("--latency-mult", type=float, default=1.0)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    tokenizer, model = load_model(args.mock)
    
    print(f"✅ Worker {args.worker_id} Ready. Polling Manager...")

    while True:
        try:
            # 1. Register / Heartbeat (Every loop ensures we stay alive)
            requests.post(f"{MANAGER_URL}/register", json={
                "id": args.worker_id,
                "type": "gpu",
                "tier": args.tier,
                "vram_total": args.vram
            })

            # 2. Get Task
            resp = requests.get(f"{MANAGER_URL}/get_task?worker_id={args.worker_id}")
            
            if resp.status_code == 200:
                task = resp.json()
                # print(f"   ⚡ Processing {task['id']}...")
                
                # 3. Run Inference
                prompt = task.get("prompt", "Hello world")
                out_len = task.get("output_len", 50)
                
                lat = generate(tokenizer, model, prompt, out_len, args.latency_mult)
                
                # 4. Report
                requests.post(f"{MANAGER_URL}/report", json={
                    "task_id": task['id'],
                    "worker_id": args.worker_id,
                    "latency": lat,
                    "tokens": out_len
                })
            else:
                time.sleep(0.01) # Short sleep to prevent CPU spin

        except Exception as e:
            # print(f"   ⚠️ Connection Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()