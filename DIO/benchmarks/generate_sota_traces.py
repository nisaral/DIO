import json
import numpy as np
import os

# Config
RAW_ARXIV = "benchmarks/arxiv_data.json"
RAW_CODE = "benchmarks/code_data.json"
TRACE_DIR = "benchmarks/traces"
os.makedirs(TRACE_DIR, exist_ok=True)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_trace(data, filename, task_type):
    output_path = os.path.join(TRACE_DIR, filename)
    print(f"Generating {output_path} ({len(data)} reqs)...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        arrival_time = 0.0
        for i, item in enumerate(data):
            # Bursty Arrivals (Poisson Process)
            inter_arrival = np.random.exponential(scale=1.0)
            arrival_time += inter_arrival
            
            # Logic to handle the specific raw format from your download script
            if task_type == "arxiv":
                prompt = item.get('article', '')[:12000] # Truncate for safety
                output_len = len(item.get('abstract', '').split())
                prompt_len = len(prompt) // 4
            elif task_type == "code":
                prompt = item.get('func_documentation_string', 'Write code')
                code = item.get('func_code_string', '')
                output_len = len(code.split())
                prompt_len = len(prompt) // 4
            
            req = {
                "request_id": f"{task_type}_{i}",
                "timestamp": arrival_time,
                "prompt": prompt,
                "prompt_len": prompt_len,
                "output_len": max(50, output_len),
                "task": task_type
            }
            f.write(json.dumps(req) + "\n")

# Execute
print("🔄 Processing raw datasets into SOTA traces...")

# 1. ArXiv
if os.path.exists(RAW_ARXIV):
    arxiv_data = load_json(RAW_ARXIV)
    save_trace(arxiv_data, "workload_arxiv_long.jsonl", "arxiv")
else:
    print(f"⚠️ {RAW_ARXIV} not found. Run download_datasets.py first!")

# 2. Code
if os.path.exists(RAW_CODE):
    code_data = load_json(RAW_CODE)
    save_trace(code_data, "workload_code_gen.jsonl", "code")
else:
    print(f"⚠️ {RAW_CODE} not found. Run download_datasets.py first!")

print("✅ Trace generation complete.")