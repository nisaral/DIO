import json
import os
import random

# Paths to your existing datasets
SHAREGPT_PATH = "benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json"
CODE_PATH = "benchmarks/code_data.json"
ARXIV_PATH = "benchmarks/arxiv_data.json"

DATA_DIR = "benchmarks/data"

def load_json(path):
    if not os.path.exists(path):
        print(f"⚠️ Warning: {path} not found. Skipping.")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return []

def save_jsonl(data, filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"✅ Saved {len(data)} items to {filepath}")

def prepare_workload():
    # 1. ShareGPT (General Chat -> Small/Large based on length)
    print("🔄 Processing ShareGPT...")
    sharegpt_data = load_json(SHAREGPT_PATH)
    sharegpt_workload = []
    for item in sharegpt_data[:1000]: # Sample 1000
        if len(item.get("conversations", [])) < 2: continue
        prompt = item["conversations"][0]["value"]
        tier = "small" if len(prompt.split()) < 150 else "large"
        sharegpt_workload.append({
            "id": f"sharegpt_{item.get('id', 'unknown')}",
            "prompt": prompt[:3000],
            "output_len": 150,
            "tier": tier,
            "source": "sharegpt"
        })
    save_jsonl(sharegpt_workload, "sharegpt.jsonl")

    # 2. Code Data (Coding -> Large Tier)
    print("🔄 Processing Code Data...")
    code_data = load_json(CODE_PATH)
    code_workload = []
    for i, item in enumerate(code_data[:500]):
        prompt = f"Write Python code for the following: {item.get('docstring', '')}"
        code_workload.append({
            "id": f"code_{i}",
            "prompt": prompt,
            "output_len": 256,
            "tier": "large", 
            "source": "code"
        })
    save_jsonl(code_workload, "azure_code.jsonl")

    # 3. Arxiv Data (Summarization -> Large Tier)
    print("🔄 Processing Arxiv Data...")
    arxiv_data = load_json(ARXIV_PATH)
    arxiv_workload = []
    for i, item in enumerate(arxiv_data[:500]):
        abstract = item.get('abstract', item.get('summary', ''))
        if not abstract: continue
        prompt = f"Summarize this scientific paper:\n\n{abstract}"
        arxiv_workload.append({
            "id": f"arxiv_{i}",
            "prompt": prompt[:4000],
            "output_len": 200,
            "tier": "large",
            "source": "arxiv"
        })
    save_jsonl(arxiv_workload, "arxiv.jsonl")

if __name__ == "__main__":
    prepare_workload()