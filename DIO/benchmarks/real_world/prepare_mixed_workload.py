import json
import os
import random

# Paths to your existing datasets
SHAREGPT_PATH = "benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json"
CODE_PATH = "benchmarks/code_data.json"
ARXIV_PATH = "benchmarks/arxiv_data.json"

OUTPUT_FILE = "benchmarks/real_world/mixed_workload.jsonl"

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

def prepare_workload():
    workload = []
    
    # 1. ShareGPT (General Chat -> Small/Large based on length)
    print("🔄 Processing ShareGPT...")
    sharegpt_data = load_json(SHAREGPT_PATH)
    for item in sharegpt_data[:1000]: # Sample 1000
        if len(item.get("conversations", [])) < 2: continue
        prompt = item["conversations"][0]["value"]
        # Heuristic: Short prompts -> Small Tier, Long -> Large Tier
        tier = "small" if len(prompt.split()) < 150 else "large"
        workload.append({
            "id": f"sharegpt_{item.get('id', 'unknown')}",
            "prompt": prompt[:3000], # Truncate to fit context
            "output_len": 150,
            "tier": tier,
            "source": "sharegpt"
        })

    # 2. Code Data (Coding -> Large Tier)
    print("🔄 Processing Code Data...")
    code_data = load_json(CODE_PATH)
    for i, item in enumerate(code_data[:500]):
        # Use docstring as prompt to generate code
        prompt = f"Write Python code for the following: {item.get('docstring', '')}"
        workload.append({
            "id": f"code_{i}",
            "prompt": prompt,
            "output_len": 256,
            "tier": "large", 
            "source": "code"
        })

    # 3. Arxiv Data (Summarization -> Large Tier)
    print("🔄 Processing Arxiv Data...")
    arxiv_data = load_json(ARXIV_PATH)
    for i, item in enumerate(arxiv_data[:500]):
        # Use abstract as input for summarization
        abstract = item.get('abstract', item.get('summary', ''))
        if not abstract: continue
        prompt = f"Summarize this scientific paper:\n\n{abstract}"
        workload.append({
            "id": f"arxiv_{i}",
            "prompt": prompt[:4000],
            "output_len": 200,
            "tier": "large",
            "source": "arxiv"
        })

    # Shuffle to simulate random arrival
    random.shuffle(workload)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for item in workload:
            f.write(json.dumps(item) + "\n")
    print(f"✅ Saved {len(workload)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_workload()