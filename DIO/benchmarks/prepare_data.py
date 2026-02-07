import json
import os
import random
from huggingface_hub import hf_hub_download
import os
import shutil

# Target directory
target_dir = "benchmarks"
os.makedirs(target_dir, exist_ok=True)

datasets = [
    {
        "repo": "openchat/openchat_sharegpt4_dataset",
        "file": "sharegpt_gpt4.json",
        "local": "ShareGPT_V3_unfiltered_cleaned_split.json"
    },
    {
        "repo": "Muennighoff/mbpp",
        "file": "data/mbpp.jsonl",
        "local": "code_data.json"
    },
    {
        "repo": "CShorten/ML-ArXiv-Papers",
        "file": "ml_arxiv_papers.json",
        "local": "arxiv_data.json"
    }
]

for d in datasets:
    print(f"📥 Downloading {d['file']} from {d['repo']}...")
    try:
        path = hf_hub_download(repo_id=d["repo"], filename=d["file"], repo_type="dataset")
        shutil.copy(path, os.path.join(target_dir, d["local"]))
        print(f"✅ Saved to {target_dir}/{d['local']}")
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n🚀 All datasets are ready for processing.")
# Paths to your existing datasets
# Ensure these files are in your benchmarks/ folder
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
    """Saves data in JSONL format, which is better for streaming to Llama-3."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            # ensure_ascii=False is important for non-English chars in Arxiv/ShareGPT
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(data)} items to {filepath}")

def prepare_workload():
    # 1. ShareGPT (General Chat)
    # Strategy: Split into 'small' (<512 tokens) and 'large' (complex reasoning)
    print("🔄 Processing ShareGPT...")
    sharegpt_data = load_json(SHAREGPT_PATH)
    sharegpt_workload = []
    
    # We sample up to 1000 to keep the benchmark run-time under 1 hour
    for item in sharegpt_data[:1000]:
        if "conversations" not in item or len(item["conversations"]) < 1:
            continue
            
        prompt = item["conversations"][0]["value"]
        
        # Llama-3-8B thresholding: 
        # Using 200 words as a proxy for ~250-300 tokens
        tier = "small" if len(prompt.split()) < 200 else "large"
        
        sharegpt_workload.append({
            "id": f"sharegpt_{item.get('id', random.randint(1000, 9999))}",
            "prompt": prompt[:4096], # Llama-3 can handle 8k+, but we cap for speed
            "output_len": 128,       # standard chat response length
            "tier": tier,
            "source": "sharegpt"
        })
    save_jsonl(sharegpt_workload, "sharegpt.jsonl")

    # 2. Code Data (Always 'large' tier due to compute density)
    print("🔄 Processing Code Data...")
    code_data = load_json(CODE_PATH)
    code_workload = []
    for i, item in enumerate(code_data[:500]):
        # We wrap the docstring in an instruction prompt for Llama-3-Instruct
        docstring = item.get('docstring', '')
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWrite high-quality Python code for: {docstring}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        code_workload.append({
            "id": f"code_{i}",
            "prompt": prompt,
            "output_len": 256, # Coding tasks need longer completion
            "tier": "large",
            "source": "code"
        })
    save_jsonl(code_workload, "azure_code.jsonl")

    # 3. Arxiv Data (Always 'large' tier due to context length)
    print("🔄 Processing Arxiv Data...")
    arxiv_data = load_json(ARXIV_PATH)
    arxiv_workload = []
    for i, item in enumerate(arxiv_data[:500]):
        abstract = item.get('abstract', item.get('summary', ''))
        if not abstract: continue
        
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSummarize the following scientific abstract for a general audience:\n\n{abstract}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        arxiv_workload.append({
            "id": f"arxiv_{i}",
            "prompt": prompt[:5000],
            "output_len": 150,
            "tier": "large",
            "source": "arxiv"
        })
    save_jsonl(arxiv_workload, "arxiv.jsonl")

if __name__ == "__main__":
    prepare_workload()