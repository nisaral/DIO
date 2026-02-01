import requests
import json
import os
import time

def download_hf_slice(dataset, config, split, output_filename, target_length=20000, field_map=None):
    """
    Downloads a small slice of a HuggingFace dataset using the Datasets Server API.
    This avoids downloading terabytes of data for a benchmark.
    """
    print(f"⬇️  Downloading target {target_length} rows from {dataset} ({config})...")
    
    rows = []
    offset = 0
    batch_size = 100 # Max allowed by HF API
    
    while len(rows) < target_length:
        remaining = target_length - len(rows)
        fetch_size = min(batch_size, remaining)
        
        url = f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={fetch_size}"
        
        success = False
        for attempt in range(5):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    success = True
                    break
                elif resp.status_code == 429: # Rate limit
                    time.sleep(5 * (attempt + 1))
                elif resp.status_code >= 500: # Server error
                    time.sleep(2)
                else:
                    print(f"\n❌ Failed batch at offset {offset}: {resp.status_code} {resp.text}")
                    break
            except Exception as e:
                print(f"\n⚠️ Connection error: {e}. Retrying ({attempt+1}/5)...")
                time.sleep(2)
        
        if not success:
            break
            
        data = resp.json()
        batch_rows = [item['row'] for item in data['rows']]
        
        if not batch_rows:
            break
        
        # Apply field mapping if provided
        if field_map:
            remapped_rows = []
            for row in batch_rows:
                new_row = row.copy()
                for src, dst in field_map.items():
                    if src in row:
                        new_row[dst] = row[src]
                remapped_rows.append(new_row)
            batch_rows = remapped_rows
        
        rows.extend(batch_rows)
        offset += len(batch_rows)
        print(f"  Fetched {len(rows)}/{target_length}...", end='\r')

    if rows:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2)
        print(f"\n✅ Saved {len(rows)} rows to {output_filename}")

if __name__ == "__main__":
    print("🚀 Starting Dataset Download for DIO Benchmarks...\n")
    
    # 1. ArXiv Summarization (Real Long-Context Data)
    # Dataset: scientific_papers (arxiv)
    # Use: Tests prefill performance and memory pressure
    # Switched to ccdv/arxiv-summarization due to scientific_papers rename/issues
    download_hf_slice("ccdv/arxiv-summarization", "document", "train", "arxiv_data.json")
    
    # 2. Azure/CodeSearchNet (Real Code Data)
    # Dataset: code_search_net (python)
    # Use: Tests decoding performance (long output generation)
    # Switched to google/code_x_glue_ct_code_to_text due to code_search_net rename/issues
    download_hf_slice("google/code_x_glue_ct_code_to_text", "python", "train", "code_data.json",
                      field_map={"docstring": "func_documentation_string", "code": "func_code_string"})
    
    print("\n✨ Done! You can now run the benchmarks with real data.")