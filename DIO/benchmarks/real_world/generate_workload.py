import json
import random

OUTPUT_FILE = "benchmarks/agent_workload.jsonl"

def generate():
    data = []
    
    # 1. Single Requests (Routing Test)
    for i in range(100):
        data.append({"id": f"chat_{i}", "prompt": "Hello " * 5, "output_len": 50, "tier": "small"})
        data.append({"id": f"reason_{i}", "prompt": "Analyze " * 50, "output_len": 500, "tier": "large"})

    # 2. Agent Chains (Multi-step)
    for i in range(50):
        agent_id = f"agent_{i}"
        # Step 1: Tool Use (Small)
        data.append({"id": agent_id, "step": 1, "prompt": "Search for weather", "output_len": 20, "tier": "small"})
        # Step 2: Reasoning (Large)
        data.append({"id": agent_id, "step": 2, "prompt": "Based on weather, plan trip...", "output_len": 800, "tier": "large"})
        # Step 3: Summary (Small)
        data.append({"id": agent_id, "step": 3, "prompt": "Summarize plan", "output_len": 100, "tier": "small"})

    with open(OUTPUT_FILE, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Generated {len(data)} items in {OUTPUT_FILE}")

if __name__ == "__main__":
    generate()