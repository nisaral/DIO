import json
import random
import os
from locust import HttpUser, task, between

# Load ShareGPT Data
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ShareGPT_V3_unfiltered_cleaned_split.json")
dataset = []

if os.path.exists(DATASET_PATH):
    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        for item in raw_data:
            if "conversations" in item:
                for turn in item["conversations"]:
                    if turn["from"] == "human":
                        dataset.append(turn["value"])
                        break
    print(f"Loaded {len(dataset)} prompts.")
else:
    print("Dataset not found. Using synthetic data.")
    dataset = ["Synthetic prompt " * i for i in range(1, 100)]

class VLLMUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task
    def generate_inference(self):
        prompt = random.choice(dataset)
        
        # vLLM expects OpenAI-compatible JSON
        payload = {
            "model": "facebook/opt-125m", # Matches the model we will load
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        # Note the endpoint change: /v1/completions
        with self.client.post("/v1/completions", json=payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Status {response.status_code}: {response.text}")