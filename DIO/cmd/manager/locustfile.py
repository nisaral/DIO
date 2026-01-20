import json
import random
import os
from locust import HttpUser, task, between, events

# Load ShareGPT dataset if available, otherwise use synthetic
DATASET_PATH = "ShareGPT_V3_unfiltered_cleaned_split.json"
PROMPTS = []

if os.path.exists(DATASET_PATH):
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Extract first human message from each conversation
            for item in data:
                if 'conversations' in item:
                    for msg in item['conversations']:
                        if msg['from'] == 'human':
                            PROMPTS.append(msg['value'])
                            break
            print(f"Loaded {len(PROMPTS)} prompts from ShareGPT.")
    except Exception as e:
        print(f"Error loading dataset: {e}")

if not PROMPTS:
    print("Using synthetic prompts.")
    PROMPTS = [
        "Explain quantum computing in simple terms.",
        "Write a python function to reverse a string.",
        "What is the capital of France?",
        "Summarize the history of the Roman Empire.",
        "Translate 'Hello world' to Spanish."
    ] * 100

class DIOUser(HttpUser):
    wait_time = between(1, 3) # Simulate think time

    @task(3)
    def chat_completion(self):
        prompt = random.choice(PROMPTS)
        # Truncate very long prompts to avoid huge payloads in test
        if len(prompt) > 1000:
            prompt = prompt[:1000]
            
        self.client.post("/api/generate", json={
            "model_id": "gpt-4",
            "prompt": prompt
        })

    @task(1)
    def classification(self):
        # Simulate a lighter/different model task (Hetero Load)
        self.client.post("/api/generate", json={
            "model_id": "bert-base-uncased",
            "prompt": "Classify this text: " + random.choice(PROMPTS)[:100]
        })

    def on_start(self):
        pass