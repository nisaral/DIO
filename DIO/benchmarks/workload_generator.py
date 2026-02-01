import random
import os
import json

class WorkloadGenerator:
    def __init__(self):
        self.sharegpt_prompts = self._load_sharegpt()
        self.arxiv_trace = self._load_trace("benchmarks/traces/workload_arxiv_long.jsonl")
        self.code_trace = self._load_trace("benchmarks/traces/workload_code_gen.jsonl")
        
    def _load_sharegpt(self):
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ShareGPT_V3_unfiltered_cleaned_split.json")
        prompts = []
        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if 'conversations' in item:
                            for msg in item['conversations']:
                                if msg['from'] == 'human':
                                    prompts.append(msg['value'])
                                    break
            except Exception:
                pass
        
        if not prompts:
            # Fallback synthetic
            prompts = ["Explain quantum computing."] * 100
        return prompts

    def _load_trace(self, filepath):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filepath)
        data = []
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line))
                return data
            except Exception: pass
        return []

    def get_prompt(self, workload_name):
        """Returns (prompt_text, expected_output_tokens)"""
        if workload_name == "ShareGPT":
            return self.sharegpt()
        elif workload_name == "FlowGPT":
            return self.flowgpt()
        elif workload_name == "ArXiv":
            return self.arxiv()
        elif workload_name == "Code":
            return self.code()
        elif workload_name == "Enterprise":
            return self.enterprise()
        else:
            return self.sharegpt()

    def sharegpt(self):
        # Real-world trace: High variance
        return random.choice(self.sharegpt_prompts), 128

    def flowgpt(self):
        # Chat trace: Short inputs, short/medium outputs, bursty nature handled by Locust
        # Input: 10-50 tokens, Output: 50-200 tokens
        prompt = "Chat " * random.randint(10, 50)
        return prompt, random.randint(50, 200)

    def arxiv(self):
        # Real ArXiv Trace (Deterministic Replay)
        if self.arxiv_trace:
            item = random.choice(self.arxiv_trace)
            return f"Summarize this paper:\n\n{item['prompt']}", item['output_len']
            
        # Fallback Synthetic
        prompt = "Summarize this paper: " + ("Abstract " * random.randint(1000, 3000))
        return prompt, random.randint(100, 200)

    def code(self):
        # Real Code Trace (Deterministic Replay)
        if self.code_trace:
            item = random.choice(self.code_trace)
            return f"Write a Python function that {item['prompt']}", item['output_len']

        # Fallback Synthetic
        prompt = "Write a python function to " + ("do x " * random.randint(50, 150))
        return prompt, random.randint(500, 1000)

    def enterprise(self):
        # Enterprise: Email drafting, Q&A. Balanced.
        # Input: 100-500 tokens, Output: 100-500 tokens
        prompt = "Draft an email about " + ("business " * random.randint(50, 250))
        return prompt, random.randint(100, 500)

    def get_slo(self, workload_name):
        """Returns (TTFT_SLO_sec, TPOT_SLO_ms)"""
        slos = {
            "ShareGPT": (0.5, 50),
            "FlowGPT": (0.2, 40),   # Chat needs fast TTFT
            "ArXiv": (2.0, 100),    # Long prefill allows slower TTFT
            "Code": (1.0, 30),      # Coding needs fast generation (TPOT)
            "Enterprise": (0.5, 50)
        }
        return slos.get(workload_name, (0.5, 50))