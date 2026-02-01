import subprocess
import time
import os
import requests
import pandas as pd

# Ensure we are running from the project root so docker-compose and locust paths work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.getcwd() != project_root:
    os.chdir(project_root)

# Configuration
RESULTS_DIR = "results"
LOCUST_FILE = "benchmarks/locustfile.py"
HOST = "http://localhost:8080"

# The 5 Core "Gap Closure" Tests
TESTS = [
    {
        "id": "T1",
        "name": "SJF_vs_RR_HoL",
        "description": "Proves Head-of-Line Blocking fix. Target: p99 < 1800ms",
        "users": 50,
        "spawn_rate": 10,
        "duration": "2m",
        "env": {"THROTTLE_FACTOR": "1.0"} # Normal baseline
    },
    {
        "id": "T2",
        "name": "NLMS_Adaptation",
        "description": "Proves Online Learning. One worker is 3x slower.",
        "users": 100,
        "spawn_rate": 20,
        "duration": "3m",
        "env": {"THROTTLE_FACTOR": "MIXED"} # Special flag for docker-compose
    },
    {
        "id": "T3",
        "name": "Cold_Start",
        "description": "Proves Zero-Data Init. New worker joins mid-test.",
        "users": 20,
        "spawn_rate": 5,
        "duration": "1m",
        "action": "scale_up" 
    },
    {
        "id": "T4",
        "name": "Hetero_Roofline",
        "description": "Proves VRAM awareness. Simulates A100 vs T4.",
        "users": 100,
        "spawn_rate": 10,
        "duration": "3m",
        "env": {"ROOFLINE_TEST": "TRUE"}
    },
    {
        "id": "T5",
        "name": "Queue_Awareness",
        "description": "Proves Little's Law adherence during spikes.",
        "users": 200,
        "spawn_rate": 50, # Sudden spike
        "duration": "2m",
        "env": {"THROTTLE_FACTOR": "1.0"}
    }
]

def setup_env(test_config):
    print(f"🔧 Setting up environment for {test_config['name']}...")
    subprocess.run(["docker-compose", "down"], stdout=subprocess.DEVNULL)
    
    # FIX 1: Explicitly pass env vars for Throttle
    env = os.environ.copy()
    if test_config.get("env"):
        env.update(test_config["env"])

    # FIX 2: Correct Service Name and handle MIXED throttle
    if test_config.get("env", {}).get("THROTTLE_FACTOR") == "MIXED":
        print("⚠️ Starting Heterogeneous Cluster (2 Fast, 1 Slow)...")
        cmd = ["docker-compose", "up", "-d", "--scale", "dio-worker=2", "--scale", "dio-worker-slow=1"]
        subprocess.run(cmd, env=env)
    else:
        cmd = ["docker-compose", "up", "-d", "--scale", "dio-worker=3"]
        subprocess.run(cmd, env=env)
    
    print("⏳ Waiting 30s for Python Workers to boot...")
    time.sleep(30)

def run_action(action):
    if action == "scale_up":
        print("🚀 ACTION: Scaling up workers mid-test...")
        subprocess.Popen(["docker-compose", "up", "-d", "--scale", "dio-worker=6"])

def run_test(test):
    print(f"\n==================================================")
    print(f"🧪 RUNNING {test['id']}: {test['name']}")
    print(f"📝 Goal: {test['description']}")
    print(f"==================================================")
    
    setup_env(test)
    
    csv_path = os.path.join(RESULTS_DIR, test['name'])
    
    # Start Locust
    cmd = [
        "locust", "-f", LOCUST_FILE,
        "--headless",
        f"--users={test['users']}",
        f"--spawn-rate={test['spawn_rate']}",
        f"--run-time={test['duration']}",
        f"--host={HOST}",
        "--csv", csv_path
    ]
    
    # If there's a mid-test action, run it in parallel
    if "action" in test:
        proc = subprocess.Popen(cmd)
        time.sleep(20) # Wait 20s then trigger action
        run_action(test['action'])
        proc.wait()
    else:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("❌ Test Failed! Fetching Manager Logs...")
            subprocess.run(["docker", "logs", "dio-dio-manager-1"]) # Debug Info

    print(f"✅ {test['id']} Complete. Results saved to {csv_path}_stats.csv")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for t in TESTS:
        run_test(t)