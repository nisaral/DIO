import subprocess
import time
import os
import sys
import urllib.request
import json

# Ensure we are running from the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if os.getcwd() != project_root:
    os.chdir(project_root)

# --- Configuration ---
ITERATIONS = 10
HOST = "http://localhost:8080"
RESULTS_DIR = os.path.join("benchmarks", "real_world", "results")
LOCUST_FILE = os.path.join("benchmarks", "real_world", "locustfile.py")

# Define the rigorous test suite
TEST_SUITE = [
    {
        "id": "A1_Baseline_Small",
        "desc": "Baseline: 4050 Only (Small Tier)",
        "users": 20, "spawn_rate": 5, "duration": "30s",
        "env": {"SCHEDULER_STRATEGY": "RoundRobin"},
        "docker_scale": {"dio-worker": 1, "dio-worker-slow": 0}, # Assume dio-worker is 4050
        "mode": "ROUTING"
    },
    {
        "id": "B1_DIO_Routing",
        "desc": "DIO-NLMS: Intent-Aware Routing (4050 + T4)",
        "users": 50, "spawn_rate": 10, "duration": "45s",
        "env": {"SCHEDULER_STRATEGY": "NLMS"},
        "docker_scale": {"dio-worker": 1, "dio-worker-slow": 1}, # Simulate T4 as 'slow' or remote
        "mode": "ROUTING"
    },
    {
        "id": "B4_Sudden_Spike",
        "desc": "Stress Test: 50 Req/s Spike",
        "users": 100, "spawn_rate": 50, "duration": "30s",
        "env": {"SCHEDULER_STRATEGY": "NLMS"},
        "docker_scale": {"dio-worker": 2}, # Scale up for spike test (Laptop safe)
        "mode": "ROUTING"
    },
    {
        "id": "B5_Agent_Chain",
        "desc": "Multi-Step Agent Workflows",
        "users": 10, "spawn_rate": 1, "duration": "60s",
        "env": {"SCHEDULER_STRATEGY": "NLMS"},
        "docker_scale": {"dio-worker": 2},
        "mode": "AGENT_CHAIN"
    }
]

def run_command(cmd, env=None, check=True):
    # Merge current env with passed env
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    # Removed output suppression to allow debugging Docker errors
    subprocess.run(cmd, shell=True, check=check, env=full_env)

def wait_for_workers(expected_count, timeout=120):
    print(f"   ⏳ Waiting for {expected_count} workers to register (timeout={timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(f"{HOST}/debug/workers") as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    workers = data.get("workers", [])
                    if len(workers) >= expected_count:
                        print(f"      ✅ {len(workers)} workers ready: {workers}")
                        return True
        except Exception:
            pass
        time.sleep(2)
    
    print("      ❌ Timeout waiting for workers to register.")
    print("      --- Manager Logs ---")
    subprocess.run("docker logs dio-dio-manager-1", shell=True)
    print("      --- Worker Logs ---")
    subprocess.run("docker logs dio-dio-worker-1", shell=True)
    return False

def run_test_config(config):
    print(f"\n🧪 Starting Configuration: {config['id']} ({config['desc']})")
    
    # 1. Setup Infrastructure (Docker)
    print("   Creating infrastructure...")
    run_command("docker-compose down", check=False)
    
    scale_args = ""
    for service, count in config["docker_scale"].items():
        scale_args += f"--scale {service}={count} "
    
    # Start Docker with specific env vars for the Manager. Added --build to ensure fresh images.
    cmd = f"docker-compose up -d --build {scale_args}"
    try:
        run_command(cmd, env=config["env"])
    except subprocess.CalledProcessError:
        print("   ❌ Failed to start infrastructure. See logs above.")
        return
    
    # Wait for workers to be ready
    expected_workers = sum(config["docker_scale"].values())
    if not wait_for_workers(expected_workers):
        print("   ❌ Skipping test due to infrastructure failure.")
        return
    
    print("   ⏳ Warming up (5s)...")
    time.sleep(5)

    # 2. Run Iterations
    for i in range(1, ITERATIONS + 1):
        log_file = os.path.join(RESULTS_DIR, f"{config['id']}_iter{i}.csv")
        print(f"   ▶ Iteration {i}/{ITERATIONS} -> {log_file}")
        
        # Locust Environment Variables
        locust_env = os.environ.copy()
        locust_env["LOG_FILE"] = log_file
        locust_env["TEST_MODE"] = config["mode"]
        
        cmd = (
            f"locust -f {LOCUST_FILE} "
            f"--headless "
            f"--users {config['users']} "
            f"--spawn-rate {config['spawn_rate']} "
            f"--run-time {config['duration']} "
            f"--host {HOST}"
        )
        
        try:
            run_command(cmd, env=locust_env)
        except subprocess.CalledProcessError:
            print(f"      ❌ Iteration {i} failed.")

    print(f"   ✅ Configuration {config['id']} complete.")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("🚀 Starting DIO Rigorous Evaluation Suite")
    print(f"   CWD: {os.getcwd()}")
    print(f"   Locust File: {LOCUST_FILE} (Exists: {os.path.exists(LOCUST_FILE)})")
    print(f"   Target: {ITERATIONS} iterations per config")
    
    for config in TEST_SUITE:
        run_test_config(config)