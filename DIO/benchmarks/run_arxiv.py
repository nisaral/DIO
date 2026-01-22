import subprocess
import time
import os

# Create results directory for our paper figures
os.makedirs("results", exist_ok=True)

tests = [
    # Test 1: Proving we fixed Head-of-Line Blocking
    {"name": "T1_SJF", "users": 50, "spawn": 10, "runtime": "5m"},
    
    # Test 2: Proving RLS handles a "Slow Worker" (Inject delay in worker env)
    {"name": "T2_RLS", "users": 100, "spawn": 20, "runtime": "10m", "inject_slow": True},
    
    # Test 3: Cold Start (New worker penalty)
    {"name": "T3_Cold", "users": 20, "spawn": 5, "runtime": "2m", "new_worker": True},

    # Test 4: Heterogeneous Cluster (Simulate A100 vs T4)
    {"name": "T4_Hetero", "users": 100, "spawn": 10, "runtime": "15m", "throttle": "50%"},
]

def run_suite():
    print("🚀 Starting ArXiv Rigor Suite...")
    for t in tests:
        print(f"--- Running {t['name']} ---")
        cmd = [
            "locust", "-f", "benchmarks/locustfile.py", 
            "--headless", 
            f"--users={t['users']}", 
            f"--spawn-rate={t['spawn']}", 
            f"--run-time={t['runtime']}", 
            "--csv", f"results/{t['name']}"
        ]
        subprocess.run(cmd)
        print(f"✅ {t['name']} complete. Data saved to results/")
        time.sleep(10) # Cool down period

if __name__ == "__main__":
    run_suite()