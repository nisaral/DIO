import subprocess
import time
import numpy as np
import pandas as pd
import os
import sys

# Ensure we are running from the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.getcwd() != project_root:
    os.chdir(project_root)

WORKLOADS = ["ShareGPT", "FlowGPT", "ArXiv", "Code", "Enterprise"]
STRATEGIES = ["RLS", "RoundRobin", "LeastLoaded", "LatencyBased"]
ARRIVAL_MODES = ["UNIFORM", "BURSTY"]

def run_test(workload, strategy, arrival_mode, users=50, duration="30s"):
    print(f"\n🧪 TEST: {workload} | {strategy} | {arrival_mode} | {users} Users")
    
    # 1. Setup Environment
    env = os.environ.copy()
    env["SCHEDULER_STRATEGY"] = strategy
    env["WORKLOAD"] = workload
    env["ARRIVAL_MODE"] = arrival_mode
    
    # Restart Docker to clear state
    subprocess.run(["docker-compose", "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Scale based on workload (ArXiv needs more workers due to long processing)
    workers = 8 if workload == "ArXiv" else 4
    cmd = ["docker-compose", "up", "-d", "--scale", f"dio-worker={workers}"]
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("  ⏳ Warming up (15s)...")
    time.sleep(15)
    
    # 2. Run Locust
    os.makedirs("results/comprehensive", exist_ok=True)
    csv_name = f"results/comprehensive/{workload}_{strategy}_{arrival_mode}"
    
    # Spawn rate: 10/s
    cmd = f"locust -f benchmarks/locustfile.py --headless -u {users} -r 10 --run-time {duration} --csv {csv_name} --host http://localhost:8080"
    subprocess.run(cmd.split(), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. Parse Results
    stats = {
        "workload": workload,
        "strategy": strategy,
        "arrival": arrival_mode,
        "users": users,
        "p50": 0, "p90": 0, "p99": 0, "tps": 0, "failures": 0
    }
    
    try:
        df = pd.read_csv(f"{csv_name}_stats.csv")
        if not df.empty:
            # Locust stats
            # Filter for Type="POST" to avoid counting TTFT/TPOT events as requests
            post_reqs = df[df["Type"] == "POST"]
            if not post_reqs.empty:
                row = post_reqs.iloc[0]
                stats["p50"] = row["50%"]
                stats["p90"] = row["90%"]
                stats["p99"] = row["99%"]
                stats["tps"] = row["Requests/s"]
                stats["failures"] = row["Failure Count"]
            
            print(f"  ✅ Result: P99={stats['p99']}ms | TPS={stats['tps']:.1f}")
        else:
            print("  ⚠️ No data collected.")
    except Exception as e:
        print(f"  ❌ Error parsing CSV: {e}")
        
    return stats

def main():
    results = []
    
    # === 1. Workload Sweep (All Workloads x RLS) ===
    print("\n=== PHASE 1: Workload Characterization (RLS) ===")
    for w in WORKLOADS:
        results.append(run_test(w, "RLS", "UNIFORM"))

    # === 2. Strategy Comparison (ShareGPT x All Strategies) ===
    print("\n=== PHASE 2: Strategy Comparison (ShareGPT) ===")
    for s in STRATEGIES:
        if s == "RLS": continue # Already run
        results.append(run_test("ShareGPT", s, "UNIFORM"))
        
    # === 3. Bursty vs Uniform (FlowGPT) ===
    print("\n=== PHASE 3: Bursty Traffic Analysis (FlowGPT) ===")
    results.append(run_test("FlowGPT", "RLS", "BURSTY"))
    results.append(run_test("FlowGPT", "RoundRobin", "BURSTY"))

    # === 4. Load Sweep (ShareGPT x RLS) ===
    print("\n=== PHASE 4: Load Sweep (ShareGPT) ===")
    for u in [10, 50, 100, 200]:
        if u == 50: continue # Already run
        results.append(run_test("ShareGPT", "RLS", "UNIFORM", users=u))

    # Export
    df = pd.DataFrame(results)
    df.to_csv("results/comprehensive_metrics.csv", index=False)
    print("\n🔥 Comprehensive Suite Complete. Saved to results/comprehensive_metrics.csv")
    print(df)

if __name__ == "__main__":
    main()