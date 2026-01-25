import subprocess
import time
import numpy as np
import pandas as pd
import os
import re

ITERATIONS = 10 # Increased to 10 for better statistical significance
WORKER_SCALES = [4, 16, 32, 64] # To prove O(1) scalability

# Ensure we are running from the project root so docker-compose finds the yaml
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.getcwd() != project_root:
    os.chdir(project_root)

SCENARIOS = [
    {
        "id": "T1_HoL",
        "desc": "Head-of-Line Blocking (RLS vs RR)",
        "users": 100,
        "workers": 8,
        "strategy": "RLS",
        "env": {},
        "mixed": False
    },
    {
        "id": "T2_Hetero",
        "desc": "Heterogeneous Adaptation (Convergence)",
        "users": 200,
        "workers": 6, # 4 fast + 2 slow
        "strategy": "RLS",
        "env": {"THROTTLE_FACTOR": "MIXED"},
        "mixed": True
    },
    {
        "id": "T4_Roofline",
        "desc": "Memory Constraints (VRAM)",
        "users": 200,
        "workers": 8,
        "strategy": "RLS",
        "env": {"ROOFLINE_TEST": "TRUE"},
        "mixed": False
    },
    {
        "id": "T7_Scale_8",
        "desc": "Scalability Baseline (8 Workers)",
        "users": 40,
        "workers": 8,
        "strategy": "RLS",
        "env": {},
        "mixed": False
    },
    {
        "id": "T7_Scale_64",
        "desc": "Scalability Stress (64 Workers)",
        "users": 320,
        "workers": 64,
        "strategy": "RLS",
        "env": {},
        "mixed": False
    }
]

def get_overhead_stats(container_name="dio-dio-manager-1"):
    try:
        # Fetch logs
        result = subprocess.run(["docker", "logs", container_name], capture_output=True, text=True)
        logs = result.stderr + result.stdout
        
        # Parse [SCHED_OVERHEAD] %d ns
        overheads = []
        for line in logs.splitlines():
            match = re.search(r"\[SCHED_OVERHEAD\] (\d+) ns", line)
            if match:
                overheads.append(int(match.group(1)))
        
        if not overheads:
            return 0
            
        return np.mean(overheads)
    except Exception as e:
        print(f"  ⚠️ Failed to fetch overheads: {e}")
        return 0

def run_scenario(scenario):
    print(f"\n🚀 Running {scenario['id']}: {scenario['desc']}")
    
    p99s = []
    tpss = []
    overheads_mean = []
    
    for i in range(ITERATIONS):
        print(f"  Iteration {i+1}/{ITERATIONS}...")
        
        # 1. Setup environment
        env = os.environ.copy()
        env["SCHEDULER_STRATEGY"] = scenario["strategy"]
        if scenario["env"]:
            env.update(scenario["env"])
        
        # Restart to ensure clean state
        subprocess.run(["docker-compose", "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if scenario["mixed"]:
            # Dynamic scaling for mixed (approx 2:1 ratio)
            slow = max(1, scenario['workers'] // 3)
            fast = scenario['workers'] - slow
            cmd = ["docker-compose", "up", "-d", "--scale", f"dio-worker={fast}", "--scale", f"dio-worker-slow={slow}"]
        else:
            cmd = ["docker-compose", "up", "-d", "--scale", f"dio-worker={scenario['workers']}"]
            
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(20) # Warmup
        
        # 2. Run Locust
        os.makedirs("results", exist_ok=True)
        csv_name = f"results/{scenario['id']}_iter{i}"
        
        cmd = f"locust -f benchmarks/locustfile.py --headless -u {scenario['users']} -r 10 --run-time 45s --csv {csv_name} --host http://localhost:8080"
        subprocess.run(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 3. Collect Metrics
        try:
            df = pd.read_csv(f"{csv_name}_stats.csv")
            if not df.empty:
                p99s.append(df.iloc[-1]['99%'])
                tpss.append(df.iloc[-1]['Requests/s'])
            else:
                p99s.append(0)
                tpss.append(0)
        except Exception as e:
            p99s.append(0)
            tpss.append(0)
            
        # 4. Collect Overhead
        overheads_mean.append(get_overhead_stats())
        
    # Stats
    def calc_stats(prefix, p, t, o):
        return {
            f"{prefix}_p99_mean": np.mean(p),
            f"{prefix}_p99_std": np.std(p),
            f"{prefix}_tps_mean": np.mean(t),
            f"{prefix}_tps_std": np.std(t),
            f"{prefix}_tpot_mean": np.mean(p) / 128.0,
            f"{prefix}_tpot_std": np.std(p) / 128.0,
            f"{prefix}_overhead_mean": np.mean(o),
            f"{prefix}_overhead_std": np.std(o)
        }

    s5 = calc_stats("5iter", p99s[:5], tpss[:5], overheads_mean[:5])
    s10 = calc_stats("10iter", p99s, tpss, overheads_mean)

    stats = {
        "test": scenario["id"],
        "workers": scenario["workers"],
        **s5,
        **s10
    }
    
    print(f"  ✅ {scenario['id']} Result:")
    print(f"     p99 (10 iter): {stats['10iter_p99_mean']:.2f} ± {stats['10iter_p99_std']:.2f} ms")
    print(f"     TPS (10 iter): {stats['10iter_tps_mean']:.2f} ± {stats['10iter_tps_std']:.2f}")
    print(f"     Overhead: {stats['10iter_overhead_mean']:.0f} ns")
    
    return stats

# === Main Execution ===
results = []
for s in SCENARIOS:
    results.append(run_scenario(s))

# Export for Paper
df = pd.DataFrame(results)
df.to_csv("results/final_rigorous_metrics.csv", index=False)
print("\n🔥 All tests complete. Results saved to results/final_rigorous_metrics.csv")
print(df)