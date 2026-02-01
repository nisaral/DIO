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
        "desc": "Head-of-Line Blocking (NLMS vs RR)",
        "users": 100,
        "workers": 2,
        "strategy": "NLMS",
        "env": {},
        "mixed": False
    },
    {
        "id": "T2_Hetero",
        "desc": "Heterogeneous Adaptation (Convergence)",
        "users": 200,
        "workers": 2, # 1 fast + 1 slow
        "strategy": "NLMS",
        "env": {"THROTTLE_FACTOR": "MIXED"},
        "mixed": True
    },
    {
       "id": "T4_Roofline",
       "desc": "Memory Constraints (VRAM)",
       "users": 200,
       "workers": 2,
       "strategy": "NLMS",
       "env": {"ROOFLINE_TEST": "TRUE"},
       "mixed": False
    },
    {
       "id": "T7_Scale_8",
       "desc": "Scalability Baseline (2 Workers)",
       "users": 40,
       "workers": 2,
       "strategy": "NLMS",
       "env": {},
       "mixed": False
    },
    {
       "id": "T7_Scale_64",
       "desc": "Scalability Stress (3 Workers)",
       "users": 320,
       "workers": 3,
       "strategy": "NLMS",
       "env": {},
       "mixed": False
    },
    {
       "id": "T3_ColdStart",
       "desc": "Scale-Up / Cold Start (1 -> 2 Workers)",
       "users": 100,
       "workers": 1, # Start small
       "strategy": "NLMS",
       "env": {},
       "mixed": False,
       "mid_test_action": {"delay": 15, "cmd": ["docker-compose", "up", "-d", "--scale", "dio-worker=2"]}
    },
    {
       "id": "T5_Spike",
       "desc": "Queue Awareness (Sudden Spike)",
       "users": 200,
       "workers": 2,
       "strategy": "NLMS",
       "env": {},
       "mixed": False,
       "spawn_rate": 50 # Aggressive spawn rate
    },
    # {
    #     "id": "T6_vLLM",
    #     "desc": "vLLM Baseline (External)",
    #     "users": 50,
    #     "workers": 0, # External
    #     "strategy": "N/A",
    #     "env": {},
    #     "mixed": False,
    #     "skip_docker": True,
    #     "locust_file": "benchmarks/locustfile_vllm.py",
    #     "host": "http://localhost:8000"
    # }
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
    ttft_p99s = []
    tpot_means = []
    
    for i in range(ITERATIONS):
        print(f"  Iteration {i+1}/{ITERATIONS}...")
        
        # 1. Setup environment
        env = os.environ.copy()
        env["SCHEDULER_STRATEGY"] = scenario["strategy"]
        if scenario["env"]:
            env.update(scenario["env"])
        
        # Docker Setup (Skip for vLLM/External)
        if not scenario.get("skip_docker", False):
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
        os.makedirs("results_final", exist_ok=True)
        csv_name = f"results_final/{scenario['id']}_iter{i}"
        
        locust_file = scenario.get("locust_file", "benchmarks/locustfile.py")
        host = scenario.get("host", "http://localhost:8080")
        spawn_rate = scenario.get("spawn_rate", 10)
        
        cmd = f"locust -f {locust_file} --headless -u {scenario['users']} -r {spawn_rate} --run-time 45s --csv {csv_name} --host {host}"
        
        # Handle Mid-Test Action (T3)
        if "mid_test_action" in scenario:
            proc = subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(scenario["mid_test_action"]["delay"])
            print(f"    ⚡ Triggering Action: {scenario['mid_test_action']['cmd']}")
            subprocess.run(scenario["mid_test_action"]["cmd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.wait()
        else:
            subprocess.run(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 3. Collect Metrics
        try:
            df = pd.read_csv(f"{csv_name}_stats.csv")
            if not df.empty:
                # Filter for Type="POST" to get accurate E2E Latency and TPS
                # Ignoring custom events (TTFT/TPOT) for the main stability metrics
                post_reqs = df[df["Type"] == "POST"]
                if not post_reqs.empty:
                    row = post_reqs.iloc[0]
                    p99s.append(row['99%'])
                    tpss.append(row['Requests/s'])
                else:
                    # Fallback if no POST requests found (unlikely)
                    p99s.append(0)
                    tpss.append(0)
                
                # Extract TTFT (P99)
                ttft_reqs = df[df["Type"] == "TTFT"]
                if not ttft_reqs.empty:
                    ttft_p99s.append(ttft_reqs.iloc[0]['99%'])
                else:
                    ttft_p99s.append(0)

                # Extract TPOT (Mean)
                tpot_reqs = df[df["Type"] == "TPOT"]
                if not tpot_reqs.empty:
                    tpot_means.append(tpot_reqs.iloc[0]['Average Response Time'])
                else:
                    tpot_means.append(0)

            else:
                p99s.append(0)
                tpss.append(0)
        except Exception as e:
            p99s.append(0)
            tpss.append(0)
            ttft_p99s.append(0)
            
        # 4. Collect Overhead
        if not scenario.get("skip_docker", False):
            overheads_mean.append(get_overhead_stats())
        
    # Stats
    def calc_stats(prefix, p, t, o, ttft, tpot):
        # Handle empty overheads (e.g. vLLM test where docker logs aren't checked)
        if len(o) > 0:
            o_mean = np.mean(o)
            o_std = np.std(o)
        else:
            o_mean = 0.0
            o_std = 0.0

        return {
            f"{prefix}_p99_mean": np.mean(p),
            f"{prefix}_p99_std": np.std(p),
            f"{prefix}_tps_mean": np.mean(t),
            f"{prefix}_tps_std": np.std(t),
            f"{prefix}_tpot_mean": np.mean(p) / 128.0,
            f"{prefix}_tpot_std": np.std(p) / 128.0,
            f"{prefix}_real_ttft_p99": np.mean(ttft),
            f"{prefix}_real_tpot_mean": np.mean(tpot),
            f"{prefix}_overhead_mean": o_mean,
            f"{prefix}_overhead_std": o_std
        }

    s5 = calc_stats("5iter", p99s[:5], tpss[:5], overheads_mean[:5], ttft_p99s[:5], tpot_means[:5])
    s10 = calc_stats("10iter", p99s, tpss, overheads_mean, ttft_p99s, tpot_means)

    stats = {
        "test": scenario["id"],
        "workers": scenario["workers"],
        **s5,
        **s10
    }
    
    print(f"  ✅ {scenario['id']} Result:")
    print(f"     p99 (10 iter): {stats['10iter_p99_mean']:.2f} ± {stats['10iter_p99_std']:.2f} ms")
    print(f"     TPS (10 iter): {stats['10iter_tps_mean']:.2f} ± {stats['10iter_tps_std']:.2f}")
    if stats['10iter_real_ttft_p99'] > 0:
        print(f"     TTFT P99:      {stats['10iter_real_ttft_p99']:.2f} ms")
    else:
        print(f"     TTFT P99:      N/A (Not Reported)")
    print(f"     TPOT Mean:     {stats['10iter_real_tpot_mean']:.2f} ms")
    print(f"     Overhead: {stats['10iter_overhead_mean']:.0f} ns")
    
    return stats

# === Main Execution ===
results = []
for s in SCENARIOS:
    results.append(run_scenario(s))

# Export for Paper
df = pd.DataFrame(results)
file_exists = os.path.isfile("results_final/final_rigorous_metrics.csv")
df.to_csv("results_final/final_rigorous_metrics.csv", index=False, mode='a', header=not file_exists)
print("\n🔥 All tests complete. Results saved to results_final/final_rigorous_metrics.csv")
print(df)