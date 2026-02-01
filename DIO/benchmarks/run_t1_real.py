import subprocess
import time
import os
import pandas as pd

# Config
USERS = 50
SPAWN_RATE = 5
DURATION = "45s"
HOST = "http://localhost:8080"
RESULTS_DIR = "results_4050"

def run_test(strategy):
    print(f"\n🚀 Running T1 Real: {strategy} on RTX 4050 (DialoGPT-medium)")
    
    # 1. Restart DIO with specific strategy
    print("  ⚡ Restarting Docker Cluster...")
    env = os.environ.copy()
    env["SCHEDULER_STRATEGY"] = strategy
    
    subprocess.run(["docker-compose", "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker-compose", "up", "-d", "--build"], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("  ⏳ Warming up model (20s)...")
    time.sleep(20)
    
    # 2. Run Locust
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_name = f"{RESULTS_DIR}/T1_{strategy}"
    
    cmd = [
        "locust", "-f", "benchmarks/locustfile.py",
        "--headless",
        f"--users={USERS}",
        f"--spawn-rate={SPAWN_RATE}",
        f"--run-time={DURATION}",
        f"--host={HOST}",
        "--csv", csv_name
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. Parse P99
    try:
        df = pd.read_csv(f"{csv_name}_stats.csv")
        p99 = df[df["Type"] == "POST"].iloc[0]["99%"]
        print(f"  ✅ {strategy} Result: p99 = {p99} ms")
        return p99
    except Exception as e:
        print(f"  ❌ Failed to parse results: {e}")
        return 0

if __name__ == "__main__":
    # Run Round Robin (Baseline)
    rr_p99 = run_test("RoundRobin")
    
    # Run RLS (DIO)
    rls_p99 = run_test("RLS")
    
    print("\n========================================")
    print(f"🏆 FINAL RESULTS (Lower is Better)")
    print(f"   Round Robin: {rr_p99} ms")
    print(f"   DIO (RLS):   {rls_p99} ms")
    
    if rr_p99 > 0:
        improvement = ((rr_p99 - rls_p99) / rr_p99) * 100
        print(f"   🚀 Improvement: {improvement:.1f}%")
    print("========================================")