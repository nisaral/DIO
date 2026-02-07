import subprocess, time, os, requests, pandas as pd, sys, shutil

# --- CONFIG ---
ROOT_DIR = "/teamspace/studios/this_studio/DIO/DIO"
MANAGER_BIN = os.path.join(ROOT_DIR, "dio-manager")
WORKER_SCRIPT = os.path.join(ROOT_DIR, "benchmarks/worker_gpu.py")
LOCUST_FILE = os.path.join(ROOT_DIR, "benchmarks/real_world/locustfile.py")
RESULTS_DIR = os.path.join(ROOT_DIR, "benchmarks/results_cloud")
MANAGER_URL = "http://127.0.0.1:8085"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# --- THE GOLDEN TEST SUITE (T1-T11) Adapted for 1x A100 ---
TEST_SUITE =[
    # T1: Convergence (Single Worker - Can use up to 75GB alone)
    {
        "id": "T1_Convergence",
        "desc": "Verify NLMS Convergence",
        "type": "PROBE",
        "duration": 60,
        "workers": [{"id": "t1_p1", "tier": "large", "vram": 70000, "gpu_idx": 0}]
    },
    
    # T2: Heterogeneity (Dual Workers - 38GB each fits perfectly in 80GB)
    {
        "id": "T2_Heterogeneity",
        "desc": "Simulated Heterogeneous Routing",
        "type": "LOCUST", "mode": "ROUTING", 
        "users": 20, "rate": 2, "duration": "60s",
        "workers": [
            {"id": "gpu_fast", "tier": "large", "vram": 38000, "gpu_idx": 0}, 
            {"id": "gpu_slow", "tier": "small", "vram": 38000, "gpu_idx": 0, "slow_factor": 2.5}
        ]
    },
    
    # T3: Cold Start
    {
        "id": "T3_ColdStart",
        "desc": "Cold Start Latency",
        "type": "LOCUST", "mode": "ROUTING", "users": 15, "rate": 2, "duration": "60s",
        "workers": [
            {"id": "w1", "tier": "large", "vram": 38000, "gpu_idx": 0},
            {"id": "w2", "tier": "large", "vram": 38000, "gpu_idx": 0, "delay": 20}
        ]
    },
    
    # T4: Roofline (VRAM Safety)
    # On an 80GB card, we set vram to 85,000 MB to trigger the proactive guard
    {
        "id": "T4_Roofline",
        "desc": "VRAM Saturation Safety",
        "type": "LOCUST", "mode": "ROUTING", "users": 40, "rate": 5, "duration": "45s",
        "workers": [{"id": "fragile", "tier": "small", "vram": 85000, "gpu_idx": 0}]
    },
    
    # T7: Scalability (64 Mocks)
    {
        "id": "T7_Scalability_64",
        "desc": "64 Workers Overhead",
        "type": "LOCUST", "mode": "ROUTING", "users": 100, "rate": 10, "duration": "60s",
        "workers": [{"id": f"w_{i}", "tier": "small", "vram": 1000, "mock": True, "gpu_idx": 0} for i in range(64)]
    },
    
    # T9-T11: Real Traces (Full 80GB utilization)
    {
        "id": "T9_RealData_ShareGPT",
        "desc": "Real World Trace: ShareGPT",
        "type": "LOCUST", "mode": "ROUTING", "users": 20, "rate": 2, "duration": "60s",
        "env": {"WORKLOAD_FILE": os.path.join(ROOT_DIR, "benchmarks/data/sharegpt.jsonl")},
        "workers": [{"id": "gpu_sharegpt", "tier": "large", "vram": 75000, "gpu_idx": 0}]
    },
    {
        "id": "T10_RealData_Arxiv",
        "desc": "Real World Trace: Arxiv Summarization",
        "type": "LOCUST", "mode": "ROUTING", "users": 20, "rate": 2, "duration": "60s",
        "env": {"WORKLOAD_FILE": os.path.join(ROOT_DIR, "benchmarks/data/arxiv.jsonl")},
        "workers": [{"id": "gpu_arxiv", "tier": "large", "vram": 75000, "gpu_idx": 0}]
    },
    {
        "id": "T11_RealData_AzureCode",
        "desc": "Real World Trace: Azure Code",
        "type": "LOCUST", "mode": "ROUTING", "users": 20, "rate": 2, "duration": "60s",
        "env": {"WORKLOAD_FILE": os.path.join(ROOT_DIR, "benchmarks/data/azure_code.jsonl")},
        "workers": [{"id": "gpu_code", "tier": "large", "vram": 75000, "gpu_idx": 0}]
    }
]
processes = []

def cleanup():
    print("\n🧹 Cleaning up processes...")
    subprocess.run("pkill -9 -f dio-manager", shell=True)
    subprocess.run("pkill -9 -f worker_gpu.py", shell=True)
    time.sleep(1)

def start_manager():
    print("🔹 Starting Manager...")
    p = subprocess.Popen([MANAGER_BIN], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(p)
    for _ in range(10):
        try:
            if requests.get(f"{MANAGER_URL}/api/test", timeout=1).status_code == 200:
                print("✅ Manager HTTP API is live.")
                return
        except:
            time.sleep(1)
    print("⚠️ Manager taking a long time to start...")

def wait_for_workers(expected_count):
    print(f"⏳ Verifying {expected_count} workers via Manager API...")
    debug_url = f"{MANAGER_URL}/debug/workers"
    for i in range(30):
        try:
            r = requests.get(debug_url, timeout=1)
            if r.status_code == 200:
                data = r.json()
                count = data.get("worker_count", 0)
                if count >= expected_count:
                    print(f"✅ Manager Confirmed: {count} Workers Active")
                    return True
                else:
                    if i % 5 == 0: print(f"   [Attempt {i+1}] Manager sees {count}/{expected_count} workers...")
        except:
            if i % 5 == 0: print(f"   [Attempt {i+1}] Waiting for Manager HTTP API...")
        time.sleep(2)
    return False

def start_workers(worker_list):
    base_port = 50060
    for i, w in enumerate(worker_list):
        port = base_port + i
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(w.get("gpu_idx", 0))
        env["PYTHONPATH"] = os.path.join(ROOT_DIR, "benchmarks")
        
        cmd = [
            "python", WORKER_SCRIPT,
            "--worker-id", w["id"],
            "--port", str(port),
            "--tier", w.get("tier", "large"),
            "--vram", str(w.get("vram", 12000)),
            "--model-id", MODEL_ID,
            "--manager-addr", "localhost:50055" 
        ]
        
        if w.get("mock"): cmd.append("--mock")
        if w.get("slow_factor"): cmd.extend(["--latency-mult", str(w["slow_factor"])])
        
        subprocess.Popen(cmd, env=env)
        print(f"   🚀 Spawning {w['id']} on GPU 0 (Port {port})")
        time.sleep(1) 
    
    return wait_for_workers(len(worker_list))

def run_locust_test(config, csv_base):
    env = os.environ.copy()
    env["TEST_MODE"] = config.get('mode', 'ROUTING')
    env["MODEL_ID"] = MODEL_ID
    if "env" in config: env.update(config["env"])
    subprocess.run(["locust", "-f", LOCUST_FILE, "--headless", "--users", str(config['users']), "--spawn-rate", str(config['rate']), "--run-time", config['duration'], "--host", MANAGER_URL, "--csv", csv_base], env=env)

def run_probe_test(config, csv_path):
    results = []
    start = time.time()
    while time.time() - start < config['duration']:
        try:
            r = requests.post(f"{MANAGER_URL}/api/generate", 
                             json={"prompt": "test", "model_id": MODEL_ID, "tier": "large"}, 
                             timeout=5)
            if r.status_code == 200:
                d = r.json()
                results.append({"timestamp": time.time(), "latency_ms": d.get("latency_ms", 0)})
        except:
            pass
        time.sleep(0.5)
    
    if results:
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"✅ Saved {len(results)} rows to {csv_path}")

def main():
    if os.path.exists(RESULTS_DIR): shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR)
    summary = []
    for test in TEST_SUITE:
        print(f"\n▶️ RUNNING: {test['id']}")
        cleanup(); start_manager()
        if not start_workers(test['workers']):
            summary.append({"Test": test['id'], "Status": "FAILED_REGISTRATION"})
            continue
        
        csv_p = os.path.join(RESULTS_DIR, f"{test['id']}_stats.csv")
        if test['type'] == 'LOCUST':
            run_locust_test(test, os.path.join(RESULTS_DIR, test['id']))
        else:
            run_probe_test(test, csv_p)
        
        status = "PASS" if os.path.exists(csv_p) and os.path.getsize(csv_p) > 50 else "EMPTY"
        summary.append({"Test": test['id'], "Status": status})
    
    cleanup()
    print("\n📊 FINAL SUMMARY\n", pd.DataFrame(summary))

if __name__ == "__main__": main()