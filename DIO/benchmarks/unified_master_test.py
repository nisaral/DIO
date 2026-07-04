import subprocess
import time
import os
import requests
import shutil
import socket
import sys

# --- CONFIG ---
# Prefer env override; default supports Lightning studio layout
ROOT_DIR = os.environ.get("DIO_ROOT", "/teamspace/studios/this_studio/Go-serve/DIO")
if not os.path.exists(ROOT_DIR):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANAGER_BIN = os.path.join(ROOT_DIR, "dio-manager")
WORKER_SCRIPT = os.path.join(ROOT_DIR, "benchmarks/worker_gpu.py")
LOCUST_FILE = os.path.join(ROOT_DIR, "benchmarks/real_world/locustfile.py")
RESULTS_DIR = os.path.join(ROOT_DIR, "benchmarks/results_final")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# --- THE FULL 4-WORKER MATRIX ---
DATASETS = ["sharegpt.jsonl", "arxiv.jsonl", "azure_code.jsonl"]
STRATEGIES = ["nlms", "rls", "round_robin", "least_load"]
NUM_REAL_WORKERS = int(os.environ.get("NUM_REAL_WORKERS", "2"))  # 4 replicas on 1 GPU causes VRAM thrash

def rebuild_manager():
    print("🔨 Rebuilding DIO Manager...")
    try:
        subprocess.run(["go", "build", "-o", "dio-manager", "./cmd/manager/main.go"], cwd=ROOT_DIR, check=True)
        print("✅ Build Successful.")
    except subprocess.CalledProcessError:
        print("❌ Build Failed! Check your Go code."); sys.exit(1)

def cleanup():
    print("🧹 Forcefully clearing ports and processes...")
    subprocess.run(["pkill", "-9", "-f", "dio-manager"])
    subprocess.run(["pkill", "-9", "-f", "worker_gpu.py"])
    subprocess.run(["pkill", "-9", "-f", "locust"])
    time.sleep(5) 

def check_port_open(host, port, timeout=1):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port)); s.close(); return True
    except: return False

def start_manager(strategy):
    print(f"🔹 Starting Manager (Strategy: {strategy})...")
    env = os.environ.copy()
    env["SCHEDULER_STRATEGY"] = strategy
    log_f = open("manager.log", "w")
    subprocess.Popen([MANAGER_BIN], env=env, stdout=log_f, stderr=log_f)
    
    start_wait = time.time()
    while time.time() - start_wait < 10:
        if check_port_open("127.0.0.1", 50055): return True
        time.sleep(0.5)
    return False

def start_workers(count, mock=False):
    print(f"🚀 Spawning {count} {'MOCK' if mock else 'REAL'} workers...")
    for i in range(count):
        w_env = os.environ.copy()
        w_env["CUDA_VISIBLE_DEVICES"] = "0"
        
        # 18,000 MB per worker ensures we stay under 80GB with overhead
        vram_limit = "18000" if not mock else "1000"
        
        cmd = [
            "python", WORKER_SCRIPT, "--worker-id", f"w_{i}", "--port", str(50060 + i),
            "--model-id", MODEL_ID, "--manager-addr", "127.0.0.1:50055", "--vram", vram_limit
        ]
        if mock: cmd.append("--mock")
        
        with open(f"worker_{i}.log", "w") as f:
            subprocess.Popen(cmd, env=w_env, stdout=f, stderr=f)
        
        # Real GPU models need time to materialise weights
        time.sleep(0.5 if mock else 10.0)
    
    for _ in range(40):
        try:
            r = requests.get("http://127.0.0.1:8085/debug/workers")
            if r.status_code == 200 and r.json().get("worker_count", 0) >= count:
                print(f"✅ All {count} workers registered."); return True
        except: pass
        time.sleep(1)
    return False

def main():
    rebuild_manager()
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    # 1. SCALABILITY TEST (32 Mocks - Standard Paper Baseline)
    print("\n▶️ RUNNING: T7_Scalability_32")
    cleanup()
    if start_manager("nlms") and start_workers(32, mock=True):
        env = os.environ.copy()
        env["WORKLOAD_FILE"] = os.path.join(ROOT_DIR, "benchmarks/data/sharegpt.jsonl")
        subprocess.run(["locust", "-f", LOCUST_FILE, "--headless", "-u", "50", "-r", "10", "-t", "60s", 
                        "--host", "http://127.0.0.1:8085", "--csv", os.path.join(RESULTS_DIR, "T7_Scalability")], env=env)

    # 2. THE MAIN STUDY (default 2 real workers — safe on single A100)
    for strategy in STRATEGIES:
        for ds in DATASETS:
            test_id = f"{strategy}_{ds.split('.')[0]}_{NUM_REAL_WORKERS}w"
            print(f"\n🚀 EXPERIMENT: {test_id}")
            cleanup()
            
            if start_manager(strategy) and start_workers(NUM_REAL_WORKERS, mock=False):
                print(f"🧘 Warming up {NUM_REAL_WORKERS} workers (30s)...")
                time.sleep(30)
                
                l_env = os.environ.copy()
                l_env["WORKLOAD_FILE"] = os.path.join(ROOT_DIR, "benchmarks/data", ds)
                l_env["MODEL_ID"] = MODEL_ID
                
                # Using 20 users for higher saturation with 4 workers
                subprocess.run(["locust", "-f", LOCUST_FILE, "--headless", "-u", "20", "-r", "4", "-t", "90s", 
                                "--host", "http://127.0.0.1:8085", "--csv", os.path.join(RESULTS_DIR, test_id)], env=l_env)
    
    cleanup()
    print(f"\n✅ FULL MATRIX COMPLETE. Results in {RESULTS_DIR}")

if __name__ == "__main__": main()