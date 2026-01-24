import subprocess
import os
import sys

# Ensure we are running from the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.getcwd() != project_root:
    os.chdir(project_root)

RESULTS_DIR = "results"
LOCUST_FILE = "benchmarks/locustfile_vllm.py"
HOST = "http://localhost:8000"

def run_vllm_benchmark():
    print("==================================================")
    print(" RUNNING T6: vLLM Baseline")
    print("Goal: Establish Speed-of-Light baseline with C++/CUDA engine.")
    print(" Ensure vLLM is running on port 8000!")
    print("==================================================")
    
    csv_path = os.path.join(RESULTS_DIR, "T6_vLLM_Baseline")
    
    cmd = [
        "locust", "-f", LOCUST_FILE,
        "--headless",
        "--users=50",
        "--spawn-rate=10",
        "--run-time=2m",
        f"--host={HOST}",
        "--csv", csv_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"T6 Complete. Results saved to {csv_path}_stats.csv")
    except subprocess.CalledProcessError:
        print("Test Failed. Is vLLM running?")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_vllm_benchmark()