import subprocess
import time
import os
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def check_dataset():
    if not os.path.exists("ShareGPT_V3_unfiltered_cleaned_split.json"):
        print("⚠️ ShareGPT dataset not found.")
        print("To run with real data, download it using:")
        print("huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered --repo-type dataset ShareGPT_V3_unfiltered_cleaned_split.json --local-dir .")
        print("Running with synthetic data for now...\n")

def run_throughput_baseline():
    print("\n=== Test 1: Throughput Baseline (vLLM Comparison) ===")
    print("Target: RPS 50-100, Output tok/s 200+")
    # Spawn 50 users, hatch rate 10/s, run for 30s
    cmd = "locust -f benchmarks/locustfile.py --headless --users 50 --spawn-rate 10 --run-time 30s --host http://localhost:8080"
    run_command(cmd)

def run_tail_latency():
    print("\n=== Test 2: Tail Latency (PARS Comparison) ===")
    print("Target: p99 Latency < 500ms")
    # Higher load, checking latency stats in locust output
    cmd = "locust -f benchmarks/locustfile.py --headless --users 100 --spawn-rate 20 --run-time 45s --host http://localhost:8080"
    run_command(cmd)

def run_autoscaling_stress():
    print("\n=== Test 3: Autoscaling Stress (Ray Serve Comparison) ===")
    print("Target: Scale to max workers, Error Rate < 1%")
    print("NOTE: Watch 'docker ps' in another terminal to see containers spawning!")
    # Ramp up to 200 users over 60s
    cmd = "locust -f benchmarks/locustfile.py --headless --users 200 --spawn-rate 5 --run-time 90s --host http://localhost:8080"
    run_command(cmd)

def run_straggler_simulation():
    print("\n=== Test 4: Straggler Simulation (Clockwork Comparison) ===")
    print("Target: p99 Latency < 2s despite delays")
    print("Injecting STRAGGLER_MODE into a new worker...")
    
    # Manually start a straggler worker
    # Note: This assumes docker is running and we can talk to the manager
    try:
        subprocess.Popen([
            "docker", "run", "-d", "--rm", 
            "--network", "dio_default", 
            "-e", "MANAGER_ADDRESS=dio-manager:50052",
            "-e", "STRAGGLER_MODE=true",
            "--name", "dio-straggler",
            "dio-worker"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Started 'dio-straggler' container.")
        time.sleep(5) # Wait for registration
        
        cmd = "locust -f benchmarks/locustfile.py --headless --users 50 --spawn-rate 10 --run-time 30s --host http://localhost:8080"
        run_command(cmd)
        
    finally:
        print("Cleaning up straggler...")
        subprocess.run("docker stop dio-straggler", shell=True)

if __name__ == "__main__":
    check_dataset()
    
    print("Make sure DIO is running (docker-compose up --build) before starting!")
    time.sleep(2)

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "throughput": run_throughput_baseline()
        elif test_name == "latency": run_tail_latency()
        elif test_name == "autoscale": run_autoscaling_stress()
        elif test_name == "straggler": run_straggler_simulation()
        else: print(f"Unknown test: {test_name}")
    else:
        # Run all
        run_throughput_baseline()
        time.sleep(5)
        run_tail_latency()
        time.sleep(5)
        run_autoscaling_stress()