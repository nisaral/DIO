import sys
import os
import grpc
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../workers/python_worker')))

from api.proto import dio_pb2
from api.proto import dio_pb2_grpc

def send_request(req_id):
    channel = grpc.insecure_channel('localhost:50050')
    stub = dio_pb2_grpc.OrchestratorServiceStub(channel)
    try:
        stub.ExecuteInference(dio_pb2.InferenceRequest(model_id="bert-base", data=b"spike_load"))
        return "OK"
    except Exception:
        return "FAIL"

def run_spike_test():
    print("--- Test B: The 'Sudden Spike' Test ---")
    print("Goal: Hit Manager with 50 concurrent requests to trigger Autoscaling.")
    
    # Simulate 50 concurrent users
    concurrency = 50
    total_requests = 200
    
    print(f"Launching {total_requests} requests with {concurrency} concurrency...")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(send_request, range(total_requests)))
    
    print(f"Completed. Success: {results.count('OK')}, Failed: {results.count('FAIL')}")
    print(">> Check 'docker ps' to see if new worker containers were created.")

if __name__ == '__main__':
    run_spike_test()