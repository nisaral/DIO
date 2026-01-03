import sys
import os
import grpc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../workers/python_worker')))

from api.proto import dio_pb2
from api.proto import dio_pb2_grpc

def run_budget_test():
    print("--- Test C: The 'Token Budget' Test ---")
    print("Goal: Verify Manager rejects requests that exceed token/cost budget.")
    
    channel = grpc.insecure_channel('localhost:50050')
    stub = dio_pb2_grpc.OrchestratorServiceStub(channel)

    # 1. Normal Request
    try:
        print("\n1. Sending Normal Request (Small payload)...")
        stub.ExecuteInference(dio_pb2.InferenceRequest(model_id="bert-base", data=b"Hello world"))
        print("   -> Success (Expected)")
    except grpc.RpcError as e:
        print(f"   -> Failed: {e.details()}")

    # 2. Expensive Request
    try:
        print("\n2. Sending Expensive Request (Large payload > Budget)...")
        # Simulate a large input that would consume many tokens
        large_payload = b"word " * 5000 
        stub.ExecuteInference(dio_pb2.InferenceRequest(model_id="bert-base", data=large_payload))
        print("   -> Success (Unexpected! Should have been rejected)")
    except grpc.RpcError as e:
        print(f"   -> Rejected: {e.details()} (Expected)")

if __name__ == '__main__':
    run_budget_test()