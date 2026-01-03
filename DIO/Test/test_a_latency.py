import sys
import os
import time
import grpc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../workers/python_worker')))

from api.proto import dio_pb2
from api.proto import dio_pb2_grpc

def run_test():
    print("--- Test A: The 'Straggler' Test ---")
    print("Goal: Verify if the Manager avoids the slow worker (localhost:50052) after detecting latency.")
    
    channel = grpc.insecure_channel('localhost:50050')
    stub = dio_pb2_grpc.OrchestratorServiceStub(channel)

    for i in range(1, 11):
        start = time.time()
        try:
            # Sending request
            response = stub.ExecuteInference(dio_pb2.InferenceRequest(
                model_id="bert-base",
                data=b"test_payload"
            ))
            elapsed = (time.time() - start) * 1000
            print(f"Req #{i}: Latency={elapsed:.0f}ms | Worker Reported={response.latency_ms}ms")
        except grpc.RpcError as e:
            print(f"Req #{i}: Failed ({e.code()}) - {e.details()}")

if __name__ == '__main__':
    run_test()