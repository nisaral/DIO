import grpc
import sys
import os

# Add the api/proto directory to path so we can import the generated protobuf files
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api', 'proto'))

import dio_pb2
import dio_pb2_grpc

def run():
    # 1. Open a connection to the Go Manager
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = dio_pb2_grpc.OrchestratorStub(channel)
        
        # 2. Prepare the registration data
        request = dio_pb2.RegisterRequest(
            worker_id="python-gpu-node-01",
            address="localhost:50052", # This node's address
            models=["llama-3", "fraud-detection"]
        )
        
        # 3. Call the Go Manager
        response = stub.RegisterWorker(request)
        
        if response.success:
            print("Successfully registered with DIO Manager!")
        else:
            print("Registration failed.")

if __name__ == '__main__':
    run()