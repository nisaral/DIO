import sys
import os
import time
import grpc
from concurrent import futures

# Ensure we can import the generated protos
# Assumes running from project root or tests/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../workers/python_worker')))

from api.proto import dio_pb2
from api.proto import dio_pb2_grpc
from google.protobuf import empty_pb2

class SlowWorker(dio_pb2_grpc.InferenceWorkerServicer):
    def Predict(self, request, context):
        print(f"[SlowWorker] Received request for {request.model_id}. Sleeping for 5s...")
        time.sleep(5) # The "Straggler" behavior
        return dio_pb2.InferenceResponse(output=b"Delayed Response", latency_ms=5000)

    def CheckHealth(self, request, context):
        return empty_pb2.Empty()

def register(manager_addr, worker_port):
    try:
        channel = grpc.insecure_channel(manager_addr)
        stub = dio_pb2_grpc.OrchestratorServiceStub(channel)
        stub.RegisterWorker(dio_pb2.RegisterRequest(
            worker_id="slow-worker-1",
            address=f"localhost:{worker_port}",
            models=["bert-base"]
        ))
        print(f"[SlowWorker] Registered with Manager at {manager_addr}")
    except Exception as e:
        print(f"[SlowWorker] Failed to register: {e}")

def serve(port, manager_addr):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    dio_pb2_grpc.add_InferenceWorkerServicer_to_server(SlowWorker(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"[SlowWorker] Listening on port {port}")
    register(manager_addr, port)
    server.wait_for_termination()

if __name__ == '__main__':
    # Run on a different port than the normal worker (usually 50051)
    serve(50052, 'localhost:50050')