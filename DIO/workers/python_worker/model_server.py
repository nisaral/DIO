import sys
import os
import time

import grpc
from concurrent import futures
import tiktoken
from google.protobuf import empty_pb2
from api.proto import dio_pb2
from api.proto import dio_pb2_grpc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the tokenizer for your specific model (e.g., GPT-4 or Llama-3)
tokenizer = tiktoken.get_encoding("cl100k_base") 

class LLMWorker(dio_pb2_grpc.InferenceWorkerServicer):
    def Predict(self, request, context):
        input_text = request.data.decode('utf-8')
        
        # 1. Count Input Tokens
        input_tokens = len(tokenizer.encode(input_text))
        
        # 2. Simulate LLM Inference (Replace with your model call)
        model_output = f"Processed: {input_text[:50]}..." 
        output_tokens = len(tokenizer.encode(model_output))
        
        total_tokens = input_tokens + output_tokens
        
        # 3. Check Context Window (e.g., 8192 tokens)
        context_full = total_tokens > 7000 

        logger.info(f"[Worker] Tokens used: {total_tokens}")

        return dio_pb2.InferenceResponse(
            output=model_output.encode('utf-8'),
            latency_ms=150.5,
            tokens_used=total_tokens,
            context_full=context_full
        )

    def CheckHealth(self, request, context):
        return empty_pb2.Empty()

def register_worker(port):
    """Registers this worker with the Go Manager."""
    manager_addr = os.environ.get('MANAGER_ADDRESS', 'localhost:50052')
    # In Docker, HOSTNAME is usually the container ID, which resolves to the IP
    worker_host = os.environ.get('HOSTNAME', 'localhost')
    worker_address = f"{worker_host}:{port}"

    logger.info(f"Attempting to register {worker_address} with Manager at {manager_addr}...")
    
    channel = grpc.insecure_channel(manager_addr)
    stub = dio_pb2_grpc.OrchestratorStub(channel)

    for i in range(15): # Retry for 30 seconds
        try:
            stub.RegisterWorker(dio_pb2.RegisterRequest(
                worker_id=worker_host,
                address=worker_address,
                models=["fraud-detection", "gpt-4"]
            ))
            logger.info("✅ Worker registered successfully!")
            return
        except grpc.RpcError:
            logger.warning(f"Manager not ready yet, retrying in 2s... ({i+1}/15)")
            time.sleep(2)
    logger.error("❌ Failed to register with Manager. Is it running?")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dio_pb2_grpc.add_InferenceWorkerServicer_to_server(LLMWorker(), server)
    server.add_insecure_port('[::]:50053')
    logger.info("Python Worker listening on port 50053")
    server.start()
    
    # Register with the manager after starting
    register_worker(50053)
    
    server.wait_for_termination()

if __name__ == '__main__':
    serve()