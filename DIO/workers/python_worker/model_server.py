import os
import grpc
import sys
import time
import socket
import logging
from concurrent import futures
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Add parent dir to path to import proto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import api.proto.dio_pb2 as dio_pb2
import api.proto.dio_pb2_grpc as dio_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Worker")

# --- Configuration ---
MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIER = os.environ.get("WORKER_TIER", "small")

logger.info(f"🚀 Worker Starting | Tier: {TIER} | Device: {DEVICE}")
logger.info(f"⏳ Loading Model: {MODEL_ID}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    logger.info(f"✅ Model {MODEL_ID} Loaded Successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    pipe = None

class InferenceWorker(dio_pb2_grpc.InferenceWorkerServicer):
    def Predict(self, request, context):
        start_time = time.time()
        input_text = request.data.decode("utf-8")
        logger.info(f"📨 Request: {len(input_text)} chars | Model: {request.model_id}")
        
        if pipe:
            try:
                # Real Inference
                outputs = pipe(
                    input_text, 
                    max_new_tokens=100, 
                    do_sample=True, 
                    temperature=0.7,
                    truncation=True
                )
                generated_text = outputs[0]["generated_text"]
                # Strip prompt from output for cleaner logging
                if generated_text.startswith(input_text):
                    generated_text = generated_text[len(input_text):]
            except Exception as e:
                logger.error(f"Inference Error: {e}")
                generated_text = f"Error: {str(e)}"
        else:
            # Fallback if model failed to load
            time.sleep(0.1)
            generated_text = "Model not loaded, simulation mode."

        latency = (time.time() - start_time) * 1000
        tokens_used = len(generated_text.split()) # Approx token count
        
        return dio_pb2.InferenceResponse(
            output=generated_text.encode("utf-8"),
            latency_ms=latency,
            ttft_ms=latency / 10.0, # Mock TTFT for non-streaming
            tokens_used=tokens_used
        )

    def CheckHealth(self, request, context):
        return dio_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

def register_with_manager(port):
    manager_addr = os.environ.get('MANAGER_ADDRESS', 'localhost:50052')
    hostname = os.environ.get('HOSTNAME', socket.gethostname())
    
    # Resolve IP (hack for Docker networking)
    try:
        worker_ip = socket.gethostbyname(hostname)
    except:
        worker_ip = "127.0.0.1"
        
    worker_address = f"{worker_ip}:{port}"
    
    # FIX: Handle ngrok secure connection
    if "ngrok" in manager_addr or "443" in manager_addr:
        # Strip protocol if present
        target = manager_addr.replace("https://", "").replace("http://", "")
        creds = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(target, creds)
    else:
        channel = grpc.insecure_channel(manager_addr)

    stub = dio_pb2_grpc.OrchestratorStub(channel)
    
    logger.info(f"📡 Registering with Manager at {manager_addr}...")
    try:
        # Add metadata to skip ngrok warning page
        stub.RegisterWorker(dio_pb2.RegisterRequest(
            worker_id=f"{hostname}-{TIER}",
            address=worker_address,
            tier=TIER,
            vram_gb=int(os.environ.get("WORKER_VRAM_GB", "8"))
        ), metadata=[('ngrok-skip-browser-warning', 'true')])
        logger.info("✅ Registered!")
    except Exception as e:
        logger.warning(f"⚠️ Registration failed: {e}")

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dio_pb2_grpc.add_InferenceWorkerServicer_to_server(InferenceWorker(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"🎧 Worker listening on port {port}")
    
    # Register once on startup
    register_with_manager(port)
    
    server.wait_for_termination()

if __name__ == '__main__':
    serve()