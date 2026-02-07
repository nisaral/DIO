import argparse
import time
import torch
import os
import sys
import grpc
from concurrent import futures
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure we can import the protobufs
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import dio_pb2 as pb
import dio_pb2_grpc as pb_grpc

class InferenceWorker(pb_grpc.InferenceWorkerServicer):
    def __init__(self, args, tokenizer, model):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model

    def Predict(self, request, context):

        try:
            # Python gRPC converted your proto 'Data' field to lowercase 'data'
            prompt = request.data.decode("utf-8")
        except AttributeError:
            # Fallback in case your environment uses a different generator
            prompt = request.Data.decode("utf-8")
            
        output_len = 128 
        start_time = time.perf_counter()
        ttft = 0
        
        # 2. Execution logic (optimized for CPU stability)
        if self.model:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            # Use a slightly more efficient generation for CPU
            with torch.no_grad():
                # Measure TTFT
                _ = self.model.generate(**inputs, max_new_tokens=1, pad_token_id=self.tokenizer.eos_token_id)
                ttft = (time.perf_counter() - start_time) * 1000
                # Generate full response
                outputs = self.model.generate(**inputs, max_new_tokens=output_len, pad_token_id=self.tokenizer.eos_token_id)
            tokens_generated = len(outputs[0]) - len(inputs[0])
        else:
            # MOCK logic
            time.sleep(0.05) 
            ttft = 50.0
            time.sleep(output_len * 0.01) 
            tokens_generated = output_len

        total_latency = (time.perf_counter() - start_time) * 1000
        
        return pb.InferenceResponse(
            latency_ms=total_latency * self.args.latency_mult,
            tokens_used=tokens_generated,
            ttft_ms=ttft * self.args.latency_mult
        )
        
        prompt = request.Data.decode("utf-8")
        output_len = 128
        start_time = time.perf_counter()
        
        # MOCK INFERENCE (Fast Path for debugging)
        if self.args.mock or self.model is None:
            time.sleep(0.05)
            ttft = 50.0
            time.sleep(output_len * 0.005)
            tokens_generated = output_len
        else:
            # REAL INFERENCE
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                # TTFT
                _ = self.model.generate(**inputs, max_new_tokens=1)
                ttft = (time.perf_counter() - start_time) * 1000
                # Full Generation
                outputs = self.model.generate(**inputs, max_new_tokens=output_len)
                tokens_generated = len(outputs[0]) - len(inputs[0])

        total_latency = (time.perf_counter() - start_time) * 1000
        
        # Apply Heterogeneity Multiplier
        final_latency = total_latency * self.args.latency_mult
        if self.args.latency_mult > 1.0:
            time.sleep((final_latency - total_latency) / 1000)

        return pb.InferenceResponse(
            latency_ms=final_latency,
            tokens_used=int(tokens_generated),
            ttft_ms=ttft
        )

def load_model(is_mock=False, model_id="meta-llama/Llama-3.2-1B-Instruct"):
    if is_mock: 
        return None, None
    
    # Check for CPU vs GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading {model_id} on {device.upper()}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # CPU Optimization: removed torch_dtype=torch.float16 as it can be slow on CPU
        # Added low_cpu_mem_usage for better Studio performance
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, # Change back from default for 2x GPU speed
            device_map="auto"
        )
        return tokenizer, model
    except Exception as e:
        print(f"❌ LOAD FAILURE: {e}. Falling back to MOCK.")
        return None, None

def register_with_manager(args, worker_address):
    print(f"📢 Attempting gRPC Registration to {args.manager_addr}...")
    try:
        # Create a channel to the Manager's gRPC port
        with grpc.insecure_channel(args.manager_addr) as channel:
            stub = pb_grpc.OrchestratorStub(channel)
            
            # Send the RegisterWorker RPC
            resp = stub.RegisterWorker(pb.RegisterRequest(
                worker_id=args.worker_id,
                address=worker_address,
                tier=args.tier,
                vram_gb=int(args.vram) # Proto expects int64
            ))
            
            if resp.success:
                print(f"✅ Successfully Registered with Manager via gRPC!")
            else:
                print(f"⚠️ Manager rejected registration (Success=False).")
                
    except Exception as e:
        print(f"❌ gRPC Registration Failed: {e}")
        # We don't exit here because the server should stay alive for debug, 
        # but in prod we would exit.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--port", type=int, default=50060)
    parser.add_argument("--tier", default="large")
    parser.add_argument("--vram", type=int, default=12000) # MB
    parser.add_argument("--latency-mult", type=float, default=1.0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    # This defaults to the gRPC port now
    parser.add_argument("--manager-addr", type=str, default="localhost:50055") 
    args = parser.parse_args()

    tokenizer, model = load_model(args.mock, args.model_id)
    # 1. Start Worker gRPC Server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_InferenceWorkerServicer_to_server(
        InferenceWorker(args, tokenizer, model), server
    )
    server.add_insecure_port(f'127.0.0.1:{args.port}')
    server.start()
    
    worker_address = f"localhost:{args.port}"
    print(f"✅ Worker {args.worker_id} listening on {worker_address}")

    # 2. Register via gRPC (THE FIX)
    print(f"📢 Attempting gRPC Registration to {args.manager_addr}...")
    try:
        # Create a channel to the Manager's gRPC port
        with grpc.insecure_channel(args.manager_addr) as channel:
            stub = pb_grpc.OrchestratorStub(channel)
            
            # Match the Proto definition exactly
            req = pb.RegisterRequest(
                worker_id=args.worker_id,
                address=worker_address,
                tier=args.tier,
                vram_gb=args.vram # Your Go code treats this as int64
            )
            
            resp = stub.RegisterWorker(req)
            
            if resp.success:
                print(f"✅ Successfully Registered with Manager via gRPC!")
            else:
                print(f"⚠️ Manager rejected registration.")
                exit(1) # Fail fast if rejected
    except Exception as e:
        print(f"❌ gRPC Registration Failed: {e}")
        exit(1) # Fail fast so the Runner knows

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    main()