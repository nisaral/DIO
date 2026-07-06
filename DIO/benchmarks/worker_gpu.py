import argparse
import os
import sys
import time
import grpc
from concurrent import futures
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import dio_pb2 as pb
import dio_pb2_grpc as pb_grpc
from mock_latency_model import MockLatencySimulator, resolve_profile


class InferenceWorker(pb_grpc.InferenceWorkerServicer):
    def __init__(self, args, tokenizer, model):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.mock_sim = None
        if args.mock or model is None:
            profile = resolve_profile(
                profile_name=args.latency_profile or None,
                profile_role=args.profile_role or "slow",
                latency_mult=args.latency_mult,
            )
            self.mock_sim = MockLatencySimulator(profile, seed=args.mock_seed)
            print(
                f"Mock worker profile={profile.name} gpu={profile.gpu} "
                f"decode_slope={profile.decode_slope_ms_per_token}ms/tok "
                f"jitter={profile.jitter_pct} thermal={profile.thermal.enabled}"
            )

    def Predict(self, request, context):
        try:
            prompt = request.data.decode("utf-8")
        except AttributeError:
            prompt = request.Data.decode("utf-8")

        output_len = 128
        start_time = time.perf_counter()

        if self.mock_sim is not None:
            result = self.mock_sim.execute(prompt, output_len)
            return pb.InferenceResponse(
                latency_ms=result.total_latency_ms,
                tokens_used=result.tokens_generated,
                ttft_ms=result.ttft_ms,
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            _ = self.model.generate(
                **inputs, max_new_tokens=1, pad_token_id=self.tokenizer.eos_token_id
            )
            ttft = (time.perf_counter() - start_time) * 1000
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=output_len,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        tokens_generated = len(outputs[0]) - len(inputs[0])
        total_latency = (time.perf_counter() - start_time) * 1000

        if self.args.latency_mult > 1.0:
            extra = (total_latency * self.args.latency_mult) - total_latency
            time.sleep(extra / 1000.0)
            total_latency *= self.args.latency_mult
            ttft *= self.args.latency_mult

        return pb.InferenceResponse(
            latency_ms=total_latency,
            tokens_used=int(tokens_generated),
            ttft_ms=ttft,
        )


def load_model(is_mock=False, model_id="meta-llama/Llama-3.2-1B-Instruct"):
    if is_mock:
        return None, None

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device.upper()}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        return tokenizer, model
    except Exception as e:
        print(f"LOAD FAILURE: {e}. Falling back to MOCK.")
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--port", type=int, default=50060)
    parser.add_argument("--tier", default="large")
    parser.add_argument("--vram", type=int, default=12000)
    parser.add_argument("--latency-mult", type=float, default=1.0)
    parser.add_argument(
        "--latency-profile",
        type=str,
        default="",
        help="Profile name or pairing (e.g. t4_vs_a100, t4_emulated_slow)",
    )
    parser.add_argument(
        "--profile-role",
        type=str,
        default="slow",
        choices=["fast", "slow"],
        help="Role when --latency-profile is a pairing name",
    )
    parser.add_argument("--mock-seed", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--manager-addr", type=str, default="localhost:50055")
    args = parser.parse_args()

    tokenizer, model = load_model(args.mock, args.model_id)
    if args.mock and model is not None:
        model = None
        tokenizer = None

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_InferenceWorkerServicer_to_server(
        InferenceWorker(args, tokenizer, model), server
    )
    server.add_insecure_port(f"127.0.0.1:{args.port}")
    server.start()

    worker_address = f"localhost:{args.port}"
    print(f"Worker {args.worker_id} listening on {worker_address}")

    try:
        with grpc.insecure_channel(args.manager_addr) as channel:
            stub = pb_grpc.OrchestratorStub(channel)
            resp = stub.RegisterWorker(
                pb.RegisterRequest(
                    worker_id=args.worker_id,
                    address=worker_address,
                    tier=args.tier,
                    vram_gb=args.vram,
                )
            )
            if not resp.success:
                print("Manager rejected registration.")
                sys.exit(1)
            print("Successfully registered with manager.")
    except Exception as e:
        print(f"gRPC registration failed: {e}")
        sys.exit(1)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()