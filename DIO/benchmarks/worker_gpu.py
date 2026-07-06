import argparse
import os
import sys
import time
import threading
import grpc
from concurrent import futures
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import dio_pb2 as pb
import dio_pb2_grpc as pb_grpc
from mock_latency_model import MockLatencySimulator, resolve_profile


def gpu_vram_mb(fallback: int = 24000) -> int:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    return fallback


def load_model(is_mock=False, model_id="meta-llama/Llama-3.2-1B-Instruct"):
    if is_mock:
        return None, None

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device}...", flush=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"Model loaded on {device}.", flush=True)
        return tokenizer, model
    except Exception as e:
        print(f"LOAD FAILURE: {e}", flush=True)
        return None, None


class InferenceWorker(pb_grpc.InferenceWorkerServicer):
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.mock_sim = None
        self._ready = threading.Event()
        self._load_error = None

        if args.mock:
            profile = resolve_profile(
                profile_name=args.latency_profile or None,
                profile_role=args.profile_role or "slow",
                latency_mult=args.latency_mult,
            )
            self.mock_sim = MockLatencySimulator(profile, seed=args.mock_seed)
            print(
                f"Mock worker profile={profile.name} gpu={profile.gpu} "
                f"decode_slope={profile.decode_slope_ms_per_token}ms/tok",
                flush=True,
            )
            self._ready.set()
        else:
            threading.Thread(target=self._load_model_bg, daemon=True).start()

    def _load_model_bg(self):
        try:
            tokenizer, model = load_model(False, self.args.model_id)
            if model is None:
                profile = resolve_profile(latency_mult=self.args.latency_mult)
                self.mock_sim = MockLatencySimulator(profile, seed=self.args.mock_seed)
                print("Falling back to MOCK after load failure.", flush=True)
            else:
                self.tokenizer = tokenizer
                self.model = model
        except Exception as e:
            self._load_error = str(e)
            print(f"Background load error: {e}", flush=True)
        finally:
            self._ready.set()

    def Predict(self, request, context):
        if not self._ready.wait(timeout=600):
            context.abort(grpc.StatusCode.UNAVAILABLE, "model still loading (timeout)")

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

        if self.model is None:
            context.abort(
                grpc.StatusCode.INTERNAL,
                self._load_error or "model not available",
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


def register_worker(stub, args, worker_address: str, vram_mb: int) -> bool:
    resp = stub.RegisterWorker(
        pb.RegisterRequest(
            worker_id=args.worker_id,
            address=worker_address,
            tier=args.tier,
            vram_gb=vram_mb,
        )
    )
    return resp.success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--port", type=int, default=50060)
    parser.add_argument("--tier", default="large")
    parser.add_argument("--vram", type=int, default=0, help="VRAM MB to report (0=auto-detect GPU)")
    parser.add_argument("--latency-mult", type=float, default=1.0)
    parser.add_argument("--latency-profile", type=str, default="")
    parser.add_argument("--profile-role", type=str, default="slow", choices=["fast", "slow"])
    parser.add_argument("--mock-seed", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--manager-addr", type=str, default="localhost:50055")
    args = parser.parse_args()

    vram_mb = args.vram if args.vram > 0 else gpu_vram_mb()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker_servicer = InferenceWorker(args)
    pb_grpc.add_InferenceWorkerServicer_to_server(worker_servicer, server)
    server.add_insecure_port(f"127.0.0.1:{args.port}")
    server.start()

    worker_address = f"localhost:{args.port}"
    print(f"Worker {args.worker_id} listening on {worker_address}", flush=True)

    try:
        with grpc.insecure_channel(args.manager_addr) as channel:
            stub = pb_grpc.OrchestratorStub(channel)
            if not register_worker(stub, args, worker_address, vram_mb):
                print("Manager rejected registration.", flush=True)
                sys.exit(1)
            print("Successfully registered with manager (model may still be loading).", flush=True)
    except Exception as e:
        print(f"gRPC registration failed: {e}", flush=True)
        sys.exit(1)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()