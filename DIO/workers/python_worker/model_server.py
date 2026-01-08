import tiktoken
import dio_pb2
import dio_pb2_grpc

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

        print(f"[Worker] Tokens used: {total_tokens}")

        return dio_pb2.InferenceResponse(
            output=model_output.encode('utf-8'),
            latency_ms=150.5,
            tokens_used=total_tokens,
            context_full=context_full
        )