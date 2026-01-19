# Use a slim Python image for efficiency
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
# Explicitly setting versions to avoid the conflicts you saw earlier
RUN pip install --default-timeout=1000 --no-cache-dir \
    grpcio==1.70.0 \
    grpcio-tools==1.70.0 \
    tiktoken

# Copy worker source code
COPY workers/python_worker/ .
# Copy proto definitions
COPY api/proto/ api/proto/

# Regenerate Python Protobuf code inside the container to ensure version compatibility
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. api/proto/dio.proto

# Expose the gRPC port (Matches model_server.py)
EXPOSE 50053

# Run the worker
CMD ["python", "model_server.py"]