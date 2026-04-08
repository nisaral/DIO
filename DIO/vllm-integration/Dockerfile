# ============================================================
# DIO v3 — vLLM Worker Proxy Dockerfile
# ============================================================
# This builds the Python sidecar that bridges DIO's gRPC
# protocol with vLLM's OpenAI HTTP API.
#
# The vLLM engine itself runs in a SEPARATE container (see
# docker-compose.yaml) using the official vllm/vllm-openai image.
# ============================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install Python dependencies
COPY vllm-integration/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the DIO protobuf definitions
COPY api/proto/ ./api/proto/
COPY api/__init__.py ./api/__init__.py

# Copy the vLLM integration code
COPY vllm-integration/ ./vllm-integration/

# Expose the gRPC port for DIO Manager → Proxy communication
EXPOSE 50060

# Default: start the worker proxy
# Override via docker-compose command/env vars
ENTRYPOINT ["python", "vllm-integration/worker_proxy.py"]
CMD ["--worker-id", "vllm-worker-0", "--port", "50060"]
