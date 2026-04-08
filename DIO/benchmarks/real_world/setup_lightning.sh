#!/bin/bash
set -e

echo "⚡ Setting up DIO on Lightning AI..."

# 1. Install Go (Lightning AI usually doesn't have it)
if ! command -v go &> /dev/null; then
    echo "   Installing Go..."
    wget https://go.dev/dl/go1.21.6.linux-amd64.tar.gz
    sudo tar -C /usr/local -xzf go1.21.6.linux-amd64.tar.gz
    export PATH=$PATH:/usr/local/go/bin
    echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
fi

# 2. Install Python Dependencies
echo "   Installing Python libs..."
pip install locust pandas transformers torch requests numpy

# 3. Login to HuggingFace (Required for Llama 3)
echo "   ⚠️  Please login to HuggingFace to download Llama 3:"
huggingface-cli login

echo "✅ Setup Complete. Run: python benchmarks/run_cloud_suite.py"