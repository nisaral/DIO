#!/bin/bash
set -e

echo "Setting up DIO on Lightning AI..."

# Detect repo root
if [[ -f "go.mod" ]]; then
  ROOT="."
elif [[ -f "DIO/go.mod" ]]; then
  ROOT="DIO"
else
  echo "Run from repo root or DIO/"; exit 1
fi
cd "$ROOT"

if ! command -v go &> /dev/null; then
    echo "Installing Go..."
    wget -q https://go.dev/dl/go1.22.5.linux-amd64.tar.gz
    sudo tar -C /usr/local -xzf go1.22.5.linux-amd64.tar.gz
    export PATH=$PATH:/usr/local/go/bin
    echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
fi

pip install -q locust pandas transformers torch requests numpy matplotlib grpcio grpcio-tools accelerate

go build -o dio-manager ./cmd/manager/main.go

if [[ ! -f benchmarks/data/sharegpt.jsonl ]]; then
  python3 benchmarks/prepare_data.py || echo "Dataset prep failed — check HuggingFace access"
fi

echo ""
echo "HuggingFace: huggingface-cli login"
echo "Run: bash benchmarks/run_lightning_budget.sh"