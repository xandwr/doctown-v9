#!/bin/bash
# Quick script to run doctown with vLLM backend

set -e

# Check if vLLM server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ vLLM server is not running on localhost:8000"
    echo ""
    echo "Start it with:"
    echo "  ./run_vllm_server.sh"
    echo ""
    echo "Or manually:"
    echo "  vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000"
    exit 1
fi

echo "✅ vLLM server detected"
echo ""

# Set defaults
export VLLM_URL="${VLLM_URL:-http://localhost:8000}"
export VLLM_MODEL="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

echo "🚀 Running doctown with vLLM backend"
echo "   URL: $VLLM_URL"
echo "   Model: $VLLM_MODEL"
echo ""

cargo run --release
