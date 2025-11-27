#!/bin/bash
# Script to run vLLM server with optimal settings for doctown

set -e

# Default values
MODEL="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${VLLM_PORT:-8000}"
GPU_MEMORY="${VLLM_GPU_MEMORY:-0.9}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

echo "🚀 Starting vLLM server..."
echo "   Model: $MODEL"
echo "   Port: $PORT"
echo "   GPU Memory Utilization: $GPU_MEMORY"
echo "   Max Model Length: $MAX_MODEL_LEN"
echo ""

# Check if vllm is installed
if ! command -v vllm &> /dev/null; then
    echo "❌ vllm not found. Installing..."
    pip install vllm
fi

# Run vLLM server
vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype auto \
    --enable-prefix-caching \
    --disable-log-requests

echo ""
echo "✅ vLLM server started on http://localhost:$PORT"
echo "   Set VLLM_URL=http://localhost:$PORT to use with doctown"
