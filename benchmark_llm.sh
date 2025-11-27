#!/bin/bash
# Benchmark script to compare Ollama vs vLLM performance

set -e

echo "🏁 LLM Backend Benchmark"
echo "========================"
echo ""

# Build the project first
echo "📦 Building project..."
cargo build --release
echo ""

# Test with Ollama
echo "🦙 Testing with Ollama..."
echo "-------------------------"
time cargo run --release 2>&1 | tee ollama_run.log
echo ""
echo "✅ Ollama run complete"
echo ""

# Extract timing from logs
OLLAMA_LLM_TIME=$(grep "LLM Generation:" ollama_run.log | awk '{print $3}')
OLLAMA_TOTAL_TIME=$(grep "Total Pipeline:" ollama_run.log | awk '{print $3}')

echo ""
echo "Waiting 5 seconds before vLLM test..."
sleep 5
echo ""

# Test with vLLM
echo "🚀 Testing with vLLM..."
echo "----------------------"
echo "⚠️  Make sure vLLM server is running on localhost:8000"
echo "   Run: ./run_vllm_server.sh in another terminal"
echo ""
read -p "Press Enter when vLLM server is ready..."

VLLM_URL=http://localhost:8000 \
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct \
time cargo run --release 2>&1 | tee vllm_run.log

echo ""
echo "✅ vLLM run complete"
echo ""

# Extract timing from logs
VLLM_LLM_TIME=$(grep "LLM Generation:" vllm_run.log | awk '{print $3}')
VLLM_TOTAL_TIME=$(grep "Total Pipeline:" vllm_run.log | awk '{print $3}')

# Display comparison
echo ""
echo "📊 Performance Comparison"
echo "========================="
echo ""
echo "LLM Generation Time:"
echo "  Ollama: $OLLAMA_LLM_TIME"
echo "  vLLM:   $VLLM_LLM_TIME"
echo ""
echo "Total Pipeline Time:"
echo "  Ollama: $OLLAMA_TOTAL_TIME"
echo "  vLLM:   $VLLM_TOTAL_TIME"
echo ""
echo "Results saved to:"
echo "  - ollama_run.log"
echo "  - vllm_run.log"
