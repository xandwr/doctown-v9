# vLLM Integration Guide

This document explains how to use vLLM as the LLM backend for doctown instead of Ollama.

## Why vLLM?

vLLM offers several advantages:
- **Higher throughput**: Optimized for batch processing and continuous batching
- **Better GPU utilization**: PagedAttention algorithm for efficient memory management
- **Faster inference**: Up to 24x faster than standard implementations
- **Compatible with HuggingFace models**: Easy to use with any Llama, Mistral, Qwen, etc.

## Quick Start

### Option 1: Using vLLM Python Server (Recommended)

1. **Install vLLM**:
```bash
pip install vllm
```

2. **Start the vLLM server**:
```bash
./run_vllm_server.sh
```

Or manually:
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --enable-prefix-caching
```

3. **Run doctown with vLLM**:
```bash
VLLM_URL=http://localhost:8000 \
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct \
cargo run --release
```

### Option 2: Using candle-vllm (Rust Native)

For a pure Rust implementation (still experimental):

1. **Clone and build candle-vllm**:
```bash
git clone https://github.com/EricLBuehler/candle-vllm.git
cd candle-vllm

# For CUDA with flash attention
cargo build --release --features cuda,nccl,graph,flash-attn
```

2. **Run the server**:
```bash
./target/release/candle-vllm --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

3. **Connect doctown** (same as above):
```bash
VLLM_URL=http://localhost:8000 cargo run --release
```

## Environment Variables

- `VLLM_URL`: Base URL of the vLLM server (required to use vLLM)
- `VLLM_MODEL`: Model name (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `VLLM_API_KEY`: API key if authentication is enabled (optional)

## Performance Comparison

To benchmark Ollama vs vLLM:

1. **Run with Ollama** (default):
```bash
cargo run --release
```

2. **Run with vLLM**:
```bash
VLLM_URL=http://localhost:8000 cargo run --release
```

Check the timing summary at the end - the "LLM Generation" time should be significantly faster with vLLM.

## Recommended Models

For documentation generation, good options include:

- **Llama 3.1 8B Instruct**: Balanced performance and quality
- **Qwen 2.5 Coder 7B**: Specialized for code understanding
- **Mistral 7B Instruct**: Fast and efficient
- **DeepSeek Coder 6.7B**: Good for technical documentation

Example with Qwen:
```bash
VLLM_URL=http://localhost:8000 \
VLLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
cargo run --release
```

## Troubleshooting

### Server not responding
Check if vLLM server is running:
```bash
curl http://localhost:8000/health
```

### Out of memory errors
Reduce `--gpu-memory-utilization` or `--max-model-len`:
```bash
vllm serve <model> --gpu-memory-utilization 0.8 --max-model-len 4096
```

### Model not found
Make sure the model is downloaded:
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

## Implementation Details

The vLLM integration uses the OpenAI-compatible API that vLLM exposes by default. The implementation is in `src/llm.rs` with a provider abstraction that allows switching between:

- `OllamaProvider`: Uses ollama-rs client
- `VLlmProvider`: Uses HTTP streaming to vLLM's `/v1/completions` endpoint

Both providers implement the same `LLMProvider` trait, making it easy to add more backends in the future.
