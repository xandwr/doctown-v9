#!/bin/bash
# Run doctown with CUDA diagnostics enabled

echo "🔍 Running with CUDA diagnostics..."
echo ""

# Enable ONNX Runtime verbose logging
export ORT_LOG_SEVERITY_LEVEL=1
export ORT_TENSORRT_ENGINE_CACHE_ENABLE=0

# Show CUDA information if available
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
fi

# Run the program
cargo run --release
