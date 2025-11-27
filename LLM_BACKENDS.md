# LLM Backend Support

Doctown now supports multiple LLM backends for flexible deployment and performance optimization.

## Supported Backends

### 1. Ollama (Default)
- **Pros**: Easy to set up, good for local development
- **Cons**: Slower inference, limited batch optimization
- **Setup**: Install Ollama and pull a model
```bash
ollama pull qwen3:4b
cargo run --release
```

### 2. vLLM (High Performance)
- **Pros**: 10-24x faster, better GPU utilization, production-ready
- **Cons**: Requires separate server process
- **Setup**: See [VLLM_SETUP.md](VLLM_SETUP.md)
```bash
# Terminal 1: Start vLLM server
./run_vllm_server.sh

# Terminal 2: Run doctown with vLLM
VLLM_URL=http://localhost:8000 cargo run --release
```

## Architecture

The LLM abstraction is implemented in `src/llm.rs` with a trait-based design:

```rust
pub trait LLMProvider {
    async fn generate_stream(&self, prompt: String) 
        -> Result<Stream<Item = Result<String>>>;
}
```

This allows easy addition of new backends (e.g., Anthropic, OpenAI, local GGUF, etc.) by implementing the trait.

### Current Implementations

1. **OllamaProvider**: Uses `ollama-rs` client library
2. **VLlmProvider**: Uses OpenAI-compatible HTTP streaming API

## Environment Variables

Configure the LLM backend using environment variables:

### Ollama (default if no VLLM_URL set)
```bash
# Uses default settings: http://localhost:11434 with qwen3:4b
cargo run --release
```

### vLLM
```bash
export VLLM_URL=http://localhost:8000
export VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct  # optional
export VLLM_API_KEY=your-api-key  # optional
cargo run --release
```

## Performance Benchmarking

Use the provided benchmark script to compare backends:

```bash
./benchmark_llm.sh
```

This will:
1. Run doctown with Ollama
2. Run doctown with vLLM
3. Compare timing results

Expected speedup: **3-10x faster** LLM generation with vLLM.

## Adding New Backends

To add a new backend:

1. **Implement the trait** in `src/llm.rs`:
```rust
pub struct MyBackendProvider { /* ... */ }

#[async_trait]
impl LLMProvider for MyBackendProvider {
    async fn generate_stream(&self, prompt: String) 
        -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>>> 
    {
        // Your implementation
    }
}
```

2. **Add to enum** in `src/llm.rs`:
```rust
pub enum LLMBackend {
    Ollama { /* ... */ },
    VLlm { /* ... */ },
    MyBackend { /* fields */ },
}
```

3. **Update factory** in `create_provider()`:
```rust
pub fn create_provider(backend: LLMBackend) -> Box<dyn LLMProvider> {
    match backend {
        // ...
        LLMBackend::MyBackend { /* ... */ } => Box::new(MyBackendProvider::new(/* ... */)),
    }
}
```

4. **Update main.rs** to detect/configure your backend

## Implementation Notes

### Streaming Support
Both backends implement streaming to provide real-time feedback during generation. This is essential for long documentation generation tasks.

### Error Handling
The trait uses `anyhow::Result` for flexible error handling across different backend implementations.

### Token Management
Context length and token limits are managed by the backend servers (Ollama Modelfile or vLLM flags), not per-request in doctown.

### Async/Await
All LLM calls are async to prevent blocking the main thread during generation.

## Future Enhancements

Potential additions:
- [ ] Direct GGUF support (via llama.cpp bindings)
- [ ] OpenAI API support
- [ ] Anthropic Claude support
- [ ] Azure OpenAI support
- [ ] Local transformers.rs support
- [ ] Batch processing optimization
- [ ] Response caching
- [ ] Multi-model ensembles
