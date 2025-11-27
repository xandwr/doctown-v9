use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::parameters::FormatType;
use tokio::io::{self, AsyncWriteExt};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() {
    let ollama = Ollama::new("http://localhost".to_string(), 11434);

    let model = "qwen3:4b".to_string();
    let prompt = "Why is the sky blue?".to_string();
    let format = FormatType::Json;

    // Enable thinking + streaming
    let req = GenerationRequest::new(model, prompt)
        .think(true)
        .format(format);

    let mut stream = ollama
        .generate_stream(req)
        .await
        .expect("Failed to start stream");

    let mut stdout = io::stdout();
    // Buffer for assembling streaming JSON when `think(true)` is enabled
    let mut json_buf = String::new();

    while let Some(chunk) = stream.next().await {
        let responses = chunk.expect("Chunk error");

        for resp in responses {
            // Prefer `resp.response` when present (non-streaming / final text).
            if !resp.response.is_empty() {
                stdout.write_all(resp.response.as_bytes()).await.unwrap();
                stdout.flush().await.unwrap();
                continue;
            }

            // If JSON is streaming via `resp.thinking`, accumulate it.
            if let Some(thinking) = resp.thinking {
                json_buf.push_str(&thinking);
            }

            // When the response is marked `done`, emit the assembled JSON as a
            // single JSON-only payload to stdout and clear the buffer.
            if resp.done {
                if !json_buf.is_empty() {
                    stdout.write_all(json_buf.as_bytes()).await.unwrap();
                    stdout.write_all(b"\n").await.unwrap();
                    stdout.flush().await.unwrap();
                    json_buf.clear();
                }
            }
        }
    }
}
