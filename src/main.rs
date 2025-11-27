use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::parameters::FormatType;
use std::io::{self, Write};
use std::env;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() {
    let ollama = Ollama::new("http://localhost".to_string(), 11434);

    // Determine verbosity from CLI args. By default streaming logs are hidden.
    let args: Vec<String> = env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");

    let model = "qwen3:4b";
    let prompt = "Why is the sky blue?";
    let format = FormatType::Json;

    println!("│ Ollama Streaming Response");
    println!("│ Model: {}", model);
    println!("│ Prompt: {}", prompt);

    // Enable thinking + streaming
    let req = GenerationRequest::new(model.to_string(), prompt.to_string())
        .think(true)
        .format(format);

    let mut stream = ollama
        .generate_stream(req)
        .await
        .expect("Failed to start stream");

    let mut json_buf = String::new();
    let mut live_buf = String::new();
    let mut chunk_count = 0;

    while let Some(chunk) = stream.next().await {
        let responses = chunk.expect("Chunk error");

        for resp in responses {
            // Handle non-streaming final text
            if !resp.response.is_empty() {
                // Clear the live preview line
                print!("\r\x1b[K");
                io::stdout().flush().unwrap();

                println!("\rRESPONSE: {}", resp.response);
                continue;
            }

            // Handle streaming thinking JSON
            if let Some(thinking) = resp.thinking {
                json_buf.push_str(&thinking);
                live_buf.push_str(&thinking);
                chunk_count += 1;

                // Create a clean preview with truncation
                let preview = if live_buf.len() > 80 {
                    format!("...{}", &live_buf[live_buf.len().saturating_sub(80)..])
                } else {
                    live_buf.clone()
                };

                // Update in-place using carriage return (only when verbose)
                if verbose {
                    print!("\r\x1b[KStreaming [{} chunks] {}", chunk_count, preview);
                    io::stdout().flush().unwrap();
                }
            }

            // Handle completion
            if resp.done {
                if !json_buf.is_empty() {
                    // Clear the live preview line
                    print!("\r\x1b[K");
                    io::stdout().flush().unwrap();

                    if verbose {
                        println!("│ ✓ Complete - {} chunks received", chunk_count);
                    }

                    println!("\nFinal JSON Output:");
                    println!("{}", json_buf);

                    json_buf.clear();
                    live_buf.clear();
                }
            }
        }
    }

    println!("\n✨ Stream completed!");
}
