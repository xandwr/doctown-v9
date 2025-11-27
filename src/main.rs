use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::parameters::FormatType;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() {
    let ollama = Ollama::new("http://localhost".to_string(), 11434);

    let model = "qwen3:8b";
    let prompt = "Why is the sky blue?";
    let format = FormatType::Json;

    // Enable thinking + streaming
    let req = GenerationRequest::new(model.to_string(), prompt.to_string())
        .think(true)
        .format(format);

    let mut stream = ollama
        .generate_stream(req)
        .await
        .expect("Failed to start stream");

    let mut thinking_output = String::new();
    let mut response_output = String::new();

    while let Some(chunk) = stream.next().await {
        let responses = chunk.expect("Chunk error");

        for resp in responses {
            // Accumulate thinking JSON
            if let Some(thinking) = resp.thinking {
                thinking_output.push_str(&thinking);
            }

            // Accumulate response text
            if !resp.response.is_empty() {
                response_output.push_str(&resp.response);
            }
        }
    }

    // Determine which output to use (prefer thinking if available, otherwise response)
    let final_output = if !thinking_output.is_empty() {
        thinking_output
    } else if !response_output.is_empty() {
        response_output
    } else {
        String::new()
    };

    // Output result or error
    if !final_output.is_empty() {
        println!("{}", final_output);
    } else {
        eprintln!("Error: No output received from model");
        std::process::exit(1);
    }
}
