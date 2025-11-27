use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio_stream::StreamExt;

#[derive(Debug, Clone)]
pub enum LLMBackend {
    Ollama {
        host: String,
        port: u16,
        model: String,
    },
    VLlm {
        base_url: String,
        model: String,
        api_key: Option<String>,
    },
}

#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn generate_stream(
        &self,
        prompt: String,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>>>;
}

// Ollama implementation
pub struct OllamaProvider {
    ollama: ollama_rs::Ollama,
    model: String,
}

impl OllamaProvider {
    pub fn new(host: String, port: u16, model: String) -> Self {
        Self {
            ollama: ollama_rs::Ollama::new(host, port),
            model,
        }
    }
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    async fn generate_stream(
        &self,
        prompt: String,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>>> {
        use ollama_rs::generation::completion::request::GenerationRequest;

        let req = GenerationRequest::new(self.model.clone(), prompt);
        let stream = self.ollama.generate_stream(req).await?;

        let mapped_stream = stream.map(|chunk_result| {
            chunk_result
                .map_err(|e| anyhow::anyhow!("Ollama error: {}", e))
                .map(|responses| {
                    responses
                        .into_iter()
                        .map(|resp| resp.response)
                        .collect::<Vec<_>>()
                        .join("")
                })
        });

        Ok(Box::pin(mapped_stream))
    }
}

// vLLM OpenAI-compatible API implementation
pub struct VLlmProvider {
    client: Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

#[derive(Serialize)]
struct VLlmRequest {
    model: String,
    prompt: String,
    stream: bool,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct VLlmStreamResponse {
    choices: Vec<VLlmChoice>,
}

#[derive(Deserialize)]
struct VLlmChoice {
    text: String,
}

impl VLlmProvider {
    pub fn new(base_url: String, model: String, api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model,
            api_key,
        }
    }
}

#[async_trait]
impl LLMProvider for VLlmProvider {
    async fn generate_stream(
        &self,
        prompt: String,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>>> {
        let url = format!("{}/v1/completions", self.base_url.trim_end_matches('/'));

        let request = VLlmRequest {
            model: self.model.clone(),
            prompt,
            stream: true,
            max_tokens: Some(4096),
            temperature: Some(0.7),
        };

        let mut req_builder = self.client.post(&url).json(&request);

        if let Some(api_key) = &self.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req_builder.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("vLLM API error {}: {}", status, text);
        }

        let stream = response.bytes_stream();

        let mapped_stream = stream.map(|chunk_result| {
            chunk_result
                .map_err(|e| anyhow::anyhow!("Stream error: {}", e))
                .and_then(|bytes| {
                    let text = String::from_utf8_lossy(&bytes);

                    // Parse SSE format: "data: {...}\n\n"
                    let mut result = String::new();
                    for line in text.lines() {
                        if let Some(json_str) = line.strip_prefix("data: ") {
                            if json_str.trim() == "[DONE]" {
                                continue;
                            }

                            match serde_json::from_str::<VLlmStreamResponse>(json_str) {
                                Ok(resp) => {
                                    for choice in resp.choices {
                                        result.push_str(&choice.text);
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Warning: Failed to parse vLLM response: {}", e);
                                }
                            }
                        }
                    }

                    Ok(result)
                })
        });

        Ok(Box::pin(mapped_stream))
    }
}

pub fn create_provider(backend: LLMBackend) -> Box<dyn LLMProvider> {
    match backend {
        LLMBackend::Ollama { host, port, model } => {
            Box::new(OllamaProvider::new(host, port, model))
        }
        LLMBackend::VLlm {
            base_url,
            model,
            api_key,
        } => Box::new(VLlmProvider::new(base_url, model, api_key)),
    }
}
