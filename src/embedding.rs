use anyhow::{Context, Result};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;
use tokenizers::Tokenizer;

/// Check if CUDA is available for ONNX Runtime
fn check_cuda_availability() -> bool {
    // Check if CUDA libraries are available
    // This is a simple check - ORT will do the real validation
    std::env::var("CUDA_PATH").is_ok() || std::path::Path::new("/usr/local/cuda").exists()
}

/// Embedding model for generating vector representations of text
pub struct EmbeddingModel {
    session: Session,
    tokenizer: Tokenizer,
    dims: usize,
}

impl EmbeddingModel {
    /// Load an embedding model from an ONNX file
    pub fn from_file<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        println!("🔧 Configuring embedding model with CUDA support...");

        // Check CUDA availability
        let cuda_available = check_cuda_availability();
        if cuda_available {
            println!("  CUDA environment detected");
        } else {
            println!("  No CUDA environment detected, will use CPU");
        }

        // Try to use CUDA if available, fall back to CPU
        let session = Session::builder()
            .context("Failed to create session builder")?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build(),
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])
            .context("Failed to set execution providers")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set optimization level")?
            .with_intra_threads(4)
            .context("Failed to set thread count")?
            .commit_from_file(&model_path)
            .context("Failed to load ONNX model")?;

        println!("✓ Embedding model loaded (CUDA enabled if available)");

        // Load tokenizer from the same directory as the model
        let model_dir = model_path
            .as_ref()
            .parent()
            .context("Failed to get model directory")?;
        let tokenizer_path = model_dir.join("tokenizer.json");

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // For MiniLM-L6, the output dimension is 384
        let dims = 384;

        Ok(Self {
            session,
            tokenizer,
            dims,
        })
    }

    /// Get the embedding dimension
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Embed a single text string
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        // Log first inference to confirm execution provider
        static FIRST_RUN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        if FIRST_RUN.swap(false, std::sync::atomic::Ordering::Relaxed) {
            println!("  Running first inference (execution provider will be initialized)...");
        }

        // Tokenize the text using the proper tokenizer
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize text: {}", e))?;

        // Get input IDs and attention mask
        // Clamp token IDs to valid vocab range for the model (0-30521 for BERT base)
        // This handles mismatches between tokenizer and model vocabularies
        const MAX_VOCAB_ID: i64 = 30521;
        let input_ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .map(|&id| {
                let id_i64 = id as i64;
                if id_i64 > MAX_VOCAB_ID {
                    // Replace out-of-vocab tokens with [UNK] token (typically ID 100 in BERT)
                    100_i64
                } else {
                    id_i64
                }
            })
            .collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

        // Pad or truncate to max length (512 for BERT-based models)
        let max_len = 512;
        let mut padded_input_ids = input_ids;
        let mut padded_attention_mask = attention_mask;
        let mut padded_token_type_ids = token_type_ids;

        if padded_input_ids.len() > max_len {
            padded_input_ids.truncate(max_len);
            padded_attention_mask.truncate(max_len);
            padded_token_type_ids.truncate(max_len);
        } else {
            let padding = max_len - padded_input_ids.len();
            padded_input_ids.extend(vec![0; padding]);
            padded_attention_mask.extend(vec![0; padding]);
            padded_token_type_ids.extend(vec![0; padding]);
        }

        // Clone the attention mask for later use
        let attention_mask_copy = padded_attention_mask.clone();

        // Run inference - create tensors properly
        use ort::value::Tensor;
        let input_ids_tensor = Tensor::from_array((
            vec![1_i64, max_len as i64],
            padded_input_ids.into_boxed_slice(),
        ))
        .context("Failed to create input_ids tensor")?;
        let attention_mask_tensor = Tensor::from_array((
            vec![1_i64, max_len as i64],
            padded_attention_mask.into_boxed_slice(),
        ))
        .context("Failed to create attention_mask tensor")?;
        let token_type_ids_tensor = Tensor::from_array((
            vec![1_i64, max_len as i64],
            padded_token_type_ids.into_boxed_slice(),
        ))
        .context("Failed to create token_type_ids tensor")?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor
            ])
            .map_err(|e| anyhow::anyhow!("Failed to run model inference: {}", e))?;

        // Extract the output tensor
        // The output is typically the last hidden state or pooled output
        let embedding = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract embedding tensor")?;

        // Mean pooling over the sequence dimension
        let embedding_data = embedding.1;

        // Perform mean pooling to get sentence embedding
        let pooled = mean_pooling(embedding_data, &attention_mask_copy, self.dims);

        // Normalize the embedding
        let normalized = normalize(&pooled);

        Ok(normalized)
    }

    /// Embed multiple texts in batch
    #[allow(dead_code)]
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // For simplicity, process one at a time
        // A real implementation would batch these together
        texts.iter().map(|text| self.embed_text(text)).collect()
    }
}

/// Mean pooling over sequence dimension
fn mean_pooling(embeddings: &[f32], attention_mask: &[i64], dims: usize) -> Vec<f32> {
    let mut pooled = vec![0.0; dims];
    let mut sum_mask = 0.0;

    for (i, &mask_val) in attention_mask.iter().enumerate() {
        if mask_val == 1 {
            for d in 0..dims {
                pooled[d] += embeddings[i * dims + d];
            }
            sum_mask += 1.0;
        }
    }

    if sum_mask > 0.0 {
        for d in 0..dims {
            pooled[d] /= sum_mask;
        }
    }

    pooled
}

/// Normalize a vector to unit length
fn normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let vec = vec![3.0, 4.0];
        let normalized = normalize(&vec);
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
