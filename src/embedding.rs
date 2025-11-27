use anyhow::{Context, Result};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;

/// Embedding model for generating vector representations of text
pub struct EmbeddingModel {
    session: Session,
    dims: usize,
}

impl EmbeddingModel {
    /// Load an embedding model from an ONNX file
    pub fn from_file<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()
            .context("Failed to create session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set optimization level")?
            .with_intra_threads(4)
            .context("Failed to set thread count")?
            .commit_from_file(&model_path)
            .context("Failed to load ONNX model")?;

        // For MiniLM-L6, the output dimension is 384
        // This should ideally be detected from the model metadata
        let dims = 384;

        Ok(Self { session, dims })
    }

    /// Get the embedding dimension
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Embed a single text string
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        // Tokenize the text
        let tokens = tokenize_simple(text);

        // Convert to input IDs (simple word hashing for now)
        let input_ids = tokens_to_ids(&tokens);

        // Create attention mask (all ones)
        let attention_mask: Vec<i64> = vec![1; input_ids.len()];

        // Pad or truncate to max length (512 for BERT-based models)
        let max_len = 512;
        let mut padded_input_ids = input_ids;
        let mut padded_attention_mask = attention_mask;

        if padded_input_ids.len() > max_len {
            padded_input_ids.truncate(max_len);
            padded_attention_mask.truncate(max_len);
        } else {
            let padding = max_len - padded_input_ids.len();
            padded_input_ids.extend(vec![0; padding]);
            padded_attention_mask.extend(vec![0; padding]);
        }

        // Clone the attention mask for later use
        let attention_mask_copy = padded_attention_mask.clone();

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(([1, max_len], padded_input_ids))?,
                "attention_mask" => ort::value::Tensor::from_array(([1, max_len], padded_attention_mask))?
            ])
            .context("Failed to run model inference")?;

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

/// Simple tokenization (word-level)
/// In production, use a proper tokenizer like `tokenizers` crate
fn tokenize_simple(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Convert tokens to IDs using a simple hash
/// In production, use a proper vocabulary from the tokenizer
fn tokens_to_ids(tokens: &[String]) -> Vec<i64> {
    tokens
        .iter()
        .map(|token| {
            // Simple hash-based ID assignment
            // This is a placeholder - real models need proper vocab
            let hash = token
                .bytes()
                .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
            (hash % 30000 + 1000) as i64 // Keep in reasonable range
        })
        .collect()
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
    fn test_tokenize() {
        let tokens = tokenize_simple("Hello, world! This is a test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_normalize() {
        let vec = vec![3.0, 4.0];
        let normalized = normalize(&vec);
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
