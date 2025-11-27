use crate::chunking::{ChunkConfig, chunk_files};
use crate::docpack::{DocPack, Metadata, create_docpack};
use crate::embedding::EmbeddingModel;
use crate::ingest::IngestWorkspace;
use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use std::collections::HashMap;
use tokio_stream::StreamExt;

#[derive(Clone, Debug)]
pub struct DocGenConfig {
    pub host: String,
    pub port: u16,
    pub model: String,
    pub embedding_model_path: Option<String>,
}

impl Default for DocGenConfig {
    fn default() -> Self {
        Self {
            host: "http://localhost".into(),
            port: 11434,
            model: "qwen3:8b".into(),
            embedding_model_path: Some("embedding/minilm-l6/model.onnx".into()),
        }
    }
}

/// Result returned from the docgen stream.
#[derive(Debug)]
#[allow(dead_code)]
pub struct DocGenResult {
    pub docpack: DocPack,
    pub raw_llm_output: String,
    pub num_chunks: usize,
    pub num_embeddings: usize,
}

/// Generate a docgen output and populate a SQLite docpack
pub async fn run_docgen(
    config: DocGenConfig,
    workspace: &IngestWorkspace,
    repo_url: &str,
    project_name: &str,
) -> anyhow::Result<DocGenResult> {
    let ollama = Ollama::new(config.host, config.port);

    // Create the docpack database
    let (docpack, _path) = create_docpack(project_name)?;

    // Insert metadata
    let metadata = Metadata {
        docpack_version: "0.1.0".to_string(),
        project_name: project_name.to_string(),
        project_description: None,
        repo_url: Some(repo_url.to_string()),
        commit_sha: None,
        created_at: chrono::Utc::now().to_rfc3339(),
        generator_version: env!("CARGO_PKG_VERSION").to_string(),
        language: None,
        extra: None,
    };
    docpack.insert_metadata(&metadata)?;

    // Log build event
    docpack.add_build_event(
        &chrono::Utc::now().to_rfc3339(),
        "ingest_start",
        Some(&serde_json::json!({
            "file_count": workspace.files.len()
        })),
    )?;

    // Step 1: Insert all files into the database (Ingest + Index)
    let mut file_map = HashMap::new();
    for (path, contents) in &workspace.files {
        let line_count = if let Ok(text) = String::from_utf8(contents.clone()) {
            Some(text.lines().count())
        } else {
            None
        };

        let file_id = docpack.insert_file(path, None, contents, line_count, None)?;
        file_map.insert(path.clone(), (file_id, contents.clone()));
    }

    // Step 2: Chunk the files into semantic units
    println!("📐 Chunking files into semantic units...");
    let chunk_config = ChunkConfig::default();
    let chunks = chunk_files(&file_map, &chunk_config)?;
    println!("✓ Created {} chunks", chunks.len());

    // Insert chunks into database
    let mut chunk_ids = Vec::new();
    for chunk in &chunks {
        let chunk_id = docpack.insert_chunk(
            chunk.file_id,
            chunk.start_line,
            chunk.end_line,
            &chunk.text,
            chunk.symbol_hint.as_deref(),
            None,
        )?;
        chunk_ids.push(chunk_id);
    }

    docpack.add_build_event(
        &chrono::Utc::now().to_rfc3339(),
        "chunking_complete",
        Some(&serde_json::json!({
            "chunk_count": chunks.len()
        })),
    )?;

    // Step 3: Embed the chunks
    let num_embeddings = if let Some(model_path) = &config.embedding_model_path {
        println!("🔢 Loading embedding model...");
        match EmbeddingModel::from_file(model_path) {
            Ok(mut embedding_model) => {
                println!("✓ Model loaded, embedding {} chunks...", chunks.len());

                let mut embedded_count = 0;
                for (idx, chunk) in chunks.iter().enumerate() {
                    if idx % 100 == 0 && idx > 0 {
                        println!("  Embedded {}/{} chunks", idx, chunks.len());
                    }

                    match embedding_model.embed_text(&chunk.text) {
                        Ok(embedding) => {
                            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                            docpack.insert_embedding(chunk_ids[idx], &embedding, Some(norm))?;
                            embedded_count += 1;
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to embed chunk {}: {}", idx, e);
                        }
                    }
                }

                println!("✓ Embedded {} chunks", embedded_count);

                docpack.add_build_event(
                    &chrono::Utc::now().to_rfc3339(),
                    "embedding_complete",
                    Some(&serde_json::json!({
                        "embedding_count": embedded_count,
                        "embedding_dims": embedding_model.dims()
                    })),
                )?;

                embedded_count
            }
            Err(e) => {
                eprintln!("Warning: Could not load embedding model: {}", e);
                eprintln!(
                    "Skipping embedding step. Make sure the model exists at: {}",
                    model_path
                );
                0
            }
        }
    } else {
        println!("⚠ No embedding model configured, skipping embedding step");
        0
    };

    // Build comprehensive prompt for LLM
    let mut full_prompt = String::new();
    full_prompt.push_str("You are an expert code documentation generator.\n\n");
    full_prompt.push_str("Here is the codebase to document:\n\n");

    for (path, contents) in &workspace.files {
        full_prompt.push_str(&format!("=== File: {} ===\n", path));
        if let Ok(text) = String::from_utf8(contents.clone()) {
            full_prompt.push_str(&text);
        } else {
            full_prompt.push_str("[Binary file]\n");
        }
        full_prompt.push_str("\n\n");
    }

    full_prompt.push_str(
        r#"
Generate comprehensive documentation for this project. Focus on:

1. **Architecture Overview**: High-level system design and component interactions
2. **Key Modules**: Main functional areas and their responsibilities
3. **Important Symbols**: Critical functions, structs, traits, enums with their purpose
4. **Code Flow**: How the main operations work step-by-step
5. **Usage Examples**: Practical examples of how to use the code

Respond in a structured format with clear sections for each aspect.
"#,
    );

    let req = GenerationRequest::new(config.model.clone(), full_prompt).think(true);

    let mut stream = ollama.generate_stream(req).await?;

    let mut thinking_output = String::new();
    let mut response_output = String::new();

    while let Some(chunk) = stream.next().await {
        let responses = chunk?;

        for resp in responses {
            if let Some(t) = resp.thinking {
                thinking_output.push_str(&t);
            }

            if !resp.response.is_empty() {
                response_output.push_str(&resp.response);
            }
        }
    }

    // For now, store the LLM output as a single architecture doc node
    // In the future, we'll parse this into structured symbols/modules
    let node_id = docpack.insert_node(
        "architecture",
        Some("project_overview"),
        None,
        None,
        None,
        None,
        None,
        None,
    )?;

    let reasoning_json = if !thinking_output.is_empty() {
        Some(serde_json::json!({ "thinking": thinking_output }))
    } else {
        None
    };

    docpack.insert_doc(
        node_id,
        Some("Project Architecture and Documentation"),
        Some(&response_output),
        None,
        reasoning_json.as_ref(),
        None,
    )?;

    // Log completion event
    docpack.add_build_event(
        &chrono::Utc::now().to_rfc3339(),
        "docgen_complete",
        Some(&serde_json::json!({
            "llm_output_length": response_output.len(),
            "thinking_length": thinking_output.len()
        })),
    )?;

    Ok(DocGenResult {
        docpack,
        raw_llm_output: response_output,
        num_chunks: chunks.len(),
        num_embeddings,
    })
}
