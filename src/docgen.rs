use crate::chunking::{ChunkConfig, chunk_files};
use crate::clustering::{cluster_embeddings, get_cluster_sizes};
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
            model: "qwen3:4b".into(),
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

    // Step 4: Cluster the embeddings
    let num_clusters = if num_embeddings > 0 {
        println!("\n🔍 Clustering embeddings...");

        let embeddings = docpack.get_all_embeddings()?;
        let num_clusters = (embeddings.len() as f32 / 10.0).ceil() as usize;
        let num_clusters = num_clusters.clamp(3, 20); // Between 3-20 clusters

        let cluster_result = cluster_embeddings(&embeddings, num_clusters)?;
        println!("✓ Found {} clusters", cluster_result.num_clusters);

        // Insert clusters into database
        let cluster_sizes = get_cluster_sizes(&cluster_result);
        let mut cluster_id_map = HashMap::new();

        for cluster_idx in 0..cluster_result.num_clusters {
            let cluster_idx_i64 = cluster_idx as i64;
            let size = cluster_sizes.get(&cluster_idx_i64).copied().unwrap_or(0);

            let db_cluster_id =
                docpack.insert_cluster(Some(&format!("cluster_{}", cluster_idx)), size, None)?;
            cluster_id_map.insert(cluster_idx_i64, db_cluster_id);
        }

        // Insert cluster memberships
        for (chunk_id, cluster_idx) in &cluster_result.assignments {
            if let Some(&db_cluster_id) = cluster_id_map.get(cluster_idx) {
                docpack.insert_cluster_membership(*chunk_id, db_cluster_id)?;
            }
        }

        docpack.add_build_event(
            &chrono::Utc::now().to_rfc3339(),
            "clustering_complete",
            Some(&serde_json::json!({
                "num_clusters": cluster_result.num_clusters,
                "cluster_sizes": cluster_sizes
            })),
        )?;

        cluster_result.num_clusters
    } else {
        0
    };

    // Step 5: Build comprehensive single-shot documentation prompt
    println!("\n📝 Generating documentation (single batch)...");

    let mut batch_prompt = String::new();
    batch_prompt.push_str("You are an expert code documentation generator. Analyze this codebase and generate comprehensive documentation.\n\n");
    
    // Add file structure overview
    batch_prompt.push_str("## Project Structure\n\n");
    for (path, contents) in workspace.files.iter().take(100) {
        if let Ok(text) = String::from_utf8(contents.clone()) {
            let lines = text.lines().count();
            batch_prompt.push_str(&format!("- {} ({} lines)\n", path, lines));
        }
    }
    if workspace.files.len() > 100 {
        batch_prompt.push_str(&format!("\n... and {} more files\n", workspace.files.len() - 100));
    }
    
    // Add clustered code sections with CPU-extracted metadata
    if num_clusters > 0 {
        batch_prompt.push_str("\n## Code Clusters (Semantically Related)\n\n");
        
        for cluster_id in 1..=num_clusters as i64 {
            let cluster_chunks = docpack.get_cluster_chunks(cluster_id)?;
            if cluster_chunks.is_empty() {
                continue;
            }
            
            batch_prompt.push_str(&format!("### Cluster {} ({} chunks)\n\n", cluster_id, cluster_chunks.len()));
            
            // CPU-side deterministic extraction
            let mut symbols_found = std::collections::HashSet::new();
            let mut combined_text = String::new();
            
            for (_chunk_id, text) in &cluster_chunks {
                combined_text.push_str(text);
                combined_text.push('\n');
                
                // Extract symbols deterministically
                for line in text.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") {
                        if let Some(name) = trimmed.split_whitespace().nth(1) {
                            symbols_found.insert(name.trim_end_matches('(').to_string());
                        }
                    } else if trimmed.starts_with("struct ") || trimmed.starts_with("pub struct ") {
                        if let Some(name) = trimmed.split_whitespace().nth(1) {
                            symbols_found.insert(name.trim_end_matches('<').trim_end_matches('{').to_string());
                        }
                    } else if trimmed.starts_with("enum ") || trimmed.starts_with("pub enum ") {
                        if let Some(name) = trimmed.split_whitespace().nth(1) {
                            symbols_found.insert(name.trim_end_matches('{').to_string());
                        }
                    } else if trimmed.starts_with("trait ") || trimmed.starts_with("pub trait ") {
                        if let Some(name) = trimmed.split_whitespace().nth(1) {
                            symbols_found.insert(name.trim_end_matches('<').trim_end_matches('{').to_string());
                        }
                    }
                }
            }
            
            // Show extracted symbols
            if !symbols_found.is_empty() {
                batch_prompt.push_str("**Symbols detected**: ");
                batch_prompt.push_str(&symbols_found.into_iter().collect::<Vec<_>>().join(", "));
                batch_prompt.push_str("\n\n");
            }
            
            // Include condensed code (limit to prevent overflow)
            let char_limit = 2000;
            if combined_text.len() > char_limit {
                batch_prompt.push_str("```\n");
                batch_prompt.push_str(&combined_text[..char_limit]);
                batch_prompt.push_str("\n... (truncated)\n```\n\n");
            } else {
                batch_prompt.push_str("```\n");
                batch_prompt.push_str(&combined_text);
                batch_prompt.push_str("```\n\n");
            }
        }
    }
    
    // Single instruction for all documentation
    batch_prompt.push_str(
        r#"
## Task

Generate comprehensive documentation in this exact structure:

### Project Overview
Brief summary of what this project does, its purpose, and main technologies.

### Architecture
High-level organization: main modules, their responsibilities, and interactions.

"#);

    if num_clusters > 0 {
        batch_prompt.push_str("### Cluster Analysis\n");
        batch_prompt.push_str("For each cluster above, describe:\n");
        batch_prompt.push_str("- **Theme**: Common purpose/pattern\n");
        batch_prompt.push_str("- **Key Components**: Main symbols and their roles\n");
        batch_prompt.push_str("- **Relationships**: How components interact\n\n");
    }
    
    batch_prompt.push_str("Keep each section concise. Focus on clarity and usefulness.");

    // Single LLM call with extended context
    println!("  Sending batch prompt (~{} chars)...", batch_prompt.len());
    let req = GenerationRequest::new(config.model.clone(), batch_prompt);
    // Note: Context size is configured in the Ollama Modelfile, not per-request
    
    let mut stream = ollama.generate_stream(req).await?;
    
    let mut full_response = String::new();
    while let Some(chunk_result) = stream.next().await {
        let responses = chunk_result?;
        for resp in responses {
            if !resp.response.is_empty() {
                full_response.push_str(&resp.response);
            }
        }
    }
    
    println!("✓ Generated {} chars of documentation", full_response.len());

    // Store in database - architecture overview
    let node_id = docpack.insert_node(
        "architecture",
        Some("project_documentation"),
        None,
        None,
        None,
        None,
        None,
        None,
    )?;

    docpack.insert_doc(
        node_id,
        Some("Project Documentation"),
        Some(&full_response),
        None,
        None,
        None,
    )?;
    
    // Also create per-cluster nodes by parsing response sections
    if num_clusters > 0 {
        // Simple heuristic: split by cluster headers
        for cluster_id in 1..=num_clusters as i64 {
            let cluster_node = docpack.insert_node(
                "cluster",
                Some(&format!("cluster_{}", cluster_id)),
                None,
                None,
                None,
                None,
                None,
                None,
            )?;
            
            // Store a reference to the full doc
            docpack.insert_doc(
                cluster_node,
                Some(&format!("Cluster {} Documentation", cluster_id)),
                Some("See main project documentation for cluster analysis."),
                None,
                None,
                None,
            )?;
        }
    }

    // Log completion event
    docpack.add_build_event(
        &chrono::Utc::now().to_rfc3339(),
        "docgen_complete",
        Some(&serde_json::json!({
            "num_clusters": num_clusters,
            "doc_length": full_response.len()
        })),
    )?;

    Ok(DocGenResult {
        docpack,
        raw_llm_output: full_response,
        num_chunks: chunks.len(),
        num_embeddings,
    })
}
