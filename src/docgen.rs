use crate::chunking::{ChunkConfig, chunk_files};
use crate::clustering::{cluster_embeddings, get_cluster_sizes};
use crate::docpack::{DocPack, Metadata, create_docpack};
use crate::embedding::EmbeddingModel;
use crate::ingest::IngestWorkspace;
use crate::llm::{LLMBackend, create_provider};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tokio_stream::StreamExt;

#[derive(Clone, Debug)]
pub struct DocGenConfig {
    pub llm_backend: LLMBackend,
    pub embedding_model_path: Option<String>,
}

impl Default for DocGenConfig {
    fn default() -> Self {
        Self {
            llm_backend: LLMBackend::Ollama {
                host: "http://localhost".into(),
                port: 11434,
                model: "qwen3:0.6b".into(),
            },
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
    pub chunk_duration: Option<Duration>,
    pub embedding_duration: Option<Duration>,
    pub clustering_duration: Option<Duration>,
    pub llm_duration: Option<Duration>,
}

/// Extract symbols from code text
fn extract_symbols(text: &str) -> HashSet<String> {
    let mut symbols = HashSet::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") {
            if let Some(name) = trimmed.split_whitespace().nth(1) {
                symbols.insert(name.trim_end_matches('(').to_string());
            }
        } else if trimmed.starts_with("struct ") || trimmed.starts_with("pub struct ") {
            if let Some(name) = trimmed.split_whitespace().nth(1) {
                symbols.insert(name.trim_end_matches('<').trim_end_matches('{').to_string());
            }
        } else if trimmed.starts_with("enum ") || trimmed.starts_with("pub enum ") {
            if let Some(name) = trimmed.split_whitespace().nth(1) {
                symbols.insert(name.trim_end_matches('{').to_string());
            }
        } else if trimmed.starts_with("trait ") || trimmed.starts_with("pub trait ") {
            if let Some(name) = trimmed.split_whitespace().nth(1) {
                symbols.insert(name.trim_end_matches('<').trim_end_matches('{').to_string());
            }
        } else if trimmed.starts_with("impl ") {
            if let Some(parts) = trimmed.split_whitespace().nth(1) {
                symbols.insert(
                    parts
                        .trim_end_matches('<')
                        .trim_end_matches('{')
                        .to_string(),
                );
            }
        }
    }

    symbols
}

/// Build a focused prompt for a batch of clusters (3-5 clusters per batch)
fn build_batch_prompt(batch_clusters: &[(i64, Vec<(i64, String)>)]) -> String {
    let mut prompt = String::new();

    prompt.push_str("You are a code documentation expert. Analyze these semantically-related code clusters and provide concise documentation for each.\n\n");

    for (cluster_id, cluster_chunks) in batch_clusters {
        // CPU-side deterministic symbol extraction
        let mut symbols_found = HashSet::new();
        let mut combined_text = String::new();

        for (_chunk_id, text) in cluster_chunks {
            combined_text.push_str(text);
            combined_text.push('\n');
            symbols_found.extend(extract_symbols(text));
        }

        prompt.push_str(&format!(
            "## Cluster {} ({} chunks)\n\n",
            cluster_id,
            cluster_chunks.len()
        ));

        if !symbols_found.is_empty() {
            prompt.push_str("**Symbols**: ");
            prompt.push_str(&symbols_found.into_iter().collect::<Vec<_>>().join(", "));
            prompt.push_str("\n\n");
        }

        prompt.push_str("**Code**:\n```\n");
        // Limit per cluster to keep batch manageable
        let char_limit = 2500;
        if combined_text.len() > char_limit {
            prompt.push_str(&combined_text[..char_limit]);
            prompt.push_str("\n... (truncated)\n");
        } else {
            prompt.push_str(&combined_text);
        }
        prompt.push_str("```\n\n");
    }

    prompt.push_str("**Task**: For EACH cluster above, provide brief documentation covering:\n");
    prompt.push_str("1. **Purpose**: What this code does (1-2 sentences)\n");
    prompt.push_str("2. **Key Components**: Main functions/types and their roles\n");
    prompt.push_str("3. **Usage**: How components interact or are used\n\n");
    prompt.push_str(
        "Format your response with clear cluster headers like '## Cluster N' for each one.\n",
    );
    prompt.push_str("Be concise and focus on clarity.");

    prompt
}

/// Parse batch LLM response and split into per-cluster documentation
fn parse_batch_response(
    response: &str,
    batch_clusters: &[(i64, Vec<(i64, String)>)],
) -> Vec<(i64, String)> {
    let mut cluster_docs = Vec::new();

    // Try to split by "## Cluster N" headers
    let lines: Vec<&str> = response.lines().collect();
    let mut current_cluster_id: Option<i64> = None;
    let mut current_content = String::new();

    for line in lines {
        if line.starts_with("## Cluster ") {
            // Save previous cluster if any
            if let Some(cluster_id) = current_cluster_id {
                if !current_content.trim().is_empty() {
                    cluster_docs.push((cluster_id, current_content.trim().to_string()));
                }
            }

            // Extract cluster number
            if let Some(num_str) = line.strip_prefix("## Cluster ") {
                if let Ok(cluster_id) = num_str
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .parse::<i64>()
                {
                    current_cluster_id = Some(cluster_id);
                    current_content.clear();
                }
            }
        } else if current_cluster_id.is_some() {
            current_content.push_str(line);
            current_content.push('\n');
        }
    }

    // Save last cluster
    if let Some(cluster_id) = current_cluster_id {
        if !current_content.trim().is_empty() {
            cluster_docs.push((cluster_id, current_content.trim().to_string()));
        }
    }

    // If parsing failed, distribute response evenly across clusters
    if cluster_docs.is_empty() && !batch_clusters.is_empty() {
        let doc_per_cluster = response.to_string();
        for (cluster_id, _) in batch_clusters {
            cluster_docs.push((*cluster_id, doc_per_cluster.clone()));
        }
    }

    cluster_docs
}

/// Generate a docgen output and populate a SQLite docpack
pub async fn run_docgen(
    config: DocGenConfig,
    workspace: &IngestWorkspace,
    repo_url: &str,
    project_name: &str,
) -> anyhow::Result<DocGenResult> {
    let llm_provider = create_provider(config.llm_backend.clone());

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
    let chunk_start = Instant::now();
    let chunk_config = ChunkConfig::default();
    let chunks = chunk_files(&file_map, &chunk_config)?;
    let chunk_duration = chunk_start.elapsed();
    println!(
        "✓ Created {} chunks in {:.2?}",
        chunks.len(),
        chunk_duration
    );

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
    let embed_start = Instant::now();
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

                let embed_duration = embed_start.elapsed();
                println!(
                    "✓ Embedded {} chunks in {:.2?}",
                    embedded_count, embed_duration
                );

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
    let embedding_duration = if num_embeddings > 0 {
        Some(embed_start.elapsed())
    } else {
        None
    };

    // Step 4: Cluster the embeddings
    let cluster_start = Instant::now();
    let num_clusters = if num_embeddings > 0 {
        println!("\n🔍 Clustering embeddings...");

        let embeddings = docpack.get_all_embeddings()?;
        let num_clusters = (embeddings.len() as f32 / 10.0).ceil() as usize;
        let num_clusters = num_clusters.clamp(3, 20); // Between 3-20 clusters

        let cluster_result = cluster_embeddings(&embeddings, num_clusters)?;
        let cluster_duration = cluster_start.elapsed();
        println!(
            "✓ Found {} clusters in {:.2?}",
            cluster_result.num_clusters, cluster_duration
        );

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
    let clustering_duration = if num_clusters > 0 {
        Some(cluster_start.elapsed())
    } else {
        None
    };

    // Step 5: Generate documentation in batches (3-5 clusters per batch)
    const BATCH_SIZE: usize = 4; // Sweet spot for balancing KV priming vs locality
    let num_batches = (num_clusters + BATCH_SIZE - 1) / BATCH_SIZE;

    println!(
        "\n📝 Generating documentation ({} batches of ~{} clusters)...",
        num_batches, BATCH_SIZE
    );
    let llm_start = Instant::now();

    let mut total_doc_chars = 0;
    let mut cluster_docs = Vec::new();

    if num_clusters > 0 {
        // Process clusters in batches
        for batch_idx in 0..num_batches {
            let batch_start_cluster = ((batch_idx * BATCH_SIZE) + 1) as i64;
            let batch_end_cluster = (((batch_idx + 1) * BATCH_SIZE).min(num_clusters)) as i64;

            // Collect cluster data for this batch
            let mut batch_clusters: Vec<(i64, Vec<(i64, String)>)> = Vec::new();
            for cluster_id in batch_start_cluster..=batch_end_cluster {
                let cluster_chunks = docpack.get_cluster_chunks(cluster_id)?;
                if !cluster_chunks.is_empty() {
                    batch_clusters.push((cluster_id, cluster_chunks));
                }
            }

            if batch_clusters.is_empty() {
                continue;
            }

            // Build batch prompt covering 3-5 clusters
            let batch_prompt = build_batch_prompt(&batch_clusters);
            let total_chunks: usize = batch_clusters.iter().map(|(_, c)| c.len()).sum();

            print!(
                "  Batch {} (clusters {}-{}, {} chunks, {} chars)...",
                batch_idx + 1,
                batch_start_cluster,
                batch_end_cluster,
                total_chunks,
                batch_prompt.len()
            );

            let batch_time_start = Instant::now();
            let mut stream = llm_provider.generate_stream(batch_prompt).await?;

            let mut batch_response = String::new();
            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;
                if !chunk.is_empty() {
                    batch_response.push_str(&chunk);
                }
            }

            let batch_duration = batch_time_start.elapsed();
            println!(" {} chars in {:.2?}", batch_response.len(), batch_duration);

            total_doc_chars += batch_response.len();

            // Parse batch response and split by cluster
            // Simple heuristic: split by "## Cluster N" headers
            let cluster_sections = parse_batch_response(&batch_response, &batch_clusters);

            for (cluster_id, cluster_doc) in cluster_sections {
                cluster_docs.push((cluster_id, cluster_doc));
            }
        }
    }

    let llm_duration = llm_start.elapsed();
    println!(
        "✓ Generated {} chars of documentation across {} clusters in {} batches in {:.2?}",
        total_doc_chars, num_clusters, num_batches, llm_duration
    );

    // Store per-cluster documentation in database
    for (cluster_id, cluster_doc) in &cluster_docs {
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

        docpack.insert_doc(
            cluster_node,
            Some(&format!("Cluster {} Documentation", cluster_id)),
            Some(cluster_doc),
            None,
            None,
            None,
        )?;
    }

    // Create aggregated architecture overview
    let mut full_response = String::new();
    full_response.push_str("# Project Documentation\n\n");
    full_response.push_str("## Architecture Overview\n\n");
    full_response.push_str(&format!(
        "This project is organized into {} semantic clusters:\n\n",
        num_clusters
    ));

    for (cluster_id, cluster_doc) in &cluster_docs {
        full_response.push_str(&format!("### Cluster {}\n\n", cluster_id));
        full_response.push_str(cluster_doc);
        full_response.push_str("\n\n");
    }

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
        chunk_duration: Some(chunk_duration),
        embedding_duration,
        clustering_duration,
        llm_duration: Some(llm_duration),
    })
}
