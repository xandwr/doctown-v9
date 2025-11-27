mod chunking;
mod clustering;
mod docgen;
mod docpack;
mod embedding;
mod ingest;
mod llm;

use docgen::{DocGenConfig, run_docgen};
use ingest::ingest_github_repo;
use llm::LLMBackend;
use std::env;
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = "https://github.com/xandwr/localdoc";
    let project_name = "localdoc";

    let total_start = Instant::now();

    println!("📦 Ingesting repository: {}", url);
    let ingest_start = Instant::now();
    let workspace = ingest_github_repo(url).await?;
    let ingest_duration = ingest_start.elapsed();
    println!("✓ Loaded {} files into workspace", workspace.files.len());

    println!("\n🤖 Generating documentation with LLM...");

    // Check for vLLM configuration via environment variables
    let cfg = if let Ok(vllm_url) = env::var("VLLM_URL") {
        let model = env::var("VLLM_MODEL")
            .unwrap_or_else(|_| "meta-llama/Llama-3.1-8B-Instruct".to_string());
        println!("🚀 Using vLLM backend at {} with model {}", vllm_url, model);
        DocGenConfig {
            llm_backend: LLMBackend::VLlm {
                base_url: vllm_url,
                model,
                api_key: env::var("VLLM_API_KEY").ok(),
            },
            embedding_model_path: Some("embedding/minilm-l6/model.onnx".into()),
        }
    } else {
        println!("🦙 Using Ollama backend (set VLLM_URL to use vLLM)");
        DocGenConfig::default()
    };

    let result = run_docgen(cfg, &workspace, url, project_name).await?;

    // Get metadata to display summary
    let metadata = result.docpack.get_metadata()?;

    println!("\n✅ Docpack created successfully!");
    println!("\n📊 Summary:");
    println!("  Project: {}", metadata.project_name);
    println!("  Version: {}", metadata.docpack_version);
    println!("  Created: {}", metadata.created_at);
    println!("  Source: {}", metadata.repo_url.unwrap_or_default());
    println!("  Files: {}", workspace.files.len());
    println!("  Chunks: {}", result.num_chunks);
    println!("  Embeddings: {}", result.num_embeddings);

    // Get cluster count
    let cluster_count = result.docpack.get_cluster_count().unwrap_or(0);
    if cluster_count > 0 {
        println!("  Clusters: {}", cluster_count);
    }

    // Get node counts
    let node_counts = result.docpack.get_node_counts()?;
    if !node_counts.is_empty() {
        println!("\n📈 Documentation Nodes:");
        for (kind, count) in node_counts {
            println!("  {}: {}", kind, count);
        }
    }

    println!(
        "\n💾 Location: ~/.localdoc/docpacks/{}.docpack",
        metadata.project_name.to_string()
    );
    println!("\n🔍 You can now query this docpack using SQLite or a custom reader!");

    let total_duration = total_start.elapsed();

    // Print timing summary
    println!("\n⏱️  Pipeline Timing Summary:");
    println!("  Ingestion:      {:.2?}", ingest_duration);
    if let Some(chunk_time) = result.chunk_duration {
        println!("  Chunking:       {:.2?}", chunk_time);
    }
    if let Some(embed_time) = result.embedding_duration {
        println!("  Embedding:      {:.2?}", embed_time);
    }
    if let Some(cluster_time) = result.clustering_duration {
        println!("  Clustering:     {:.2?}", cluster_time);
    }
    if let Some(llm_time) = result.llm_duration {
        println!("  LLM Generation: {:.2?}", llm_time);
    }
    println!("  {}", "-".repeat(40));
    println!("  Total:          {:.2?}", total_duration);

    Ok(())
}
