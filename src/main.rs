mod chunking;
mod docgen;
mod docpack;
mod embedding;
mod ingest;

use docgen::{DocGenConfig, run_docgen};
use ingest::ingest_github_repo;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = "https://github.com/xandwr/localdoc";
    let project_name = "localdoc";

    println!("📦 Ingesting repository: {}", url);
    let workspace = ingest_github_repo(url).await?;
    println!("✓ Loaded {} files into workspace", workspace.files.len());

    println!("\n🤖 Generating documentation with LLM...");
    let cfg = DocGenConfig::default();

    let result = run_docgen(cfg, &workspace, url, project_name).await?;

    // Get metadata to display summary
    let metadata = result.docpack.get_metadata()?;

    println!("\n✅ Docpack created successfully!");
    println!("\n📊 Summary:");
    println!("  Project: {}", metadata.project_name.to_string());
    println!("  Version: {}", metadata.docpack_version);
    println!("  Created: {}", metadata.created_at);
    println!("  Source: {}", metadata.repo_url.unwrap_or_default());
    println!("  Files: {}", workspace.files.len());
    println!("  Chunks: {}", result.num_chunks);
    println!("  Embeddings: {}", result.num_embeddings);

    // Get node counts
    let node_counts = result.docpack.get_node_counts()?;
    if !node_counts.is_empty() {
        println!("\n📈 Nodes:");
        for (kind, count) in node_counts {
            println!("  {}: {}", kind, count);
        }
    }

    println!(
        "\n💾 Location: ~/.localdoc/docpacks/{}.docpack",
        metadata.project_name.to_string()
    );
    println!("\n🔍 You can now query this docpack using SQLite or a custom reader!");

    Ok(())
}
