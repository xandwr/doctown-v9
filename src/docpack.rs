use rusqlite::{Connection, Result as SqlResult, params};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// A DocPack is a SQLite database containing code intelligence data
#[derive(Debug)]
pub struct DocPack {
    conn: Connection,
}

/// Metadata about the docpack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub docpack_version: String,
    pub project_name: String,
    pub project_description: Option<String>,
    pub repo_url: Option<String>,
    pub commit_sha: Option<String>,
    pub created_at: String,
    pub generator_version: String,
    pub language: Option<String>,
    pub extra: Option<serde_json::Value>,
}

#[allow(dead_code)]
impl DocPack {
    /// Create a new docpack at the given path
    pub fn create<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let conn = Connection::open(path)?;
        let mut pack = DocPack { conn };
        pack.init_schema()?;
        Ok(pack)
    }

    /// Open an existing docpack
    pub fn open<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let conn = Connection::open(path)?;
        Ok(DocPack { conn })
    }

    /// Initialize the database schema
    fn init_schema(&mut self) -> anyhow::Result<()> {
        self.conn.execute_batch(
            r#"
            -- Metadata: single row, global info
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                docpack_version TEXT NOT NULL,
                project_name TEXT NOT NULL,
                project_description TEXT,
                repo_url TEXT,
                commit_sha TEXT,
                created_at TEXT NOT NULL,
                generator_version TEXT NOT NULL,
                language TEXT,
                extra JSON
            );

            -- Files: each source file stored once
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                language TEXT,
                contents BLOB NOT NULL,
                line_count INTEGER,
                hash TEXT
            );

            -- Chunks: smallest semantic units
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id),
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                symbol_hint TEXT,
                hash TEXT
            );

            -- Embeddings: one row per chunk
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id),
                vector BLOB NOT NULL,
                dims INTEGER NOT NULL,
                norm REAL
            );

            -- Clusters: output of HDBSCAN / k-means
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY,
                label TEXT,
                size INTEGER,
                score REAL
            );

            CREATE TABLE IF NOT EXISTS cluster_membership (
                chunk_id INTEGER NOT NULL REFERENCES chunks(id),
                cluster_id INTEGER NOT NULL REFERENCES clusters(id),
                PRIMARY KEY (chunk_id, cluster_id)
            );

            -- Nodes: the semantic graph layer
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT,
                file_id INTEGER REFERENCES files(id),
                chunk_id INTEGER REFERENCES chunks(id),
                signature TEXT,
                docs TEXT,
                span_start INTEGER,
                span_end INTEGER
            );

            -- Edges: relationships between nodes
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY,
                src INTEGER NOT NULL REFERENCES nodes(id),
                dst INTEGER NOT NULL REFERENCES nodes(id),
                kind TEXT NOT NULL,
                weight REAL
            );

            -- Docs: LLM-generated documentation
            CREATE TABLE IF NOT EXISTS docs (
                node_id INTEGER PRIMARY KEY REFERENCES nodes(id),
                summary TEXT,
                details TEXT,
                examples TEXT,
                reasoning JSON,
                raw_json JSON
            );

            -- Full-text search (FTS5)
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                node_id UNINDEXED,
                summary,
                details,
                examples,
                content=''
            );

            -- Build events: optional provenance
            CREATE TABLE IF NOT EXISTS build_events (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                stage TEXT,
                payload JSON
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
            CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
            CREATE INDEX IF NOT EXISTS idx_cluster_membership_cluster ON cluster_membership(cluster_id);
            "#
        )?;

        Ok(())
    }

    /// Clear all data from the docpack (but keep the schema)
    pub fn clear_data(&self) -> anyhow::Result<()> {
        self.conn.execute_batch(
            r#"
            DELETE FROM docs;
            DELETE FROM edges;
            DELETE FROM nodes;
            DELETE FROM cluster_membership;
            DELETE FROM clusters;
            DELETE FROM embeddings;
            DELETE FROM chunks;
            DELETE FROM files;
            DELETE FROM build_events;
            DELETE FROM metadata;
            "#,
        )?;

        // For contentless FTS tables, we need to drop and recreate
        self.conn.execute_batch(
            r#"
            DROP TABLE IF EXISTS docs_fts;
            CREATE VIRTUAL TABLE docs_fts USING fts5(
                node_id UNINDEXED,
                summary,
                details,
                examples,
                content=''
            );
            "#,
        )?;

        Ok(())
    }

    /// Insert or replace metadata (overwrites existing metadata if present)
    pub fn insert_metadata(&self, meta: &Metadata) -> anyhow::Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO metadata (
                id, docpack_version, project_name, project_description,
                repo_url, commit_sha, created_at, generator_version,
                language, extra
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            params![
                1,
                &meta.docpack_version,
                &meta.project_name,
                &meta.project_description,
                &meta.repo_url,
                &meta.commit_sha,
                &meta.created_at,
                &meta.generator_version,
                &meta.language,
                meta.extra.as_ref().map(|v| v.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Get metadata
    pub fn get_metadata(&self) -> anyhow::Result<Metadata> {
        let mut stmt = self.conn.prepare(
            "SELECT docpack_version, project_name, project_description, repo_url,
                    commit_sha, created_at, generator_version, language, extra
             FROM metadata WHERE id = 1",
        )?;

        let meta = stmt.query_row([], |row| {
            Ok(Metadata {
                docpack_version: row.get(0)?,
                project_name: row.get(1)?,
                project_description: row.get(2)?,
                repo_url: row.get(3)?,
                commit_sha: row.get(4)?,
                created_at: row.get(5)?,
                generator_version: row.get(6)?,
                language: row.get(7)?,
                extra: row
                    .get::<_, Option<String>>(8)?
                    .and_then(|s| serde_json::from_str(&s).ok()),
            })
        })?;

        Ok(meta)
    }

    /// Insert a file
    pub fn insert_file(
        &self,
        path: &str,
        language: Option<&str>,
        contents: &[u8],
        line_count: Option<usize>,
        hash: Option<&str>,
    ) -> anyhow::Result<i64> {
        self.conn.execute(
            "INSERT INTO files (path, language, contents, line_count, hash)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![path, language, contents, line_count, hash],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Insert a chunk
    pub fn insert_chunk(
        &self,
        file_id: i64,
        start_line: usize,
        end_line: usize,
        text: &str,
        symbol_hint: Option<&str>,
        hash: Option<&str>,
    ) -> anyhow::Result<i64> {
        self.conn.execute(
            "INSERT INTO chunks (file_id, start_line, end_line, text, symbol_hint, hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![file_id, start_line, end_line, text, symbol_hint, hash],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Insert an embedding
    pub fn insert_embedding(
        &self,
        chunk_id: i64,
        vector: &[f32],
        norm: Option<f32>,
    ) -> anyhow::Result<()> {
        let vector_bytes = vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>();

        self.conn.execute(
            "INSERT INTO embeddings (chunk_id, vector, dims, norm)
             VALUES (?1, ?2, ?3, ?4)",
            params![chunk_id, vector_bytes, vector.len(), norm],
        )?;
        Ok(())
    }

    /// Get all embeddings from the database
    pub fn get_all_embeddings(&self) -> anyhow::Result<Vec<(i64, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT chunk_id, vector, dims FROM embeddings")?;

        let results = stmt.query_map([], |row| {
            let chunk_id: i64 = row.get(0)?;
            let vector_bytes: Vec<u8> = row.get(1)?;
            let dims: usize = row.get(2)?;

            Ok((chunk_id, vector_bytes, dims))
        })?;

        let mut embeddings = Vec::new();
        for result in results {
            let (chunk_id, vector_bytes, dims) = result?;

            // Decode f32 vector from bytes
            let mut vector = Vec::with_capacity(dims);
            for i in 0..dims {
                let offset = i * 4;
                if offset + 4 <= vector_bytes.len() {
                    let bytes = [
                        vector_bytes[offset],
                        vector_bytes[offset + 1],
                        vector_bytes[offset + 2],
                        vector_bytes[offset + 3],
                    ];
                    vector.push(f32::from_le_bytes(bytes));
                }
            }

            embeddings.push((chunk_id, vector));
        }

        Ok(embeddings)
    }

    /// Insert a cluster
    pub fn insert_cluster(
        &self,
        label: Option<&str>,
        size: usize,
        score: Option<f32>,
    ) -> anyhow::Result<i64> {
        self.conn.execute(
            "INSERT INTO clusters (label, size, score) VALUES (?1, ?2, ?3)",
            params![label, size, score],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Insert cluster membership
    pub fn insert_cluster_membership(&self, chunk_id: i64, cluster_id: i64) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO cluster_membership (chunk_id, cluster_id) VALUES (?1, ?2)",
            params![chunk_id, cluster_id],
        )?;
        Ok(())
    }

    /// Get chunks in a specific cluster
    pub fn get_cluster_chunks(&self, cluster_id: i64) -> anyhow::Result<Vec<(i64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.text
             FROM chunks c
             JOIN cluster_membership cm ON c.id = cm.chunk_id
             WHERE cm.cluster_id = ?1",
        )?;

        let results = stmt.query_map([cluster_id], |row| Ok((row.get(0)?, row.get(1)?)))?;

        Ok(results.collect::<SqlResult<Vec<_>>>()?)
    }

    /// Insert a node
    pub fn insert_node(
        &self,
        kind: &str,
        name: Option<&str>,
        file_id: Option<i64>,
        chunk_id: Option<i64>,
        signature: Option<&str>,
        docs: Option<&str>,
        span_start: Option<usize>,
        span_end: Option<usize>,
    ) -> anyhow::Result<i64> {
        self.conn.execute(
            "INSERT INTO nodes (kind, name, file_id, chunk_id, signature, docs, span_start, span_end)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![kind, name, file_id, chunk_id, signature, docs, span_start, span_end],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Insert an edge
    pub fn insert_edge(
        &self,
        src: i64,
        dst: i64,
        kind: &str,
        weight: Option<f32>,
    ) -> anyhow::Result<i64> {
        self.conn.execute(
            "INSERT INTO edges (src, dst, kind, weight) VALUES (?1, ?2, ?3, ?4)",
            params![src, dst, kind, weight],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Insert LLM-generated documentation
    pub fn insert_doc(
        &self,
        node_id: i64,
        summary: Option<&str>,
        details: Option<&str>,
        examples: Option<&str>,
        reasoning: Option<&serde_json::Value>,
        raw_json: Option<&serde_json::Value>,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO docs (node_id, summary, details, examples, reasoning, raw_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                node_id,
                summary,
                details,
                examples,
                reasoning.map(|v| v.to_string()),
                raw_json.map(|v| v.to_string()),
            ],
        )?;

        // Also insert into FTS index
        if summary.is_some() || details.is_some() || examples.is_some() {
            self.conn.execute(
                "INSERT INTO docs_fts (rowid, node_id, summary, details, examples)
                 VALUES (?1, ?1, ?2, ?3, ?4)",
                params![node_id, summary, details, examples],
            )?;
        }

        Ok(())
    }

    /// Add a build event
    pub fn add_build_event(
        &self,
        timestamp: &str,
        stage: &str,
        payload: Option<&serde_json::Value>,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO build_events (timestamp, stage, payload) VALUES (?1, ?2, ?3)",
            params![timestamp, stage, payload.map(|v| v.to_string())],
        )?;
        Ok(())
    }

    /// Full-text search across docs
    pub fn search_docs(&self, query: &str) -> anyhow::Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT node_id, summary, details
            FROM docs_fts
            WHERE docs_fts MATCH ?1
            ORDER BY rank
            LIMIT 50
            "#,
        )?;

        let results = stmt
            .query_map([query], |row| {
                Ok((
                    row.get(0)?,
                    row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                ))
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(results)
    }

    /// Get node count by kind
    pub fn get_node_counts(&self) -> anyhow::Result<Vec<(String, i64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY COUNT(*) DESC")?;

        let results = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(results)
    }

    /// Get cluster count
    pub fn get_cluster_count(&self) -> anyhow::Result<i64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM clusters", [], |row| row.get(0))?;
        Ok(count)
    }
}

/// Create a new docpack in ~/.localdoc/docpacks/
/// If a docpack with the same name already exists, it will be cleared and reused
pub fn create_docpack(project_name: &str) -> anyhow::Result<(DocPack, PathBuf)> {
    // Get home directory
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| anyhow::anyhow!("Could not determine home directory"))?;

    let docpacks_dir = Path::new(&home).join(".localdoc").join("docpacks");
    fs::create_dir_all(&docpacks_dir)?;

    let filename = format!("{}.docpack", project_name);
    let filepath = docpacks_dir.join(filename);

    // Check if docpack already exists
    let pack = if filepath.exists() {
        // Open existing docpack and clear all data
        let pack = DocPack::open(&filepath)?;
        pack.clear_data()?;
        pack
    } else {
        // Create new docpack
        DocPack::create(&filepath)?
    };

    Ok((pack, filepath))
}
