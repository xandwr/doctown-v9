use std::collections::HashMap;

/// Represents a semantic chunk of text from a file
#[derive(Debug, Clone)]
pub struct Chunk {
    pub file_id: i64,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub symbol_hint: Option<String>,
}

/// Configuration for chunking strategy
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum number of lines per chunk
    pub max_lines: usize,
    /// Minimum number of lines per chunk
    pub min_lines: usize,
    /// Overlap between chunks (in lines)
    pub overlap_lines: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_lines: 50,
            min_lines: 5,
            overlap_lines: 5,
        }
    }
}

/// Chunk files into semantic units
pub fn chunk_files(
    files: &HashMap<String, (i64, Vec<u8>)>,
    config: &ChunkConfig,
) -> anyhow::Result<Vec<Chunk>> {
    let mut chunks = Vec::new();

    for (path, (file_id, contents)) in files {
        // Try to parse as text
        let text = match String::from_utf8(contents.clone()) {
            Ok(t) => t,
            Err(_) => {
                // Skip binary files
                continue;
            }
        };

        // Detect file type from extension
        let extension = path.split('.').last().unwrap_or("");
        let file_chunks = match extension {
            "rs" | "py" | "js" | "ts" | "go" | "java" | "cpp" | "c" | "h" => {
                chunk_code_file(*file_id, &text, config)
            }
            "md" | "txt" | "rst" => chunk_markdown_file(*file_id, &text, config),
            "json" | "yaml" | "yml" | "toml" | "xml" => chunk_config_file(*file_id, &text, config),
            _ => chunk_generic_file(*file_id, &text, config),
        };

        chunks.extend(file_chunks);
    }

    Ok(chunks)
}

/// Chunk a code file by detecting logical blocks
fn chunk_code_file(file_id: i64, text: &str, config: &ChunkConfig) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    let mut chunks = Vec::new();

    if lines.is_empty() {
        return chunks;
    }

    let mut current_start = 0;
    let mut brace_depth = 0;
    let mut in_block = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Track brace depth to detect function/struct boundaries
        for ch in trimmed.chars() {
            match ch {
                '{' => {
                    brace_depth += 1;
                    in_block = true;
                }
                '}' => {
                    brace_depth -= 1;
                }
                _ => {}
            }
        }

        // At the end of a top-level block or max lines reached
        let chunk_size = i - current_start + 1;
        if (brace_depth == 0 && in_block && chunk_size >= config.min_lines)
            || chunk_size >= config.max_lines
        {
            let chunk_text = lines[current_start..=i].join("\n");
            let symbol_hint = detect_symbol_hint(&chunk_text);

            chunks.push(Chunk {
                file_id,
                start_line: current_start + 1,
                end_line: i + 1,
                text: chunk_text,
                symbol_hint,
            });

            // Start next chunk with overlap
            current_start = i.saturating_sub(config.overlap_lines) + 1;
            in_block = false;
        }
    }

    // Handle remaining lines
    if current_start < lines.len() {
        let chunk_text = lines[current_start..].join("\n");
        if chunk_text.trim().len() >= 10 {
            // Avoid tiny trailing chunks
            let symbol_hint = detect_symbol_hint(&chunk_text);
            chunks.push(Chunk {
                file_id,
                start_line: current_start + 1,
                end_line: lines.len(),
                text: chunk_text,
                symbol_hint,
            });
        }
    }

    chunks
}

/// Chunk markdown by headers
fn chunk_markdown_file(file_id: i64, text: &str, config: &ChunkConfig) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    let mut chunks = Vec::new();

    if lines.is_empty() {
        return chunks;
    }

    let mut current_start = 0;
    let mut current_header = None;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Detect markdown headers
        if trimmed.starts_with('#') {
            // Save previous section
            if i > current_start {
                let chunk_text = lines[current_start..i].join("\n");
                if chunk_text.trim().len() >= 10 {
                    chunks.push(Chunk {
                        file_id,
                        start_line: current_start + 1,
                        end_line: i,
                        text: chunk_text,
                        symbol_hint: current_header.clone(),
                    });
                }
            }

            current_start = i;
            current_header = Some(trimmed.trim_start_matches('#').trim().to_string());
        }

        // Also chunk if too large
        if i - current_start >= config.max_lines {
            let chunk_text = lines[current_start..=i].join("\n");
            chunks.push(Chunk {
                file_id,
                start_line: current_start + 1,
                end_line: i + 1,
                text: chunk_text,
                symbol_hint: current_header.clone(),
            });

            current_start = i + 1;
            current_header = None;
        }
    }

    // Handle remaining lines
    if current_start < lines.len() {
        let chunk_text = lines[current_start..].join("\n");
        if chunk_text.trim().len() >= 10 {
            chunks.push(Chunk {
                file_id,
                start_line: current_start + 1,
                end_line: lines.len(),
                text: chunk_text,
                symbol_hint: current_header,
            });
        }
    }

    chunks
}

/// Chunk config files (JSON, YAML, TOML) by top-level keys
fn chunk_config_file(file_id: i64, text: &str, config: &ChunkConfig) -> Vec<Chunk> {
    // For now, use generic chunking
    // Could be enhanced to parse structure
    chunk_generic_file(file_id, text, config)
}

/// Generic sliding window chunking
fn chunk_generic_file(file_id: i64, text: &str, config: &ChunkConfig) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    let mut chunks = Vec::new();

    if lines.is_empty() {
        return chunks;
    }

    let mut start = 0;

    while start < lines.len() {
        let end = (start + config.max_lines).min(lines.len());
        let chunk_text = lines[start..end].join("\n");

        if chunk_text.trim().len() >= 10 {
            chunks.push(Chunk {
                file_id,
                start_line: start + 1,
                end_line: end,
                text: chunk_text,
                symbol_hint: None,
            });
        }

        start = end.saturating_sub(config.overlap_lines);
        if start >= lines.len().saturating_sub(config.min_lines) {
            break; // Avoid tiny trailing chunks
        }
    }

    chunks
}

/// Attempt to detect what kind of symbol this chunk represents
fn detect_symbol_hint(text: &str) -> Option<String> {
    let trimmed = text.trim();

    // Rust patterns
    if let Some(captures) = trimmed.lines().next() {
        if captures.contains("fn ") {
            return extract_name_after("fn ", captures);
        }
        if captures.contains("struct ") {
            return extract_name_after("struct ", captures);
        }
        if captures.contains("enum ") {
            return extract_name_after("enum ", captures);
        }
        if captures.contains("trait ") {
            return extract_name_after("trait ", captures);
        }
        if captures.contains("impl ") {
            return Some("impl".to_string());
        }
        if captures.contains("mod ") {
            return extract_name_after("mod ", captures);
        }
    }

    None
}

fn extract_name_after(keyword: &str, line: &str) -> Option<String> {
    line.find(keyword).map(|pos| {
        let after = &line[pos + keyword.len()..];
        after
            .split(|c: char| c.is_whitespace() || c == '(' || c == '<' || c == '{')
            .next()
            .unwrap_or("")
            .to_string()
    })
}
