use reqwest::Client;
use serde::Deserialize;
use std::{
    collections::HashMap,
    io::{Cursor, Read},
};
use thiserror::Error;
use url::Url;
use zip::ZipArchive;

#[derive(Error, Debug)]
pub enum IngestError {
    #[error("Invalid GitHub URL")]
    InvalidUrl,

    #[error("HTTP: {0}")]
    Http(#[from] reqwest::Error),

    #[error("ZIP: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("IO: {0}")]
    Io(#[from] std::io::Error),

    #[error("Downloaded file is not a valid ZIP archive")]
    NotAZip,
}

#[derive(Debug)]
pub struct IngestWorkspace {
    pub files: HashMap<String, Vec<u8>>,
}

#[derive(Deserialize)]
struct RepoMetadata {
    default_branch: String,
}

pub async fn ingest_github_repo(url: &str) -> Result<IngestWorkspace, IngestError> {
    let parsed = Url::parse(url).map_err(|_| IngestError::InvalidUrl)?;

    if parsed.host_str() != Some("github.com") {
        return Err(IngestError::InvalidUrl);
    }

    let mut segments = parsed.path_segments().ok_or(IngestError::InvalidUrl)?;
    let owner = segments.next().ok_or(IngestError::InvalidUrl)?;
    let repo = segments.next().ok_or(IngestError::InvalidUrl)?;

    let client = Client::new();

    // 1 — Ask GitHub for the real default branch
    let api = format!("https://api.github.com/repos/{}/{}", owner, repo);
    let meta: RepoMetadata = client
        .get(api)
        .header("User-Agent", "doctown") // GitHub requires this
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    // 2 — Download the correct branch ZIP
    let zip_url = format!(
        "https://github.com/{}/{}/archive/refs/heads/{}.zip",
        owner, repo, meta.default_branch
    );

    let bytes = client.get(&zip_url).send().await?.bytes().await?;
    let raw = bytes.to_vec();

    // 3 — Guard against HTML pretending to be a zip
    if raw.len() < 4 || &raw[..4] != b"PK\x03\x04" {
        return Err(IngestError::NotAZip);
    }

    // 4 — Extract into memory
    let cursor = Cursor::new(raw);
    let mut archive = ZipArchive::new(cursor)?;
    let mut workspace = IngestWorkspace {
        files: HashMap::new(),
    };

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if !file.is_file() {
            continue;
        }

        let path = strip_root(file.name());
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;

        workspace.files.insert(path, contents);
    }

    Ok(workspace)
}

fn strip_root(path: &str) -> String {
    path.splitn(2, '/').nth(1).unwrap_or("").to_string()
}
