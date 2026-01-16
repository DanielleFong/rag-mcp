//! Core domain types for the RAG system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ulid::Ulid;

use crate::hlc::HybridLogicalClock;

/// Content type for documents, determines chunking strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContentType {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Java,
    Cpp,
    C,
    Ruby,
    Markdown,
    Html,
    Json,
    Yaml,
    Toml,
    PlainText,
    Unknown,
}

impl ContentType {
    /// Detect content type from file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Self::Rust,
            "py" | "pyi" => Self::Python,
            "ts" | "tsx" => Self::TypeScript,
            "js" | "jsx" | "mjs" | "cjs" => Self::JavaScript,
            "go" => Self::Go,
            "java" => Self::Java,
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Self::Cpp,
            "c" | "h" => Self::C,
            "rb" => Self::Ruby,
            "md" | "markdown" => Self::Markdown,
            "html" | "htm" => Self::Html,
            "json" => Self::Json,
            "yaml" | "yml" => Self::Yaml,
            "toml" => Self::Toml,
            "txt" => Self::PlainText,
            _ => Self::Unknown,
        }
    }

    /// Detect content type from file path.
    pub fn from_path(path: &str) -> Self {
        path.rsplit('.')
            .next()
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }

    /// Check if this content type supports AST-aware chunking.
    pub fn supports_ast_chunking(&self) -> bool {
        matches!(
            self,
            Self::Rust
                | Self::Python
                | Self::TypeScript
                | Self::JavaScript
                | Self::Go
                | Self::Java
                | Self::Cpp
                | Self::C
                | Self::Ruby
        )
    }

    /// Check if this is a markup/documentation format.
    pub fn is_markup(&self) -> bool {
        matches!(self, Self::Markdown | Self::Html)
    }
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Rust => "Rust",
            Self::Python => "Python",
            Self::TypeScript => "TypeScript",
            Self::JavaScript => "JavaScript",
            Self::Go => "Go",
            Self::Java => "Java",
            Self::Cpp => "C++",
            Self::C => "C",
            Self::Ruby => "Ruby",
            Self::Markdown => "Markdown",
            Self::Html => "HTML",
            Self::Json => "JSON",
            Self::Yaml => "YAML",
            Self::Toml => "TOML",
            Self::PlainText => "Plain Text",
            Self::Unknown => "Unknown",
        };
        write!(f, "{}", s)
    }
}

/// A document in the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier (ULID).
    pub id: Ulid,

    /// Collection this document belongs to.
    pub collection: String,

    /// Source URI (file://, https://, data:).
    pub source_uri: String,

    /// Blake3 hash of raw content for deduplication.
    #[serde(with = "serde_bytes_opt")]
    pub content_hash: Option<[u8; 32]>,

    /// Original raw content (may be None after ingestion).
    pub raw_content: Option<String>,

    /// Content type (auto-detected or specified).
    pub content_type: ContentType,

    /// User-provided metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Creation timestamp (Unix millis).
    pub created_at: u64,

    /// Last update timestamp (Unix millis).
    pub updated_at: u64,

    /// Hybrid logical clock for sync.
    pub hlc: HybridLogicalClock,
}

impl Document {
    /// Create a new document.
    pub fn new(collection: &str, source_uri: &str, content: &str, content_type: ContentType) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let content_hash = blake3::hash(content.as_bytes());

        Self {
            id: Ulid::new(),
            collection: collection.to_string(),
            source_uri: source_uri.to_string(),
            content_hash: Some(*content_hash.as_bytes()),
            raw_content: Some(content.to_string()),
            content_type,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            hlc: HybridLogicalClock::new(0), // Node ID set by store
        }
    }

    /// Check if content has changed by comparing hashes.
    pub fn content_changed(&self, new_content: &str) -> bool {
        let new_hash = blake3::hash(new_content.as_bytes());
        self.content_hash
            .map(|h| h != *new_hash.as_bytes())
            .unwrap_or(true)
    }
}

/// A chunk of a document for embedding and search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier (ULID).
    pub id: Ulid,

    /// Parent document ID.
    pub doc_id: Ulid,

    /// Index within the document (0-based).
    pub chunk_index: u32,

    /// Chunk text content.
    pub content: String,

    /// Token count (from tokenizer).
    pub token_count: u32,

    /// Start line in source (1-based).
    pub start_line: u32,

    /// End line in source (1-based, inclusive).
    pub end_line: u32,

    /// Blake3 hash of chunk content.
    #[serde(with = "serde_bytes_opt")]
    pub content_hash: Option<[u8; 32]>,

    /// Hybrid logical clock for sync.
    pub hlc: HybridLogicalClock,
}

impl Chunk {
    /// Create a new chunk.
    pub fn new(
        doc_id: Ulid,
        chunk_index: u32,
        content: &str,
        token_count: u32,
        start_line: u32,
        end_line: u32,
    ) -> Self {
        let content_hash = blake3::hash(content.as_bytes());

        Self {
            id: Ulid::new(),
            doc_id,
            chunk_index,
            content: content.to_string(),
            token_count,
            start_line,
            end_line,
            content_hash: Some(*content_hash.as_bytes()),
            hlc: HybridLogicalClock::new(0),
        }
    }
}

/// A collection of documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    /// Collection name (unique identifier).
    pub name: String,

    /// Optional description.
    pub description: Option<String>,

    /// Creation timestamp (Unix millis).
    pub created_at: u64,

    /// Hybrid logical clock for sync.
    pub hlc: HybridLogicalClock,
}

impl Collection {
    /// Create a new collection.
    pub fn new(name: &str, description: Option<&str>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            name: name.to_string(),
            description: description.map(String::from),
            created_at: now,
            hlc: HybridLogicalClock::new(0),
        }
    }
}

/// A search result with score and chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result rank (1-indexed).
    pub rank: u32,

    /// Relevance score (higher is better).
    pub score: f32,

    /// The matched chunk.
    pub chunk: Chunk,

    /// Source document URI.
    pub source_uri: String,

    /// Collection name.
    pub collection: String,
}

/// Search results container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// The original query.
    pub query: String,

    /// Total results returned.
    pub total_results: usize,

    /// Search latency in milliseconds.
    pub latency_ms: u64,

    /// Individual results.
    pub results: Vec<SearchResult>,
}

/// Statistics about the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    /// Number of collections.
    pub collections: u64,

    /// Number of documents.
    pub documents: u64,

    /// Number of chunks.
    pub chunks: u64,

    /// Number of embeddings.
    pub embeddings: u64,

    /// Database size in bytes.
    pub storage_bytes: u64,

    /// Optional collection filter applied.
    pub filter: Option<String>,
}

/// Helper module for optional byte array serialization.
mod serde_bytes_opt {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &Option<[u8; 32]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(bytes) => {
                let hex = hex::encode(bytes);
                hex.serialize(serializer)
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<[u8; 32]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<String> = Option::deserialize(deserializer)?;
        match opt {
            Some(hex) => {
                let bytes = hex::decode(&hex).map_err(serde::de::Error::custom)?;
                let arr: [u8; 32] = bytes
                    .try_into()
                    .map_err(|_| serde::de::Error::custom("invalid hash length"))?;
                Ok(Some(arr))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_type_from_extension() {
        assert_eq!(ContentType::from_extension("rs"), ContentType::Rust);
        assert_eq!(ContentType::from_extension("py"), ContentType::Python);
        assert_eq!(ContentType::from_extension("ts"), ContentType::TypeScript);
        assert_eq!(ContentType::from_extension("md"), ContentType::Markdown);
        assert_eq!(ContentType::from_extension("xyz"), ContentType::Unknown);
    }

    #[test]
    fn test_content_type_from_path() {
        assert_eq!(ContentType::from_path("src/lib.rs"), ContentType::Rust);
        assert_eq!(ContentType::from_path("README.md"), ContentType::Markdown);
        assert_eq!(ContentType::from_path("no_extension"), ContentType::Unknown);
    }

    #[test]
    fn test_document_content_changed() {
        let doc = Document::new("test", "file://test.rs", "fn main() {}", ContentType::Rust);
        assert!(!doc.content_changed("fn main() {}"));
        assert!(doc.content_changed("fn main() { println!(); }"));
    }
}
