# rag-core

Core types, traits, and error definitions shared across all crates.

## Overview

| Attribute | Value |
|-----------|-------|
| Purpose | Foundation types and abstractions |
| Dependencies | serde, thiserror, ulid, blake3 |
| Est. Lines | ~800 |
| Confidence | HIGH |

This crate has no runtime dependencies on other rag-* crates. It defines the shared vocabulary for the entire system.

---

## Module Structure

```
rag-core/
├── src/
│   ├── lib.rs           # Public exports
│   ├── types/
│   │   ├── mod.rs
│   │   ├── document.rs  # Document, ContentType
│   │   ├── chunk.rs     # Chunk, ChunkMetadata
│   │   ├── collection.rs # Collection, CollectionSettings
│   │   ├── embedding.rs  # Embedding type aliases
│   │   └── ids.rs       # ID types (DocId, ChunkId, etc.)
│   ├── traits/
│   │   ├── mod.rs
│   │   ├── store.rs     # Store trait
│   │   ├── embedder.rs  # Embedder trait
│   │   ├── chunker.rs   # Chunker trait
│   │   └── sync.rs      # SyncPeer trait
│   ├── error.rs         # RagError enum
│   ├── hlc.rs           # Hybrid Logical Clock
│   └── config.rs        # Configuration types
└── Cargo.toml
```

---

## Types

### Document

```rust
use serde::{Deserialize, Serialize};
use ulid::Ulid;

/// A document in the RAG system.
///
/// Documents are the top-level unit of content. Each document belongs to
/// exactly one collection and may be split into multiple chunks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Globally unique identifier (time-ordered).
    pub id: Ulid,

    /// Collection this document belongs to.
    pub collection: String,

    /// Source URI (file://, https://, data:).
    pub source_uri: String,

    /// Blake3 hash of raw content for deduplication.
    pub content_hash: [u8; 32],

    /// Detected or specified content type.
    pub content_type: ContentType,

    /// Original content (optional, for reconstruction).
    pub raw_content: Option<String>,

    /// User-provided metadata.
    pub metadata: serde_json::Value,

    /// Creation timestamp (Unix millis).
    pub created_at: u64,

    /// Last update timestamp (Unix millis).
    pub updated_at: u64,

    /// Hybrid logical clock for sync.
    pub hlc: HybridLogicalClock,
}

impl Document {
    /// Create a new document with generated ID and timestamps.
    pub fn new(
        collection: impl Into<String>,
        source_uri: impl Into<String>,
        content: &str,
        content_type: ContentType,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            id: Ulid::new(),
            collection: collection.into(),
            source_uri: source_uri.into(),
            content_hash: blake3::hash(content.as_bytes()).into(),
            content_type,
            raw_content: Some(content.to_string()),
            metadata: serde_json::json!({}),
            created_at: now,
            updated_at: now,
            hlc: HybridLogicalClock::new(0), // Node ID set by caller
        }
    }

    /// Check if content has changed compared to another document.
    pub fn content_changed(&self, other: &Document) -> bool {
        self.content_hash != other.content_hash
    }
}
```

### ContentType

```rust
/// Detected or specified content type for chunking strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    // Programming languages (AST-aware chunking)
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Java,
    Cpp,
    C,
    Ruby,
    Php,
    Swift,
    Kotlin,
    Scala,
    Haskell,
    Elixir,
    Zig,

    // Documentation (semantic chunking)
    Markdown,
    RestructuredText,
    AsciiDoc,
    Html,
    Latex,
    PlainText,

    // Configuration (record-based chunking)
    Json,
    Yaml,
    Toml,
    Xml,
    Ini,

    // Data formats
    Csv,
    Sql,

    // Special
    ChatLog,
    GitDiff,
    JupyterNotebook,

    // Binary with text extraction
    Pdf,

    // Fallback
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
            "cpp" | "cc" | "cxx" | "hpp" | "h" => Self::Cpp,
            "c" => Self::C,
            "rb" => Self::Ruby,
            "php" => Self::Php,
            "swift" => Self::Swift,
            "kt" | "kts" => Self::Kotlin,
            "scala" | "sc" => Self::Scala,
            "hs" => Self::Haskell,
            "ex" | "exs" => Self::Elixir,
            "zig" => Self::Zig,
            "md" | "markdown" => Self::Markdown,
            "rst" => Self::RestructuredText,
            "adoc" | "asciidoc" => Self::AsciiDoc,
            "html" | "htm" => Self::Html,
            "tex" => Self::Latex,
            "txt" => Self::PlainText,
            "json" => Self::Json,
            "yaml" | "yml" => Self::Yaml,
            "toml" => Self::Toml,
            "xml" => Self::Xml,
            "ini" | "cfg" => Self::Ini,
            "csv" => Self::Csv,
            "sql" => Self::Sql,
            "ipynb" => Self::JupyterNotebook,
            "pdf" => Self::Pdf,
            "diff" | "patch" => Self::GitDiff,
            _ => Self::Unknown,
        }
    }

    /// Detect content type from file path.
    pub fn from_path(path: &std::path::Path) -> Self {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }

    /// Get the tree-sitter language for this content type, if applicable.
    pub fn tree_sitter_language(&self) -> Option<&'static str> {
        match self {
            Self::Rust => Some("rust"),
            Self::Python => Some("python"),
            Self::TypeScript | Self::JavaScript => Some("typescript"),
            Self::Go => Some("go"),
            Self::Java => Some("java"),
            Self::Cpp | Self::C => Some("cpp"),
            Self::Ruby => Some("ruby"),
            Self::Php => Some("php"),
            _ => None,
        }
    }

    /// Check if this content type supports AST-aware chunking.
    pub fn supports_ast_chunking(&self) -> bool {
        self.tree_sitter_language().is_some()
    }
}
```

### Chunk

```rust
/// A chunk of content extracted from a document.
///
/// Chunks are the atomic unit for embedding and retrieval. Each chunk
/// has exactly one embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Globally unique identifier.
    pub id: Ulid,

    /// Parent document ID.
    pub doc_id: Ulid,

    /// Position in document (0-indexed).
    pub chunk_index: u32,

    /// Text content of this chunk.
    pub content: String,

    /// Blake3 hash for incremental update detection.
    pub content_hash: [u8; 32],

    /// Token count (model-specific).
    pub token_count: u32,

    /// Character offset in original document.
    pub char_offset_start: u64,

    /// Character offset end in original document.
    pub char_offset_end: u64,

    /// Chunk-specific metadata.
    pub metadata: ChunkMetadata,

    /// Hybrid logical clock for sync.
    pub hlc: HybridLogicalClock,
}

/// Metadata about how a chunk was created.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Line number range in source document.
    pub line_start: u32,
    pub line_end: u32,

    /// For code: AST node type (e.g., "function_item", "class_definition").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub syntax_node_type: Option<String>,

    /// For code: Name of the function/class/etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub syntax_node_name: Option<String>,

    /// For code: Programming language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// For markdown: Heading hierarchy (e.g., ["Chapter 1", "Section 1.2"]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heading_hierarchy: Option<Vec<String>>,

    /// Chunking strategy used.
    pub chunking_strategy: String,

    /// Whether this chunk overlaps with the previous chunk.
    #[serde(default)]
    pub overlaps_previous: bool,

    /// Whether this chunk overlaps with the next chunk.
    #[serde(default)]
    pub overlaps_next: bool,
}

impl Chunk {
    /// Create a chunk from raw output of a chunker.
    pub fn from_output(doc_id: Ulid, index: u32, output: ChunkOutput) -> Self {
        Self {
            id: Ulid::new(),
            doc_id,
            chunk_index: index,
            content_hash: blake3::hash(output.content.as_bytes()).into(),
            content: output.content,
            token_count: output.token_count,
            char_offset_start: output.char_offset_start,
            char_offset_end: output.char_offset_end,
            metadata: output.metadata,
            hlc: HybridLogicalClock::new(0),
        }
    }
}

/// Output from a chunker, before assignment of IDs.
#[derive(Debug, Clone)]
pub struct ChunkOutput {
    pub content: String,
    pub token_count: u32,
    pub char_offset_start: u64,
    pub char_offset_end: u64,
    pub metadata: ChunkMetadata,
}
```

### Collection

```rust
/// A collection groups related documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    /// Unique name (identifier).
    pub name: String,

    /// Human-readable description.
    pub description: Option<String>,

    /// Collection-specific settings.
    pub settings: CollectionSettings,

    /// Creation timestamp.
    pub created_at: u64,

    /// Hybrid logical clock for sync.
    pub hlc: HybridLogicalClock,
}

/// Per-collection configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CollectionSettings {
    /// Default chunking parameters.
    #[serde(default)]
    pub chunking: ChunkingSettings,

    /// Default search parameters.
    #[serde(default)]
    pub search: SearchSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingSettings {
    /// Maximum tokens per chunk.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Minimum tokens per chunk.
    #[serde(default = "default_min_tokens")]
    pub min_tokens: usize,

    /// Token overlap between chunks.
    #[serde(default)]
    pub overlap_tokens: usize,
}

impl Default for ChunkingSettings {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            min_tokens: 50,
            overlap_tokens: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSettings {
    /// Default number of results.
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Use hybrid search by default.
    #[serde(default = "default_hybrid")]
    pub hybrid: bool,

    /// Vector weight in hybrid search (0.0 = keyword only, 1.0 = vector only).
    #[serde(default = "default_hybrid_alpha")]
    pub hybrid_alpha: f32,
}

impl Default for SearchSettings {
    fn default() -> Self {
        Self {
            top_k: 10,
            hybrid: true,
            hybrid_alpha: 0.5,
        }
    }
}

fn default_max_tokens() -> usize { 512 }
fn default_min_tokens() -> usize { 50 }
fn default_top_k() -> usize { 10 }
fn default_hybrid() -> bool { true }
fn default_hybrid_alpha() -> f32 { 0.5 }
```

### Hybrid Logical Clock

```rust
/// Hybrid Logical Clock for causality tracking in sync.
///
/// HLC combines physical time with a logical counter to provide:
/// - Causality tracking: if A happens-before B, then HLC(A) < HLC(B)
/// - Bounded drift: HLC is always within epsilon of physical time
/// - Total ordering: no two events have the same HLC
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HybridLogicalClock {
    /// Physical timestamp (Unix milliseconds).
    pub wall_time: u64,

    /// Logical counter for events at the same wall_time.
    pub logical: u32,

    /// Node identifier for uniqueness.
    pub node_id: u16,
}

impl HybridLogicalClock {
    /// Create a new HLC with the current time.
    pub fn new(node_id: u16) -> Self {
        Self {
            wall_time: Self::now(),
            logical: 0,
            node_id,
        }
    }

    /// Get current time in milliseconds.
    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Generate the next HLC for a local event.
    pub fn tick(&mut self) -> Self {
        let now = Self::now();

        if now > self.wall_time {
            self.wall_time = now;
            self.logical = 0;
        } else {
            self.logical = self.logical.saturating_add(1);
        }

        *self
    }

    /// Update HLC after receiving a remote event.
    pub fn update(&mut self, remote: &HybridLogicalClock) -> Self {
        let now = Self::now();

        if now > self.wall_time && now > remote.wall_time {
            self.wall_time = now;
            self.logical = 0;
        } else if self.wall_time == remote.wall_time {
            self.logical = self.logical.max(remote.logical).saturating_add(1);
        } else if self.wall_time > remote.wall_time {
            self.logical = self.logical.saturating_add(1);
        } else {
            self.wall_time = remote.wall_time;
            self.logical = remote.logical.saturating_add(1);
        }

        *self
    }

    /// Serialize to bytes for storage.
    pub fn to_bytes(&self) -> [u8; 14] {
        let mut buf = [0u8; 14];
        buf[0..8].copy_from_slice(&self.wall_time.to_be_bytes());
        buf[8..12].copy_from_slice(&self.logical.to_be_bytes());
        buf[12..14].copy_from_slice(&self.node_id.to_be_bytes());
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(buf: &[u8; 14]) -> Self {
        Self {
            wall_time: u64::from_be_bytes(buf[0..8].try_into().unwrap()),
            logical: u32::from_be_bytes(buf[8..12].try_into().unwrap()),
            node_id: u16::from_be_bytes(buf[12..14].try_into().unwrap()),
        }
    }

    /// Serialize to hex string.
    pub fn to_hex(&self) -> String {
        hex::encode(self.to_bytes())
    }

    /// Deserialize from hex string.
    pub fn from_hex(s: &str) -> Result<Self, hex::FromHexError> {
        let bytes = hex::decode(s)?;
        if bytes.len() != 14 {
            return Err(hex::FromHexError::InvalidStringLength);
        }
        Ok(Self::from_bytes(bytes.as_slice().try_into().unwrap()))
    }
}
```

---

## Traits

### Store Trait

```rust
use async_trait::async_trait;

/// Abstraction over the storage layer.
///
/// Implementations: SqliteStore (rag-store)
#[async_trait]
pub trait Store: Send + Sync {
    // === Collections ===

    /// Create a new collection.
    async fn create_collection(&self, collection: Collection) -> Result<()>;

    /// Get collection by name.
    async fn get_collection(&self, name: &str) -> Result<Option<Collection>>;

    /// List all collections.
    async fn list_collections(&self) -> Result<Vec<Collection>>;

    /// Delete a collection and all its documents.
    async fn delete_collection(&self, name: &str) -> Result<()>;

    // === Documents ===

    /// Insert a new document.
    async fn insert_document(&self, document: Document) -> Result<Ulid>;

    /// Get document by ID.
    async fn get_document(&self, id: Ulid) -> Result<Option<Document>>;

    /// Get document by source URI within a collection.
    async fn get_document_by_uri(&self, collection: &str, uri: &str) -> Result<Option<Document>>;

    /// List documents in a collection.
    async fn list_documents(
        &self,
        collection: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<Document>>;

    /// Update a document.
    async fn update_document(&self, document: Document) -> Result<()>;

    /// Delete a document and its chunks.
    async fn delete_document(&self, id: Ulid) -> Result<()>;

    /// Count documents in a collection.
    async fn count_documents(&self, collection: Option<&str>) -> Result<u64>;

    // === Chunks ===

    /// Insert chunks (with FTS5 indexing).
    async fn insert_chunks(&self, chunks: Vec<Chunk>) -> Result<()>;

    /// Get chunks for a document.
    async fn get_chunks(&self, doc_id: Ulid) -> Result<Vec<Chunk>>;

    /// Get a single chunk by ID.
    async fn get_chunk(&self, id: Ulid) -> Result<Option<Chunk>>;

    /// Get chunk by document ID and index.
    async fn get_chunk_by_index(&self, doc_id: Ulid, index: u32) -> Result<Option<Chunk>>;

    /// Delete chunks for a document.
    async fn delete_chunks(&self, doc_id: Ulid) -> Result<()>;

    /// Delete specific chunks by ID.
    async fn delete_chunks_by_ids(&self, ids: &[Ulid]) -> Result<()>;

    /// Count chunks.
    async fn count_chunks(&self, collection: Option<&str>) -> Result<u64>;

    // === Embeddings ===

    /// Insert embeddings (chunk_id, vector pairs).
    async fn insert_embeddings(&self, embeddings: Vec<(Ulid, Vec<f32>)>) -> Result<()>;

    /// Delete embeddings by chunk IDs.
    async fn delete_embeddings(&self, chunk_ids: &[Ulid]) -> Result<()>;

    // === Search ===

    /// Vector similarity search.
    async fn vector_search(
        &self,
        query: &[f32],
        k: usize,
        collection: Option<&str>,
    ) -> Result<Vec<(Ulid, f32)>>; // (chunk_id, distance)

    /// Keyword search via FTS5.
    async fn keyword_search(
        &self,
        query: &str,
        k: usize,
        collection: Option<&str>,
    ) -> Result<Vec<(Ulid, f32)>>; // (chunk_id, BM25 score)

    // === Sync ===

    /// Get changes since a given HLC.
    async fn get_changes_since(&self, hlc: HybridLogicalClock) -> Result<Vec<Change>>;

    /// Apply changes from a peer.
    async fn apply_changes(&self, changes: Vec<Change>) -> Result<Vec<ConflictReport>>;

    /// Get current watermark (highest HLC).
    async fn get_watermark(&self) -> Result<HybridLogicalClock>;

    // === Maintenance ===

    /// Optimize indices.
    async fn optimize(&self) -> Result<()>;

    /// Get storage statistics.
    async fn stats(&self) -> Result<StoreStats>;
}

/// Storage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStats {
    pub collections: u64,
    pub documents: u64,
    pub chunks: u64,
    pub embeddings: u64,
    pub storage_bytes: u64,
    pub index_bytes: u64,
}
```

### Embedder Trait

```rust
/// Abstraction over embedding models.
///
/// Implementations: OnnxEmbedder (rag-embed)
pub trait Embedder: Send + Sync {
    /// Embed multiple texts (for documents/chunks).
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Embed a single query (may use different prefix).
    fn embed_query(&self, query: &str) -> Result<Vec<f32>>;

    /// Get embedding dimension.
    fn dimension(&self) -> usize;

    /// Get maximum input tokens.
    fn max_tokens(&self) -> usize;

    /// Get model identifier.
    fn model_id(&self) -> &str;
}
```

### Chunker Trait

```rust
/// Abstraction over chunking strategies.
///
/// Implementations: various chunkers in rag-chunk
pub trait Chunker: Send + Sync {
    /// Chunk content based on content type.
    fn chunk(
        &self,
        content: &str,
        content_type: ContentType,
        settings: &ChunkingSettings,
    ) -> Result<Vec<ChunkOutput>>;

    /// Get supported content types.
    fn supported_types(&self) -> &[ContentType];
}
```

### SyncPeer Trait

```rust
/// Abstraction over sync communication.
///
/// Implementations: HttpSyncPeer (rag-sync)
#[async_trait]
pub trait SyncPeer: Send + Sync {
    /// Get the peer's identifier.
    fn peer_id(&self) -> &str;

    /// Get the peer's endpoint.
    fn endpoint(&self) -> &str;

    /// Get the peer's current watermark.
    async fn get_watermark(&self) -> Result<HybridLogicalClock>;

    /// Get changes since a given HLC.
    async fn get_changes_since(
        &self,
        hlc: HybridLogicalClock,
        limit: usize,
    ) -> Result<ChangeSet>;

    /// Push changes to the peer.
    async fn push_changes(&self, changes: ChangeSet) -> Result<PushResult>;
}

/// A set of changes for sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeSet {
    pub changes: Vec<Change>,
    pub watermark: HybridLogicalClock,
    pub has_more: bool,
}

/// A single change in the sync log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Change {
    CollectionCreate { collection: Collection },
    CollectionDelete { name: String, hlc: HybridLogicalClock },
    DocumentInsert {
        document: Document,
        chunks: Vec<Chunk>,
        embeddings: Vec<(Ulid, Vec<f32>)>,
    },
    DocumentUpdate {
        document: Document,
        chunks_to_delete: Vec<Ulid>,
        chunks_to_insert: Vec<Chunk>,
        embeddings_to_delete: Vec<Ulid>,
        embeddings_to_insert: Vec<(Ulid, Vec<f32>)>,
    },
    DocumentDelete { doc_id: Ulid, hlc: HybridLogicalClock },
}

/// Result of pushing changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushResult {
    pub accepted: bool,
    pub conflicts: Vec<ConflictReport>,
    pub new_watermark: HybridLogicalClock,
}

/// Report of a conflict during sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictReport {
    pub entity_type: String,  // "document", "collection"
    pub entity_id: String,
    pub local_hlc: HybridLogicalClock,
    pub remote_hlc: HybridLogicalClock,
    pub resolution: String,   // "keep_local", "take_remote"
}
```

---

## Error Types

```rust
use thiserror::Error;

/// Unified error type for the RAG system.
#[derive(Debug, Error)]
pub enum RagError {
    // === Storage Errors ===
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Document not found: {0}")]
    DocumentNotFound(Ulid),

    #[error("Chunk not found: {0}")]
    ChunkNotFound(Ulid),

    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Collection already exists: {0}")]
    CollectionExists(String),

    #[error("Duplicate document: {uri} already exists in {collection}")]
    DuplicateDocument { uri: String, collection: String },

    // === Embedding Errors ===
    #[error("Embedding model error: {0}")]
    EmbeddingModel(String),

    #[error("Text too long: {tokens} tokens exceeds maximum {max}")]
    TextTooLong { tokens: usize, max: usize },

    #[error("Empty text cannot be embedded")]
    EmptyText,

    // === Chunking Errors ===
    #[error("Failed to parse {content_type}: {reason}")]
    ParseError { content_type: String, reason: String },

    #[error("Unsupported content type: {0:?}")]
    UnsupportedContentType(ContentType),

    #[error("Chunking produced no output")]
    EmptyChunks,

    // === Sync Errors ===
    #[error("Sync failed with peer {peer}: {reason}")]
    SyncFailed { peer: String, reason: String },

    #[error("Peer unreachable: {0}")]
    PeerUnreachable(String),

    #[error("Conflict resolution failed: {0}")]
    ConflictResolution(String),

    #[error("Invalid HLC format: {0}")]
    InvalidHlc(String),

    // === IO Errors ===
    #[error("Failed to load URI {uri}: {reason}")]
    LoadFailed { uri: String, reason: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(String),

    // === Validation Errors ===
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Invalid URI: {0}")]
    InvalidUri(String),

    #[error("Invalid collection name: {0}")]
    InvalidCollectionName(String),

    // === MCP Errors ===
    #[error("MCP protocol error: {0}")]
    McpProtocol(String),

    #[error("Unknown tool: {0}")]
    UnknownTool(String),

    // === Internal Errors ===
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

impl RagError {
    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Database(_)
                | Self::PeerUnreachable(_)
                | Self::SyncFailed { .. }
                | Self::Http(_)
                | Self::Io(_)
        )
    }

    /// Get error code for serialization.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Database(_) => "DATABASE_ERROR",
            Self::DocumentNotFound(_) => "DOCUMENT_NOT_FOUND",
            Self::ChunkNotFound(_) => "CHUNK_NOT_FOUND",
            Self::CollectionNotFound(_) => "COLLECTION_NOT_FOUND",
            Self::CollectionExists(_) => "COLLECTION_EXISTS",
            Self::DuplicateDocument { .. } => "DUPLICATE_DOCUMENT",
            Self::EmbeddingModel(_) => "EMBEDDING_ERROR",
            Self::TextTooLong { .. } => "TEXT_TOO_LONG",
            Self::EmptyText => "EMPTY_TEXT",
            Self::ParseError { .. } => "PARSE_ERROR",
            Self::UnsupportedContentType(_) => "UNSUPPORTED_CONTENT_TYPE",
            Self::EmptyChunks => "EMPTY_CHUNKS",
            Self::SyncFailed { .. } => "SYNC_FAILED",
            Self::PeerUnreachable(_) => "PEER_UNREACHABLE",
            Self::ConflictResolution(_) => "CONFLICT_RESOLUTION",
            Self::InvalidHlc(_) => "INVALID_HLC",
            Self::LoadFailed { .. } => "LOAD_FAILED",
            Self::Io(_) => "IO_ERROR",
            Self::Http(_) => "HTTP_ERROR",
            Self::InvalidArgument(_) => "INVALID_ARGUMENT",
            Self::InvalidUri(_) => "INVALID_URI",
            Self::InvalidCollectionName(_) => "INVALID_COLLECTION_NAME",
            Self::McpProtocol(_) => "MCP_PROTOCOL_ERROR",
            Self::UnknownTool(_) => "UNKNOWN_TOOL",
            Self::Internal(_) => "INTERNAL_ERROR",
            Self::NotImplemented(_) => "NOT_IMPLEMENTED",
        }
    }
}

/// Convenience Result type alias.
pub type Result<T> = std::result::Result<T, RagError>;
```

---

## Configuration Types

```rust
/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Path to SQLite database.
    pub database_path: PathBuf,

    /// Path to embedding model.
    pub model_path: PathBuf,

    /// Node ID for this instance (for sync).
    pub node_id: u16,

    /// Server configuration (for MCP/REST).
    #[serde(default)]
    pub server: ServerConfig,

    /// Sync configuration.
    #[serde(default)]
    pub sync: SyncConfig,

    /// Logging configuration.
    #[serde(default)]
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Bind address for HTTP server.
    #[serde(default = "default_bind_address")]
    pub bind_address: String,

    /// Port for HTTP server.
    #[serde(default = "default_port")]
    pub port: u16,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Enable sync server.
    #[serde(default)]
    pub enabled: bool,

    /// Sync peers.
    #[serde(default)]
    pub peers: Vec<PeerConfig>,

    /// Sync interval in seconds.
    #[serde(default = "default_sync_interval")]
    pub interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerConfig {
    pub id: String,
    pub endpoint: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error).
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Log format (json, pretty).
    #[serde(default = "default_log_format")]
    pub format: String,
}

fn default_bind_address() -> String { "127.0.0.1".to_string() }
fn default_port() -> u16 { 8765 }
fn default_sync_interval() -> u64 { 60 }
fn default_log_level() -> String { "info".to_string() }
fn default_log_format() -> String { "pretty".to_string() }

impl Config {
    /// Load configuration from file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| RagError::InvalidArgument(e.to_string()))?;
        Ok(config)
    }

    /// Load from default locations.
    pub fn load_default() -> Result<Self> {
        let paths = [
            dirs::config_dir().map(|p| p.join("rag-mcp/config.toml")),
            Some(PathBuf::from("./rag-mcp.toml")),
        ];

        for path in paths.into_iter().flatten() {
            if path.exists() {
                return Self::load(&path);
            }
        }

        Ok(Self::default())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database_path: dirs::data_dir()
                .map(|p| p.join("rag-mcp/rag.db"))
                .unwrap_or_else(|| PathBuf::from("./rag.db")),
            model_path: dirs::data_dir()
                .map(|p| p.join("rag-mcp/models/nomic-embed-text-v1.5"))
                .unwrap_or_else(|| PathBuf::from("./models/nomic-embed-text-v1.5")),
            node_id: rand::random(),
            server: ServerConfig::default(),
            sync: SyncConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}
```

---

## Confidence Assessment

| Component | Confidence | Notes |
|-----------|------------|-------|
| Core types | HIGH | Standard domain modeling |
| HLC implementation | HIGH | Well-documented algorithm |
| Store trait | HIGH | Standard async trait pattern |
| Embedder trait | HIGH | Simple interface |
| Chunker trait | HIGH | Simple interface |
| SyncPeer trait | HIGH | Standard patterns |
| Error types | HIGH | Comprehensive coverage |
| Configuration | HIGH | Standard TOML config |

---

## Cargo.toml

```toml
[package]
name = "rag-core"
version = "0.1.0"
edition = "2021"
description = "Core types and traits for the RAG system"
license = "MIT"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
ulid = { version = "1.1", features = ["serde"] }
blake3 = "1.5"
hex = "0.4"
async-trait = "0.1"
bitflags = "2.4"
toml = "0.8"
dirs = "5.0"
rand = "0.8"

[dev-dependencies]
tokio = { version = "1.0", features = ["macros", "rt-multi-thread"] }
```
