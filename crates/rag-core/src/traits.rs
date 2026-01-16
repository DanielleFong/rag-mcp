//! Core traits defining the interfaces between components.

use async_trait::async_trait;
use ulid::Ulid;

use crate::error::Result;
use crate::hlc::HybridLogicalClock;
use crate::types::{Chunk, Collection, ContentType, Document, Stats};

/// Storage layer trait.
#[async_trait]
pub trait Store: Send + Sync {
    // Collection operations
    async fn create_collection(&self, collection: Collection) -> Result<()>;
    async fn get_collection(&self, name: &str) -> Result<Option<Collection>>;
    async fn list_collections(&self) -> Result<Vec<Collection>>;
    async fn delete_collection(&self, name: &str) -> Result<()>;

    // Document operations
    async fn insert_document(&self, doc: Document) -> Result<()>;
    async fn get_document(&self, id: Ulid) -> Result<Option<Document>>;
    async fn get_document_by_uri(&self, uri: &str) -> Result<Option<Document>>;
    async fn list_documents(&self, collection: &str, limit: u32, offset: u32) -> Result<Vec<Document>>;
    async fn delete_document(&self, id: Ulid) -> Result<()>;

    // Chunk operations
    async fn insert_chunks(&self, chunks: &[Chunk]) -> Result<()>;
    async fn get_chunks_for_document(&self, doc_id: Ulid) -> Result<Vec<Chunk>>;
    async fn get_chunk(&self, id: Ulid) -> Result<Option<Chunk>>;
    async fn delete_chunks_for_document(&self, doc_id: Ulid) -> Result<()>;

    // Embedding operations
    async fn insert_embeddings(&self, chunk_ids: &[Ulid], embeddings: &[Vec<f32>]) -> Result<()>;

    // Search operations
    async fn vector_search(
        &self,
        embedding: &[f32],
        k: u32,
        collection: Option<&str>,
    ) -> Result<Vec<(Ulid, f32)>>;

    async fn keyword_search(
        &self,
        query: &str,
        k: u32,
        collection: Option<&str>,
    ) -> Result<Vec<(Ulid, f32)>>;

    // Stats
    async fn get_stats(&self, collection: Option<&str>) -> Result<Stats>;

    // Sync operations
    async fn get_watermark(&self) -> Result<HybridLogicalClock>;
    async fn get_changes_since(&self, hlc: &HybridLogicalClock) -> Result<Vec<SyncChange>>;
    async fn apply_changes(&self, changes: &[SyncChange]) -> Result<()>;
}

/// A change record for sync.
#[derive(Debug, Clone)]
pub enum SyncChange {
    UpsertCollection(Collection),
    DeleteCollection(String),
    UpsertDocument(Document),
    DeleteDocument(Ulid),
    UpsertChunk(Chunk, Vec<f32>), // Chunk with embedding
    DeleteChunk(Ulid),
}

/// Embedding model trait.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a batch of document texts.
    ///
    /// Prefixes each text with "search_document: " for asymmetric retrieval.
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Embed a single query text.
    ///
    /// Prefixes with "search_query: " for asymmetric retrieval.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;

    /// Count tokens in text.
    fn count_tokens(&self, text: &str) -> Result<usize>;

    /// Get the embedding dimension.
    fn dimension(&self) -> usize;

    /// Get the maximum context length in tokens.
    fn max_tokens(&self) -> usize;
}

/// Chunking configuration.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum tokens per chunk.
    pub max_tokens: usize,

    /// Minimum tokens per chunk (avoid tiny chunks).
    pub min_tokens: usize,

    /// Token overlap between chunks (for sliding window).
    pub overlap_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            min_tokens: 50,
            overlap_tokens: 0,
        }
    }
}

/// Chunking strategy trait.
pub trait Chunker: Send + Sync {
    /// Chunk text content into pieces.
    fn chunk(
        &self,
        content: &str,
        content_type: ContentType,
        config: &ChunkConfig,
    ) -> Result<Vec<ChunkData>>;

    /// Get supported content types.
    fn supported_types(&self) -> Vec<ContentType>;
}

/// Raw chunk data before ID assignment.
#[derive(Debug, Clone)]
pub struct ChunkData {
    /// Chunk text content.
    pub content: String,

    /// Token count.
    pub token_count: usize,

    /// Start line (1-based).
    pub start_line: u32,

    /// End line (1-based, inclusive).
    pub end_line: u32,
}

/// Sync peer trait for multi-node synchronization.
#[async_trait]
pub trait SyncPeer: Send + Sync {
    /// Get the peer's identifier.
    fn peer_id(&self) -> &str;

    /// Get the peer's endpoint URL.
    fn endpoint(&self) -> &str;

    /// Fetch the peer's current watermark (highest HLC).
    async fn get_watermark(&self) -> Result<HybridLogicalClock>;

    /// Pull changes from peer since the given HLC.
    async fn pull_changes(&self, since: &HybridLogicalClock) -> Result<Vec<SyncChange>>;

    /// Push changes to peer.
    async fn push_changes(&self, changes: &[SyncChange]) -> Result<()>;
}
