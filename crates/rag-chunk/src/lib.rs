//! rag-chunk - Chunking strategies
//!
//! This crate provides various text chunking strategies for splitting
//! documents into appropriately-sized pieces for embedding.
//!
//! # Chunkers
//!
//! - [`RecursiveChunker`]: Recursively splits text using progressively smaller
//!   separators (paragraphs, lines, sentences, words).
//!
//! - [`AdaptiveChunker`]: Automatically selects the best chunking strategy
//!   based on content type.
//!
//! # Example
//!
//! ```rust
//! use rag_chunk::{AdaptiveChunker, Chunker};
//! use rag_core::{ChunkConfig, ContentType};
//!
//! let chunker = AdaptiveChunker::new();
//! let config = ChunkConfig::default();
//! let chunks = chunker.chunk("Hello world", ContentType::PlainText, &config).unwrap();
//! ```

mod adaptive;
mod recursive;

pub use adaptive::AdaptiveChunker;
pub use recursive::RecursiveChunker;

// Re-export types for convenience
pub use rag_core::{ChunkConfig, ChunkData, Chunker, ContentType};
