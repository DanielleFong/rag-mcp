//! rag-query - Search and ranking engine
//!
//! This crate provides hybrid search capabilities combining vector similarity
//! and keyword search using Reciprocal Rank Fusion (RRF).
//!
//! # Features
//!
//! - Hybrid search (vector + keyword)
//! - Reciprocal Rank Fusion for combining results
//! - Context expansion with adjacent chunks
//! - Configurable weights and parameters
//!
//! # Example
//!
//! ```rust,ignore
//! use rag_query::{QueryEngine, QueryConfig};
//! use std::sync::Arc;
//!
//! let engine = QueryEngine::new(Arc::new(store), Arc::new(embedder));
//! let config = QueryConfig::default();
//! let results = engine.search("error handling", config).await?;
//! ```

mod engine;
mod fusion;

pub use engine::{QueryConfig, QueryEngine};
pub use fusion::{reciprocal_rank_fusion, weighted_fusion};

// Re-export for convenience
pub use rag_core::{SearchResult, SearchResults};
