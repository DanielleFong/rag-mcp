//! rag-embed - ONNX embedding model wrapper
//!
//! This crate provides ONNX-based text embeddings using models like
//! nomic-embed-text-v1.5 for semantic similarity search.
//!
//! # Features
//!
//! - ONNX Runtime integration for fast inference
//! - Support for asymmetric retrieval with document/query prefixes
//! - Mean pooling with attention mask
//! - L2 normalization
//! - Batch embedding support

mod onnx;

pub use onnx::{MockEmbedder, OnnxEmbedder};

// Re-export the Embedder trait for convenience
pub use rag_core::Embedder;
