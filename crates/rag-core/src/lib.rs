//! rag-core - Core types and traits for the RAG system
//!
//! This crate provides the foundational types, traits, and error handling
//! used throughout the rag-mcp system.

pub mod config;
pub mod error;
pub mod hlc;
pub mod traits;
pub mod types;

pub use config::*;
pub use error::{RagError, Result};
pub use hlc::HybridLogicalClock;
pub use traits::*;
pub use types::*;
