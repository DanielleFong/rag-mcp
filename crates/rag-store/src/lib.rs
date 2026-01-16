//! rag-store - SQLite storage layer with sqlite-vec
//!
//! This crate provides persistent storage for documents, chunks, and embeddings
//! using SQLite with the sqlite-vec extension for vector similarity search.

mod schema;
mod sqlite;

pub use sqlite::SqliteStore;

// Re-export schema for testing/migrations
pub use schema::SCHEMA;
