//! rag-mcp - MCP server implementation
//!
//! This crate provides an MCP (Model Context Protocol) server that exposes
//! RAG functionality as tools for AI assistants.
//!
//! # Tools
//!
//! - `rag_search` - Search the knowledge base
//! - `rag_ingest` - Ingest a document into the knowledge base
//! - `rag_list_collections` - List all collections
//! - `rag_create_collection` - Create a new collection
//! - `rag_delete_collection` - Delete a collection
//! - `rag_stats` - Get statistics about the knowledge base

mod server;

pub use server::{
    CollectionParams, IngestParams, RagMcpServer, SearchParams, ServerInfo, ToolInfo, ToolResult,
};
