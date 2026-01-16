//! MCP server implementation.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::info;

use rag_chunk::{AdaptiveChunker, ChunkConfig, Chunker};
use rag_core::{Collection, ContentType, Document, Store};
use rag_embed::{Embedder, MockEmbedder};
use rag_query::{QueryConfig, QueryEngine};
use rag_store::SqliteStore;

/// RAG MCP Server state.
pub struct RagMcpServer {
    /// Database store.
    store: Arc<SqliteStore>,

    /// Embedder (mock for now).
    embedder: Arc<MockEmbedder>,

    /// Chunker.
    chunker: Arc<AdaptiveChunker>,

    /// Query engine.
    engine: Arc<QueryEngine<SqliteStore, MockEmbedder>>,
}

/// Search request parameters.
#[derive(Debug, Deserialize, Serialize)]
pub struct SearchParams {
    /// The search query.
    pub query: String,

    /// Maximum number of results (default: 10).
    #[serde(default = "default_top_k")]
    pub top_k: u32,

    /// Collection to search (optional).
    pub collection: Option<String>,
}

fn default_top_k() -> u32 {
    10
}

/// Ingest request parameters.
#[derive(Debug, Deserialize, Serialize)]
pub struct IngestParams {
    /// Collection to ingest into.
    pub collection: String,

    /// Source URI (file path or URL).
    pub source_uri: String,

    /// Document content.
    pub content: String,

    /// Content type (optional, auto-detected if not specified).
    pub content_type: Option<String>,
}

/// Collection parameters.
#[derive(Debug, Deserialize, Serialize)]
pub struct CollectionParams {
    /// Collection name.
    pub name: String,

    /// Description (optional).
    pub description: Option<String>,
}

/// Stats parameters.
#[derive(Debug, Deserialize, Serialize)]
pub struct StatsParams {
    /// Collection to get stats for (optional).
    pub collection: Option<String>,
}

/// Tool result.
#[derive(Debug, Serialize)]
pub struct ToolResult {
    /// Whether the operation was successful.
    pub success: bool,

    /// Result message or content.
    pub message: String,
}

impl ToolResult {
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
        }
    }
}

impl RagMcpServer {
    /// Create a new RAG MCP server with the given database path.
    pub fn new(db_path: impl Into<PathBuf>) -> Result<Self, rag_core::RagError> {
        let db_path = db_path.into();
        info!("Initializing RAG MCP server with database at {:?}", db_path);

        let store = Arc::new(SqliteStore::open(&db_path, 1)?);
        let embedder = Arc::new(MockEmbedder::new());
        let chunker = Arc::new(AdaptiveChunker::new());
        let engine = Arc::new(QueryEngine::new(store.clone(), embedder.clone()));

        Ok(Self {
            store,
            embedder,
            chunker,
            engine,
        })
    }

    /// Create a new RAG MCP server with an in-memory database.
    pub fn new_memory() -> Result<Self, rag_core::RagError> {
        info!("Initializing RAG MCP server with in-memory database");

        let store = Arc::new(SqliteStore::open_memory(1)?);
        let embedder = Arc::new(MockEmbedder::new());
        let chunker = Arc::new(AdaptiveChunker::new());
        let engine = Arc::new(QueryEngine::new(store.clone(), embedder.clone()));

        Ok(Self {
            store,
            embedder,
            chunker,
            engine,
        })
    }

    /// Get the server info.
    pub fn info() -> ServerInfo {
        ServerInfo {
            name: "rag-mcp".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Local RAG server with vector and keyword search".to_string(),
        }
    }

    /// List available tools.
    pub fn tools() -> Vec<ToolInfo> {
        vec![
            ToolInfo {
                name: "rag_search".to_string(),
                description: "Search the knowledge base for relevant documents".to_string(),
            },
            ToolInfo {
                name: "rag_ingest".to_string(),
                description: "Ingest a document into the knowledge base".to_string(),
            },
            ToolInfo {
                name: "rag_list_collections".to_string(),
                description: "List all collections".to_string(),
            },
            ToolInfo {
                name: "rag_create_collection".to_string(),
                description: "Create a new collection".to_string(),
            },
            ToolInfo {
                name: "rag_delete_collection".to_string(),
                description: "Delete a collection".to_string(),
            },
            ToolInfo {
                name: "rag_stats".to_string(),
                description: "Get statistics about the knowledge base".to_string(),
            },
        ]
    }

    /// Search the knowledge base.
    pub async fn search(&self, params: SearchParams) -> ToolResult {
        info!("Searching for: {:?}", params.query);

        // Use keyword-only search if vector search is not available
        let results = if self.store.vec_enabled() {
            let config = QueryConfig {
                top_k: params.top_k,
                collection: params.collection,
                ..Default::default()
            };
            self.engine.search(&params.query, config).await
        } else {
            self.engine
                .keyword_only_search(&params.query, params.top_k, params.collection.as_deref())
                .await
        };

        match results {
            Ok(results) => {
                let mut output = format!(
                    "Found {} results in {}ms:\n\n",
                    results.total_results, results.latency_ms
                );

                for result in results.results {
                    output.push_str(&format!(
                        "---\n[{}] {} (score: {:.3})\n",
                        result.rank, result.source_uri, result.score
                    ));
                    output.push_str(&format!(
                        "Lines {}-{}:\n```\n{}\n```\n\n",
                        result.chunk.start_line, result.chunk.end_line, result.chunk.content
                    ));
                }

                ToolResult::success(output)
            }
            Err(e) => ToolResult::error(format!("Search failed: {}", e)),
        }
    }

    /// Ingest a document into the knowledge base.
    pub async fn ingest(&self, params: IngestParams) -> ToolResult {
        info!(
            "Ingesting document: {} into {}",
            params.source_uri, params.collection
        );

        // Ensure collection exists
        match self.store.get_collection(&params.collection).await {
            Ok(None) => {
                return ToolResult::error(format!(
                    "Collection '{}' does not exist. Create it first.",
                    params.collection
                ));
            }
            Err(e) => return ToolResult::error(format!("Database error: {}", e)),
            Ok(Some(_)) => {}
        }

        // Determine content type
        let content_type = params
            .content_type
            .as_ref()
            .map(|ct| ContentType::from_path(ct))
            .unwrap_or_else(|| ContentType::from_path(&params.source_uri));

        // Create document
        let doc = Document::new(
            &params.collection,
            &params.source_uri,
            &params.content,
            content_type,
        );
        let doc_id = doc.id;

        // Insert document
        if let Err(e) = self.store.insert_document(doc).await {
            return ToolResult::error(format!("Failed to insert document: {}", e));
        }

        // Chunk the content
        let chunk_config = ChunkConfig {
            max_tokens: 512,
            min_tokens: 50,
            overlap_tokens: 0,
        };

        let chunk_data = match self
            .chunker
            .chunk(&params.content, content_type, &chunk_config)
        {
            Ok(data) => data,
            Err(e) => return ToolResult::error(format!("Chunking failed: {}", e)),
        };

        // Create chunks
        let mut chunks = Vec::with_capacity(chunk_data.len());
        for (idx, data) in chunk_data.into_iter().enumerate() {
            chunks.push(rag_core::Chunk::new(
                doc_id,
                idx as u32,
                &data.content,
                data.token_count as u32,
                data.start_line,
                data.end_line,
            ));
        }

        let num_chunks = chunks.len();

        // Insert chunks
        if let Err(e) = self.store.insert_chunks(&chunks).await {
            return ToolResult::error(format!("Failed to insert chunks: {}", e));
        }

        // Generate embeddings
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = match self.embedder.embed_documents(&chunk_texts).await {
            Ok(e) => e,
            Err(e) => return ToolResult::error(format!("Embedding failed: {}", e)),
        };

        // Insert embeddings if available
        if self.store.vec_enabled() {
            let chunk_ids: Vec<_> = chunks.iter().map(|c| c.id).collect();
            if let Err(e) = self.store.insert_embeddings(&chunk_ids, &embeddings).await {
                return ToolResult::error(format!("Failed to insert embeddings: {}", e));
            }
        }

        ToolResult::success(format!(
            "Successfully ingested '{}' with {} chunks.",
            params.source_uri, num_chunks
        ))
    }

    /// List all collections.
    pub async fn list_collections(&self) -> ToolResult {
        match self.store.list_collections().await {
            Ok(collections) => {
                if collections.is_empty() {
                    return ToolResult::success("No collections found.");
                }

                let mut output = format!("Found {} collections:\n\n", collections.len());
                for coll in collections {
                    output.push_str(&format!(
                        "- {}: {}\n",
                        coll.name,
                        coll.description.as_deref().unwrap_or("(no description)")
                    ));
                }

                ToolResult::success(output)
            }
            Err(e) => ToolResult::error(format!("Failed to list collections: {}", e)),
        }
    }

    /// Create a new collection.
    pub async fn create_collection(&self, params: CollectionParams) -> ToolResult {
        info!("Creating collection: {}", params.name);

        let collection = Collection::new(&params.name, params.description.as_deref());

        match self.store.create_collection(collection).await {
            Ok(()) => ToolResult::success(format!("Collection '{}' created.", params.name)),
            Err(rag_core::RagError::CollectionExists { name }) => {
                ToolResult::error(format!("Collection '{}' already exists.", name))
            }
            Err(e) => ToolResult::error(format!("Failed to create collection: {}", e)),
        }
    }

    /// Delete a collection.
    pub async fn delete_collection(&self, name: &str) -> ToolResult {
        info!("Deleting collection: {}", name);

        match self.store.delete_collection(name).await {
            Ok(()) => ToolResult::success(format!("Collection '{}' deleted.", name)),
            Err(rag_core::RagError::CollectionNotFound { name }) => {
                ToolResult::error(format!("Collection '{}' not found.", name))
            }
            Err(e) => ToolResult::error(format!("Failed to delete collection: {}", e)),
        }
    }

    /// Get statistics.
    pub async fn stats(&self, collection: Option<&str>) -> ToolResult {
        match self.store.get_stats(collection).await {
            Ok(stats) => {
                let mut output = String::new();

                if let Some(coll) = collection {
                    output.push_str(&format!("Statistics for collection '{}':\n\n", coll));
                } else {
                    output.push_str("Overall statistics:\n\n");
                }

                output.push_str(&format!("- Collections: {}\n", stats.collections));
                output.push_str(&format!("- Documents: {}\n", stats.documents));
                output.push_str(&format!("- Chunks: {}\n", stats.chunks));
                output.push_str(&format!("- Embeddings: {}\n", stats.embeddings));
                output.push_str(&format!(
                    "- Storage: {:.2} MB\n",
                    stats.storage_bytes as f64 / 1024.0 / 1024.0
                ));

                ToolResult::success(output)
            }
            Err(e) => ToolResult::error(format!("Failed to get stats: {}", e)),
        }
    }
}

/// Server info.
#[derive(Debug, Serialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub description: String,
}

/// Tool info.
#[derive(Debug, Serialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_creation() {
        let server = RagMcpServer::new_memory().unwrap();
        let info = RagMcpServer::info();
        assert_eq!(info.name, "rag-mcp");
    }

    #[tokio::test]
    async fn test_create_collection() {
        let server = RagMcpServer::new_memory().unwrap();

        let params = CollectionParams {
            name: "test".to_string(),
            description: Some("Test collection".to_string()),
        };

        let result = server.create_collection(params).await;
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_list_collections() {
        let server = RagMcpServer::new_memory().unwrap();

        // Create a collection first
        let params = CollectionParams {
            name: "test".to_string(),
            description: None,
        };
        server.create_collection(params).await;

        // List collections
        let result = server.list_collections().await;
        assert!(result.success);
        assert!(result.message.contains("test"));
    }

    #[tokio::test]
    async fn test_ingest_and_search() {
        let server = RagMcpServer::new_memory().unwrap();

        // Create collection
        let params = CollectionParams {
            name: "code".to_string(),
            description: Some("Code snippets".to_string()),
        };
        server.create_collection(params).await;

        // Ingest document
        let ingest_params = IngestParams {
            collection: "code".to_string(),
            source_uri: "file://test.rs".to_string(),
            content: "fn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
            content_type: Some("rust".to_string()),
        };
        let result = server.ingest(ingest_params).await;
        assert!(result.success, "Ingest failed: {}", result.message);

        // Search
        let search_params = SearchParams {
            query: "hello".to_string(),
            top_k: 5,
            collection: Some("code".to_string()),
        };
        let result = server.search(search_params).await;
        assert!(result.success, "Search failed: {}", result.message);
    }

    #[tokio::test]
    async fn test_stats() {
        let server = RagMcpServer::new_memory().unwrap();

        let result = server.stats(None).await;
        assert!(result.success);
        assert!(result.message.contains("Collections:"));
    }

    #[tokio::test]
    async fn test_tools_list() {
        let tools = RagMcpServer::tools();
        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "rag_search"));
    }
}
