# rag-mcp

MCP (Model Context Protocol) server implementation for exposing RAG functionality to Claude and other MCP clients.

## Overview

| Attribute | Value |
|-----------|-------|
| Purpose | MCP server interface |
| Dependencies | rmcp or mcp-rust-sdk, tokio, rag-core, rag-query |
| Est. Lines | ~800 |
| Confidence | HIGH |

This crate implements the MCP server that Claude Desktop and other clients connect to.

---

## Module Structure

```
rag-mcp/
├── src/
│   ├── lib.rs           # Public exports
│   ├── server.rs        # MCP server implementation
│   ├── tools.rs         # Tool handlers
│   ├── resources.rs     # Resource handlers
│   └── transport.rs     # stdio/SSE transport
└── Cargo.toml
```

---

## MCP Server Implementation

```rust
use mcp_rust_sdk::{
    Server, ServerConfig, Tool, Resource,
    ToolRequest, ToolResponse, ResourceRequest, ResourceResponse,
};
use rag_core::{Store, Result, RagError};
use rag_query::QueryEngine;
use std::sync::Arc;

/// RAG MCP Server.
pub struct RagMcpServer {
    /// Query engine for search.
    query_engine: Arc<QueryEngine>,

    /// Store for direct access.
    store: Arc<dyn Store>,

    /// Chunker for ingestion.
    chunker: Arc<dyn Chunker>,

    /// Embedder for ingestion.
    embedder: Arc<dyn Embedder>,

    /// Server configuration.
    config: ServerConfig,
}

impl RagMcpServer {
    pub fn new(
        query_engine: Arc<QueryEngine>,
        store: Arc<dyn Store>,
        chunker: Arc<dyn Chunker>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            query_engine,
            store,
            chunker,
            embedder,
            config: ServerConfig {
                name: "rag-mcp".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: Some("Local RAG system with hybrid search".to_string()),
            },
        }
    }

    /// Start the MCP server on stdio.
    pub async fn run_stdio(self) -> Result<()> {
        let server = Server::new(self.config.clone())
            .with_tools(self.get_tools())
            .with_resources(self.get_resources())
            .with_tool_handler(move |req| self.handle_tool(req))
            .with_resource_handler(move |req| self.handle_resource(req));

        server.run_stdio().await
            .map_err(|e| RagError::McpProtocol(e.to_string()))
    }

    /// Start the MCP server with SSE transport.
    pub async fn run_sse(self, addr: &str) -> Result<()> {
        let server = Server::new(self.config.clone())
            .with_tools(self.get_tools())
            .with_resources(self.get_resources())
            .with_tool_handler(move |req| self.handle_tool(req))
            .with_resource_handler(move |req| self.handle_resource(req));

        server.run_sse(addr).await
            .map_err(|e| RagError::McpProtocol(e.to_string()))
    }

    /// Get tool definitions.
    fn get_tools(&self) -> Vec<Tool> {
        vec![
            Tool {
                name: "rag_search".to_string(),
                description: "Search the RAG knowledge base using hybrid semantic and keyword search".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "collection": {
                            "type": "string",
                            "description": "Optional collection to search within"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                            "default": 10
                        },
                        "hybrid": {
                            "type": "boolean",
                            "description": "Use hybrid search combining vector and keyword (default: true)",
                            "default": true
                        }
                    },
                    "required": ["query"]
                }),
            },
            Tool {
                name: "rag_ingest".to_string(),
                description: "Ingest a document into the RAG knowledge base".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": "URI of the document (file://, https://, or data: for raw content)"
                        },
                        "collection": {
                            "type": "string",
                            "description": "Collection to add the document to"
                        },
                        "content_type": {
                            "type": "string",
                            "description": "Optional content type override (auto-detected if not provided)"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata to attach to the document"
                        }
                    },
                    "required": ["uri", "collection"]
                }),
            },
            Tool {
                name: "rag_delete".to_string(),
                description: "Delete a document from the RAG knowledge base".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "The document ID to delete"
                        }
                    },
                    "required": ["doc_id"]
                }),
            },
            Tool {
                name: "rag_list_collections".to_string(),
                description: "List all collections in the RAG knowledge base".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
            Tool {
                name: "rag_list_documents".to_string(),
                description: "List documents in a collection".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "The collection to list documents from"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of documents to return",
                            "default": 100
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Offset for pagination",
                            "default": 0
                        }
                    },
                    "required": ["collection"]
                }),
            },
            Tool {
                name: "rag_create_collection".to_string(),
                description: "Create a new collection".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Collection name (alphanumeric and hyphens)"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description"
                        }
                    },
                    "required": ["name"]
                }),
            },
            Tool {
                name: "rag_stats".to_string(),
                description: "Get statistics about the RAG knowledge base".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Optional collection to get stats for"
                        }
                    }
                }),
            },
        ]
    }

    /// Get resource definitions.
    fn get_resources(&self) -> Vec<Resource> {
        vec![
            Resource {
                uri: "rag://collections".to_string(),
                name: "RAG Collections".to_string(),
                description: Some("List of all collections with statistics".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            Resource {
                uri: "rag://collections/{name}".to_string(),
                name: "Collection Details".to_string(),
                description: Some("Details and document list for a specific collection".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            Resource {
                uri: "rag://documents/{id}".to_string(),
                name: "Document Content".to_string(),
                description: Some("Full content and metadata for a specific document".to_string()),
                mime_type: Some("application/json".to_string()),
            },
        ]
    }
}
```

---

## Tool Handlers

```rust
impl RagMcpServer {
    /// Handle tool invocations.
    async fn handle_tool(&self, request: ToolRequest) -> ToolResponse {
        let result = match request.name.as_str() {
            "rag_search" => self.handle_search(request.arguments).await,
            "rag_ingest" => self.handle_ingest(request.arguments).await,
            "rag_delete" => self.handle_delete(request.arguments).await,
            "rag_list_collections" => self.handle_list_collections().await,
            "rag_list_documents" => self.handle_list_documents(request.arguments).await,
            "rag_create_collection" => self.handle_create_collection(request.arguments).await,
            "rag_stats" => self.handle_stats(request.arguments).await,
            _ => Err(RagError::UnknownTool(request.name.clone())),
        };

        match result {
            Ok(content) => ToolResponse::success(content),
            Err(e) => ToolResponse::error(e.to_string()),
        }
    }

    /// Handle rag_search tool.
    async fn handle_search(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let query = args["query"].as_str()
            .ok_or_else(|| RagError::InvalidArgument("query is required".into()))?;

        let collection = args.get("collection").and_then(|v| v.as_str());
        let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let hybrid = args.get("hybrid").and_then(|v| v.as_bool()).unwrap_or(true);

        // Configure search
        let config = QueryConfig {
            final_k: top_k,
            hybrid_alpha: if hybrid { 0.5 } else { 1.0 },
            ..Default::default()
        };

        // Execute search
        let results = self.query_engine
            .search_with_config(query, collection, &config)
            .await?;

        // Format results for MCP
        let formatted: Vec<serde_json::Value> = results.results.iter()
            .enumerate()
            .map(|(i, r)| {
                serde_json::json!({
                    "rank": i + 1,
                    "score": r.score,
                    "source": r.chunk.doc_id.to_string(),
                    "collection": self.get_doc_collection(&r.chunk.doc_id),
                    "content": r.chunk.content,
                    "metadata": {
                        "lines": format!("{}-{}", r.chunk.metadata.line_start, r.chunk.metadata.line_end),
                        "tokens": r.chunk.token_count,
                        "chunk_index": r.chunk.chunk_index,
                    }
                })
            })
            .collect();

        Ok(serde_json::json!({
            "query": query,
            "total_results": formatted.len(),
            "results": formatted
        }))
    }

    /// Handle rag_ingest tool.
    async fn handle_ingest(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let uri = args["uri"].as_str()
            .ok_or_else(|| RagError::InvalidArgument("uri is required".into()))?;

        let collection = args["collection"].as_str()
            .ok_or_else(|| RagError::InvalidArgument("collection is required".into()))?;

        let content_type = args.get("content_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok());

        let metadata = args.get("metadata")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        // Load content
        let content = load_uri(uri).await?;

        // Detect content type
        let content_type = content_type
            .unwrap_or_else(|| ContentType::from_path(Path::new(uri)));

        // Create document
        let doc_id = Ulid::new();
        let document = Document::new(collection, uri, &content, content_type);

        // Chunk content
        let settings = ChunkingSettings::default();
        let chunk_outputs = self.chunker.chunk(&content, content_type, &settings)?;

        // Create chunks
        let chunks: Vec<Chunk> = chunk_outputs.into_iter()
            .enumerate()
            .map(|(i, output)| Chunk::from_output(doc_id, i as u32, output))
            .collect();

        // Embed chunks
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = self.embedder.embed(&texts)?;

        // Store everything
        self.store.insert_document(document.clone()).await?;
        self.store.insert_chunks(chunks.clone()).await?;
        self.store.insert_embeddings(
            chunks.iter().map(|c| c.id).zip(embeddings).collect()
        ).await?;

        Ok(serde_json::json!({
            "success": true,
            "doc_id": doc_id.to_string(),
            "collection": collection,
            "source_uri": uri,
            "content_type": format!("{:?}", content_type),
            "chunks_created": chunks.len(),
        }))
    }

    /// Handle rag_delete tool.
    async fn handle_delete(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let doc_id_str = args["doc_id"].as_str()
            .ok_or_else(|| RagError::InvalidArgument("doc_id is required".into()))?;

        let doc_id = Ulid::from_string(doc_id_str)
            .map_err(|_| RagError::InvalidArgument("Invalid doc_id format".into()))?;

        // Get document info before deletion
        let doc = self.store.get_document(doc_id).await?
            .ok_or(RagError::DocumentNotFound(doc_id))?;

        // Delete document (cascades to chunks)
        self.store.delete_document(doc_id).await?;

        Ok(serde_json::json!({
            "success": true,
            "deleted_doc_id": doc_id_str,
            "source_uri": doc.source_uri,
            "collection": doc.collection,
        }))
    }

    /// Handle rag_list_collections tool.
    async fn handle_list_collections(&self) -> Result<serde_json::Value> {
        let collections = self.store.list_collections().await?;

        let formatted: Vec<serde_json::Value> = collections.iter()
            .map(|c| {
                serde_json::json!({
                    "name": c.name,
                    "description": c.description,
                    "created_at": c.created_at,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "collections": formatted,
            "total": formatted.len(),
        }))
    }

    /// Handle rag_list_documents tool.
    async fn handle_list_documents(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let collection = args["collection"].as_str()
            .ok_or_else(|| RagError::InvalidArgument("collection is required".into()))?;

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as u32;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

        let documents = self.store.list_documents(collection, limit, offset).await?;

        let formatted: Vec<serde_json::Value> = documents.iter()
            .map(|d| {
                serde_json::json!({
                    "id": d.id.to_string(),
                    "source_uri": d.source_uri,
                    "content_type": format!("{:?}", d.content_type),
                    "created_at": d.created_at,
                    "updated_at": d.updated_at,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "collection": collection,
            "documents": formatted,
            "count": formatted.len(),
            "offset": offset,
            "limit": limit,
        }))
    }

    /// Handle rag_create_collection tool.
    async fn handle_create_collection(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let name = args["name"].as_str()
            .ok_or_else(|| RagError::InvalidArgument("name is required".into()))?;

        // Validate collection name
        if !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            return Err(RagError::InvalidCollectionName(name.to_string()));
        }

        let description = args.get("description").and_then(|v| v.as_str());

        let collection = Collection {
            name: name.to_string(),
            description: description.map(String::from),
            settings: CollectionSettings::default(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            hlc: HybridLogicalClock::new(0),
        };

        self.store.create_collection(collection).await?;

        Ok(serde_json::json!({
            "success": true,
            "name": name,
            "description": description,
        }))
    }

    /// Handle rag_stats tool.
    async fn handle_stats(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let collection = args.get("collection").and_then(|v| v.as_str());

        let stats = self.store.stats().await?;
        let doc_count = self.store.count_documents(collection).await?;
        let chunk_count = self.store.count_chunks(collection).await?;

        Ok(serde_json::json!({
            "collections": stats.collections,
            "documents": doc_count,
            "chunks": chunk_count,
            "embeddings": stats.embeddings,
            "storage_bytes": stats.storage_bytes,
            "filter": collection,
        }))
    }
}

/// Load content from URI.
async fn load_uri(uri: &str) -> Result<String> {
    if uri.starts_with("file://") {
        let path = uri.strip_prefix("file://").unwrap();
        std::fs::read_to_string(path)
            .map_err(|e| RagError::LoadFailed {
                uri: uri.to_string(),
                reason: e.to_string(),
            })
    } else if uri.starts_with("http://") || uri.starts_with("https://") {
        let response = reqwest::get(uri).await
            .map_err(|e| RagError::LoadFailed {
                uri: uri.to_string(),
                reason: e.to_string(),
            })?;

        response.text().await
            .map_err(|e| RagError::LoadFailed {
                uri: uri.to_string(),
                reason: e.to_string(),
            })
    } else if uri.starts_with("data:") {
        // data:text/plain;base64,... or data:,raw content
        let content = uri.strip_prefix("data:").unwrap();
        if let Some(comma_idx) = content.find(',') {
            let data = &content[comma_idx + 1..];
            if content[..comma_idx].contains("base64") {
                let decoded = base64::decode(data)
                    .map_err(|e| RagError::LoadFailed {
                        uri: uri.to_string(),
                        reason: e.to_string(),
                    })?;
                String::from_utf8(decoded)
                    .map_err(|e| RagError::LoadFailed {
                        uri: uri.to_string(),
                        reason: e.to_string(),
                    })
            } else {
                Ok(data.to_string())
            }
        } else {
            Err(RagError::InvalidUri(uri.to_string()))
        }
    } else {
        // Treat as file path
        std::fs::read_to_string(uri)
            .map_err(|e| RagError::LoadFailed {
                uri: uri.to_string(),
                reason: e.to_string(),
            })
    }
}
```

---

## Resource Handlers

```rust
impl RagMcpServer {
    /// Handle resource requests.
    async fn handle_resource(&self, request: ResourceRequest) -> ResourceResponse {
        let result = self.read_resource(&request.uri).await;

        match result {
            Ok((content, mime_type)) => ResourceResponse::success(content, mime_type),
            Err(e) => ResourceResponse::error(e.to_string()),
        }
    }

    /// Read a resource by URI.
    async fn read_resource(&self, uri: &str) -> Result<(String, String)> {
        if uri == "rag://collections" {
            return self.read_collections().await;
        }

        if uri.starts_with("rag://collections/") {
            let name = uri.strip_prefix("rag://collections/").unwrap();
            return self.read_collection(name).await;
        }

        if uri.starts_with("rag://documents/") {
            let id_str = uri.strip_prefix("rag://documents/").unwrap();
            return self.read_document(id_str).await;
        }

        Err(RagError::InvalidUri(uri.to_string()))
    }

    async fn read_collections(&self) -> Result<(String, String)> {
        let collections = self.store.list_collections().await?;

        let mut formatted = Vec::new();
        for c in collections {
            let doc_count = self.store.count_documents(Some(&c.name)).await?;
            let chunk_count = self.store.count_chunks(Some(&c.name)).await?;

            formatted.push(serde_json::json!({
                "name": c.name,
                "description": c.description,
                "documents": doc_count,
                "chunks": chunk_count,
                "created_at": c.created_at,
            }));
        }

        let content = serde_json::to_string_pretty(&formatted)?;
        Ok((content, "application/json".to_string()))
    }

    async fn read_collection(&self, name: &str) -> Result<(String, String)> {
        let collection = self.store.get_collection(name).await?
            .ok_or_else(|| RagError::CollectionNotFound(name.to_string()))?;

        let documents = self.store.list_documents(name, 100, 0).await?;

        let content = serde_json::to_string_pretty(&serde_json::json!({
            "name": collection.name,
            "description": collection.description,
            "settings": collection.settings,
            "created_at": collection.created_at,
            "documents": documents.iter().map(|d| {
                serde_json::json!({
                    "id": d.id.to_string(),
                    "source_uri": d.source_uri,
                    "content_type": format!("{:?}", d.content_type),
                })
            }).collect::<Vec<_>>(),
        }))?;

        Ok((content, "application/json".to_string()))
    }

    async fn read_document(&self, id_str: &str) -> Result<(String, String)> {
        let doc_id = Ulid::from_string(id_str)
            .map_err(|_| RagError::InvalidArgument("Invalid document ID".into()))?;

        let document = self.store.get_document(doc_id).await?
            .ok_or(RagError::DocumentNotFound(doc_id))?;

        let chunks = self.store.get_chunks(doc_id).await?;

        let content = serde_json::to_string_pretty(&serde_json::json!({
            "id": document.id.to_string(),
            "collection": document.collection,
            "source_uri": document.source_uri,
            "content_type": format!("{:?}", document.content_type),
            "metadata": document.metadata,
            "created_at": document.created_at,
            "updated_at": document.updated_at,
            "raw_content": document.raw_content,
            "chunks": chunks.iter().map(|c| {
                serde_json::json!({
                    "id": c.id.to_string(),
                    "index": c.chunk_index,
                    "content": c.content,
                    "tokens": c.token_count,
                    "lines": format!("{}-{}", c.metadata.line_start, c.metadata.line_end),
                })
            }).collect::<Vec<_>>(),
        }))?;

        Ok((content, "application/json".to_string()))
    }
}
```

---

## Claude Desktop Configuration

```json
{
  "mcpServers": {
    "rag": {
      "command": "rag-mcp",
      "args": ["--config", "~/.config/rag-mcp/config.toml"],
      "env": {}
    }
  }
}
```

Or with explicit paths:

```json
{
  "mcpServers": {
    "rag": {
      "command": "/path/to/rag-mcp",
      "args": [
        "--database", "/path/to/rag.db",
        "--model", "/path/to/nomic-embed-text-v1.5"
      ]
    }
  }
}
```

---

## Confidence Assessment

| Component | Confidence | Notes |
|-----------|------------|-------|
| MCP protocol | HIGH | Well-defined spec |
| Tool handlers | HIGH | Standard request/response |
| Resource handlers | HIGH | Simple data formatting |
| URI loading | HIGH | Standard protocols |
| stdio transport | HIGH | Simple I/O |
| SSE transport | MEDIUM | May need testing |

---

## Cargo.toml

```toml
[package]
name = "rag-mcp"
version = "0.1.0"
edition = "2021"
description = "MCP server for the RAG system"
license = "MIT"

[dependencies]
rag-core = { path = "../rag-core" }
rag-query = { path = "../rag-query" }
mcp-rust-sdk = "0.1"  # Or rmcp
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json"] }
base64 = "0.21"
tracing = "0.1"

[dev-dependencies]
tokio = { version = "1.0", features = ["macros", "rt-multi-thread"] }
```
