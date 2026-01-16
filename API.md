# API.md - MCP Interface Specification

Complete specification of the MCP tools and resources exposed by rag-mcp.

## Table of Contents

1. [Overview](#overview)
2. [Tools](#tools)
3. [Resources](#resources)
4. [Error Handling](#error-handling)
5. [Examples](#examples)

---

## Overview

The rag-mcp server exposes functionality through the Model Context Protocol (MCP). This enables Claude and other MCP clients to:

- Search the knowledge base
- Ingest new documents
- Manage collections
- Query statistics

### Server Information

```json
{
  "name": "rag-mcp",
  "version": "0.1.0",
  "description": "Local RAG system with hybrid search"
}
```

### Transport

- **Primary:** stdio (for Claude Desktop)
- **Alternative:** SSE over HTTP (for other clients)

---

## Tools

### rag_search

Search the knowledge base using hybrid semantic and keyword search.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The search query"
    },
    "collection": {
      "type": "string",
      "description": "Optional collection to search within. If omitted, searches all collections."
    },
    "top_k": {
      "type": "integer",
      "description": "Number of results to return",
      "default": 10,
      "minimum": 1,
      "maximum": 100
    },
    "hybrid": {
      "type": "boolean",
      "description": "Use hybrid search combining vector and keyword",
      "default": true
    }
  },
  "required": ["query"]
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The original query"
    },
    "total_results": {
      "type": "integer",
      "description": "Number of results returned"
    },
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "rank": {
            "type": "integer",
            "description": "Result rank (1-indexed)"
          },
          "score": {
            "type": "number",
            "description": "Relevance score (higher is better)"
          },
          "source": {
            "type": "string",
            "description": "Document ID"
          },
          "collection": {
            "type": "string",
            "description": "Collection name"
          },
          "content": {
            "type": "string",
            "description": "Chunk content"
          },
          "metadata": {
            "type": "object",
            "properties": {
              "lines": {
                "type": "string",
                "description": "Line range in source file"
              },
              "tokens": {
                "type": "integer",
                "description": "Token count"
              },
              "chunk_index": {
                "type": "integer",
                "description": "Position in document"
              }
            }
          }
        }
      }
    }
  }
}
```

**Example Request:**

```json
{
  "name": "rag_search",
  "arguments": {
    "query": "error handling in async functions",
    "collection": "code",
    "top_k": 5
  }
}
```

**Example Response:**

```json
{
  "query": "error handling in async functions",
  "total_results": 5,
  "results": [
    {
      "rank": 1,
      "score": 0.847,
      "source": "01HRE4KXQN...",
      "collection": "code",
      "content": "async fn handle_error(&self, err: Error) -> Result<()> {\n    match err {\n        Error::Timeout => self.retry().await?,\n        _ => return Err(err),\n    }\n    Ok(())\n}",
      "metadata": {
        "lines": "142-150",
        "tokens": 89,
        "chunk_index": 3
      }
    }
  ]
}
```

---

### rag_ingest

Ingest a document into the knowledge base.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "uri": {
      "type": "string",
      "description": "URI of the document. Supports: file://, https://, data: (for raw content)"
    },
    "collection": {
      "type": "string",
      "description": "Target collection name"
    },
    "content_type": {
      "type": "string",
      "description": "Content type override. Auto-detected if omitted.",
      "enum": ["rust", "python", "typescript", "javascript", "go", "java", "cpp", "markdown", "html", "json", "yaml", "plain_text"]
    },
    "metadata": {
      "type": "object",
      "description": "Optional metadata to attach to the document"
    }
  },
  "required": ["uri", "collection"]
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean"
    },
    "doc_id": {
      "type": "string",
      "description": "Generated document ID"
    },
    "collection": {
      "type": "string"
    },
    "source_uri": {
      "type": "string"
    },
    "content_type": {
      "type": "string"
    },
    "chunks_created": {
      "type": "integer"
    }
  }
}
```

**Example - File:**

```json
{
  "name": "rag_ingest",
  "arguments": {
    "uri": "file:///home/user/project/src/lib.rs",
    "collection": "code"
  }
}
```

**Example - URL:**

```json
{
  "name": "rag_ingest",
  "arguments": {
    "uri": "https://raw.githubusercontent.com/user/repo/main/README.md",
    "collection": "docs"
  }
}
```

**Example - Raw Content:**

```json
{
  "name": "rag_ingest",
  "arguments": {
    "uri": "data:,fn main() { println!(\"Hello\"); }",
    "collection": "code",
    "content_type": "rust"
  }
}
```

---

### rag_delete

Delete a document from the knowledge base.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "doc_id": {
      "type": "string",
      "description": "The document ID to delete (ULID format)"
    }
  },
  "required": ["doc_id"]
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean"
    },
    "deleted_doc_id": {
      "type": "string"
    },
    "source_uri": {
      "type": "string"
    },
    "collection": {
      "type": "string"
    }
  }
}
```

**Example:**

```json
{
  "name": "rag_delete",
  "arguments": {
    "doc_id": "01HRE4KXQN8Z3J2X5Y7A9B4C6D"
  }
}
```

---

### rag_list_collections

List all collections in the knowledge base.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {}
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "collections": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          },
          "description": {
            "type": "string"
          },
          "created_at": {
            "type": "integer",
            "description": "Unix timestamp (milliseconds)"
          }
        }
      }
    },
    "total": {
      "type": "integer"
    }
  }
}
```

**Example:**

```json
{
  "name": "rag_list_collections",
  "arguments": {}
}
```

**Response:**

```json
{
  "collections": [
    {
      "name": "code",
      "description": "Source code files",
      "created_at": 1705312200000
    },
    {
      "name": "docs",
      "description": "Documentation",
      "created_at": 1705312300000
    }
  ],
  "total": 2
}
```

---

### rag_list_documents

List documents in a collection.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "collection": {
      "type": "string",
      "description": "Collection name"
    },
    "limit": {
      "type": "integer",
      "description": "Maximum documents to return",
      "default": 100,
      "maximum": 1000
    },
    "offset": {
      "type": "integer",
      "description": "Pagination offset",
      "default": 0
    }
  },
  "required": ["collection"]
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "collection": {
      "type": "string"
    },
    "documents": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string"
          },
          "source_uri": {
            "type": "string"
          },
          "content_type": {
            "type": "string"
          },
          "created_at": {
            "type": "integer"
          },
          "updated_at": {
            "type": "integer"
          }
        }
      }
    },
    "count": {
      "type": "integer"
    },
    "offset": {
      "type": "integer"
    },
    "limit": {
      "type": "integer"
    }
  }
}
```

---

### rag_create_collection

Create a new collection.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Collection name (alphanumeric, hyphens, underscores)",
      "pattern": "^[a-zA-Z0-9_-]+$"
    },
    "description": {
      "type": "string",
      "description": "Optional description"
    }
  },
  "required": ["name"]
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean"
    },
    "name": {
      "type": "string"
    },
    "description": {
      "type": "string"
    }
  }
}
```

---

### rag_stats

Get statistics about the knowledge base.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "collection": {
      "type": "string",
      "description": "Optional collection to filter stats"
    }
  }
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "collections": {
      "type": "integer",
      "description": "Total number of collections"
    },
    "documents": {
      "type": "integer",
      "description": "Total documents (or in specified collection)"
    },
    "chunks": {
      "type": "integer",
      "description": "Total chunks"
    },
    "embeddings": {
      "type": "integer",
      "description": "Total embeddings"
    },
    "storage_bytes": {
      "type": "integer",
      "description": "Database size in bytes"
    },
    "filter": {
      "type": "string",
      "description": "Collection filter applied (if any)"
    }
  }
}
```

---

## Resources

### rag://collections

List all collections with statistics.

**MIME Type:** application/json

**Response:**

```json
[
  {
    "name": "code",
    "description": "Source code",
    "documents": 1234,
    "chunks": 12450,
    "created_at": 1705312200000
  }
]
```

---

### rag://collections/{name}

Get details for a specific collection.

**URI Template:** `rag://collections/code`

**MIME Type:** application/json

**Response:**

```json
{
  "name": "code",
  "description": "Source code files",
  "settings": {
    "chunking": {
      "max_tokens": 512,
      "min_tokens": 50,
      "overlap_tokens": 0
    },
    "search": {
      "top_k": 10,
      "hybrid": true,
      "hybrid_alpha": 0.5
    }
  },
  "created_at": 1705312200000,
  "documents": [
    {
      "id": "01HRE4KXQN...",
      "source_uri": "file:///src/lib.rs",
      "content_type": "Rust"
    }
  ]
}
```

---

### rag://documents/{id}

Get full document content and metadata.

**URI Template:** `rag://documents/01HRE4KXQN8Z3J2X5Y7A9B4C6D`

**MIME Type:** application/json

**Response:**

```json
{
  "id": "01HRE4KXQN8Z3J2X5Y7A9B4C6D",
  "collection": "code",
  "source_uri": "file:///src/lib.rs",
  "content_type": "Rust",
  "metadata": {},
  "created_at": 1705312200000,
  "updated_at": 1705312200000,
  "raw_content": "//! Library documentation\n\npub mod store;",
  "chunks": [
    {
      "id": "01HRE4KXQP...",
      "index": 0,
      "content": "//! Library documentation",
      "tokens": 5,
      "lines": "1-1"
    }
  ]
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "Document not found: 01HRE4KXQN..."
  }
}
```

### Error Codes

| Code | Description | HTTP Equivalent |
|------|-------------|-----------------|
| `DOCUMENT_NOT_FOUND` | Document ID doesn't exist | 404 |
| `COLLECTION_NOT_FOUND` | Collection doesn't exist | 404 |
| `COLLECTION_EXISTS` | Collection already exists | 409 |
| `INVALID_ARGUMENT` | Invalid parameter value | 400 |
| `INVALID_URI` | Cannot parse URI | 400 |
| `LOAD_FAILED` | Failed to load content | 502 |
| `TEXT_TOO_LONG` | Input exceeds max tokens | 400 |
| `DATABASE_ERROR` | Internal database error | 500 |
| `EMBEDDING_ERROR` | Embedding model error | 500 |

---

## Examples

### Complete Search Flow

**1. Create collection:**

```json
{
  "name": "rag_create_collection",
  "arguments": {
    "name": "my-project",
    "description": "My project documentation"
  }
}
```

**2. Ingest documents:**

```json
{
  "name": "rag_ingest",
  "arguments": {
    "uri": "file:///home/user/my-project/src/main.rs",
    "collection": "my-project"
  }
}
```

**3. Search:**

```json
{
  "name": "rag_search",
  "arguments": {
    "query": "main entry point",
    "collection": "my-project",
    "top_k": 3
  }
}
```

### Bulk Ingestion

For multiple files, call `rag_ingest` multiple times:

```json
// First file
{
  "name": "rag_ingest",
  "arguments": {
    "uri": "file:///src/lib.rs",
    "collection": "code"
  }
}

// Second file
{
  "name": "rag_ingest",
  "arguments": {
    "uri": "file:///src/main.rs",
    "collection": "code"
  }
}
```

### Cross-Collection Search

Omit `collection` to search everything:

```json
{
  "name": "rag_search",
  "arguments": {
    "query": "configuration options",
    "top_k": 10
  }
}
```

### Keyword-Only Search

Disable hybrid for exact matches:

```json
{
  "name": "rag_search",
  "arguments": {
    "query": "fn main()",
    "hybrid": false
  }
}
```

---

## Best Practices

### Query Formulation

1. **Natural language works best:**
   - Good: "How does error handling work?"
   - Okay: "error handling"
   - Poor: "err"

2. **Be specific when possible:**
   - Good: "async function error handling in Rust"
   - Okay: "error handling"

3. **Use collection filters:**
   - Filter to `code` for implementation details
   - Filter to `docs` for documentation

### Document Management

1. **Use meaningful collection names:**
   - `code` - Source files
   - `docs` - Documentation
   - `config` - Configuration files

2. **Avoid duplicate ingestion:**
   - The system deduplicates by content hash
   - Re-ingesting unchanged files is a no-op

3. **Update via re-ingest:**
   - Changed files are automatically updated
   - No explicit update tool needed

### Performance

1. **Limit result count:**
   - Default 10 is usually sufficient
   - Max 100 for broad searches

2. **Use hybrid search:**
   - Combines semantic + keyword
   - Best for most queries

3. **Filter by collection:**
   - Reduces search space
   - Faster for large knowledge bases
