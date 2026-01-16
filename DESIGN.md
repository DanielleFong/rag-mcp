# DESIGN.md - Architecture Deep Dive

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Model](#data-model)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Storage Design](#storage-design)
6. [Embedding Pipeline](#embedding-pipeline)
7. [Chunking Strategies](#chunking-strategies)
8. [Query Engine](#query-engine)
9. [Synchronization](#synchronization)
10. [MCP Interface](#mcp-interface)
11. [Error Handling](#error-handling)
12. [Performance Considerations](#performance-considerations)
13. [Security Model](#security-model)
14. [Confidence Assessment](#confidence-assessment)

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MCP Clients                                   │
│                    (Claude Desktop, CLI, etc.)                          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ MCP Protocol (stdio/SSE)
┌─────────────────────────────────▼───────────────────────────────────────┐
│                           rag-mcp                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   search    │  │   ingest    │  │   delete    │  │  list_*     │   │
│  │   tool      │  │   tool      │  │   tool      │  │  tools      │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
└─────────┼────────────────┼────────────────┼────────────────┼───────────┘
          │                │                │                │
┌─────────▼────────────────▼────────────────▼────────────────▼───────────┐
│                         rag-query                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Query Planning → Parallel Search → RRF Fusion → Context Build  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                         rag-store                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  Documents  │  │   Chunks    │  │  Vectors    │  │    FTS5     │   │
│  │   Table     │  │   Table     │  │ (sqlite-vec)│  │   Index     │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│     rag-embed     │  │     rag-chunk     │  │     rag-sync      │
│  ┌─────────────┐  │  │  ┌─────────────┐  │  │  ┌─────────────┐  │
│  │ ONNX Model  │  │  │  │ tree-sitter │  │  │  │  HTTP Sync  │  │
│  │  (nomic)    │  │  │  │  Recursive  │  │  │  │  HLC Clock  │  │
│  └─────────────┘  │  │  │  Semantic   │  │  │  └─────────────┘  │
└───────────────────┘  │  └─────────────┘  │  └───────────────────┘
                       └───────────────────┘
```

### 1.2 Design Principles

| Principle | Implementation | Confidence |
|-----------|---------------|------------|
| **Locality** | All processing happens locally; no network calls for core ops | HIGH |
| **Composability** | Each crate has a single responsibility, clear interfaces | HIGH |
| **Graceful degradation** | System remains functional if sync fails | HIGH |
| **Idempotency** | Re-ingesting same content is a no-op | HIGH |
| **Observability** | Structured logging, metrics via tracing | MEDIUM |

---

## 2. Data Model

### 2.1 Core Entities

```
┌─────────────────────────────────────────────────────────────┐
│                       Collection                            │
│  - name: String (unique identifier)                         │
│  - description: Option<String>                              │
│  - created_at: Timestamp                                    │
│  - settings: CollectionSettings                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:N
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Document                             │
│  - id: Ulid (globally unique, time-ordered)                 │
│  - collection: String (FK)                                  │
│  - source_uri: String (file://, https://, etc.)            │
│  - content_hash: [u8; 32] (blake3)                          │
│  - content_type: ContentType (enum)                         │
│  - metadata: serde_json::Value                              │
│  - created_at: Timestamp                                    │
│  - updated_at: Timestamp                                    │
│  - hlc: HybridLogicalClock                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:N
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Chunk                               │
│  - id: Ulid                                                 │
│  - doc_id: Ulid (FK)                                        │
│  - chunk_index: u32 (position in document)                  │
│  - content: String                                          │
│  - content_hash: [u8; 32]                                   │
│  - token_count: u32                                         │
│  - char_offset_start: u64                                   │
│  - char_offset_end: u64                                     │
│  - metadata: ChunkMetadata                                  │
│  - hlc: HybridLogicalClock                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:1
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Embedding                             │
│  - chunk_id: Ulid (FK, unique)                              │
│  - vector: [f32; 768]                                       │
│  - model_id: String ("nomic-embed-text-v1.5")              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Content Types

```rust
enum ContentType {
    // Code files - use AST-aware chunking
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Java,
    Cpp,
    C,

    // Documentation - use semantic chunking
    Markdown,
    RestructuredText,
    Html,
    PlainText,

    // Structured data - use record-based chunking
    Json,
    Yaml,
    Toml,

    // Conversations - use windowed chunking
    ChatLog,

    // Binary/unknown - skip or extract text
    Pdf,
    Unknown,
}
```

**Confidence: HIGH** - These content types cover the vast majority of use cases. Extension is straightforward.

### 2.3 Chunk Metadata

```rust
struct ChunkMetadata {
    // Position information
    line_start: u32,
    line_end: u32,

    // Code-specific (optional)
    syntax_node_type: Option<String>,  // "function", "class", "module"
    syntax_node_name: Option<String>,  // "parse_document", "UserService"
    language: Option<String>,

    // Document-specific (optional)
    heading_hierarchy: Option<Vec<String>>,  // ["Chapter 1", "Section 1.2"]

    // Extraction info
    chunking_strategy: String,

    // For overlap tracking
    overlaps_previous: bool,
    overlaps_next: bool,
}
```

**Confidence: HIGH** - Metadata schema is extensible via serde; can add fields without migration.

---

## 3. Component Architecture

### 3.1 Dependency Graph

```
                    rag-cli
                       │
                       ▼
                   rag-mcp
                       │
                       ▼
                  rag-query
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      rag-embed    rag-store    rag-sync
          │            │            │
          └────────────┼────────────┘
                       ▼
                  rag-chunk
                       │
                       ▼
                   rag-core
```

### 3.2 Crate Responsibilities

| Crate | Responsibility | External Deps | Lines Est. |
|-------|---------------|---------------|------------|
| rag-core | Types, traits, errors | serde, thiserror | ~800 |
| rag-chunk | Chunking strategies | tree-sitter, tokenizers | ~1500 |
| rag-store | Persistence layer | rusqlite, sqlite-vec | ~2000 |
| rag-embed | Embedding generation | ort, tokenizers | ~600 |
| rag-query | Search orchestration | (internal only) | ~1000 |
| rag-sync | Multi-node sync | reqwest, tokio | ~1200 |
| rag-mcp | MCP server | rmcp, tokio | ~800 |
| rag-cli | CLI interface | clap, tokio | ~500 |

**Total estimated: ~8,400 lines of Rust**

**Confidence: MEDIUM** - Line estimates are rough; actual may vary ±30%.

### 3.3 Trait Boundaries

```rust
// rag-core defines these traits; other crates implement them

/// Storage abstraction
trait Store: Send + Sync {
    async fn insert_document(&self, doc: Document) -> Result<Ulid>;
    async fn get_document(&self, id: Ulid) -> Result<Option<Document>>;
    async fn delete_document(&self, id: Ulid) -> Result<()>;
    async fn list_documents(&self, collection: &str, limit: u32, offset: u32) -> Result<Vec<Document>>;

    async fn insert_chunks(&self, chunks: Vec<Chunk>) -> Result<()>;
    async fn get_chunks(&self, doc_id: Ulid) -> Result<Vec<Chunk>>;
    async fn delete_chunks(&self, doc_id: Ulid) -> Result<()>;

    async fn insert_embeddings(&self, embeddings: Vec<(Ulid, Vec<f32>)>) -> Result<()>;
    async fn vector_search(&self, query: &[f32], k: usize, collection: Option<&str>) -> Result<Vec<(Ulid, f32)>>;
    async fn keyword_search(&self, query: &str, k: usize, collection: Option<&str>) -> Result<Vec<(Ulid, f32)>>;
}

/// Embedding model abstraction
trait Embedder: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn embed_query(&self, query: &str) -> Result<Vec<f32>>;
    fn dimension(&self) -> usize;
    fn max_tokens(&self) -> usize;
}

/// Chunking abstraction
trait Chunker: Send + Sync {
    fn chunk(&self, content: &str, content_type: ContentType) -> Result<Vec<ChunkOutput>>;
}

/// Sync abstraction
trait SyncPeer: Send + Sync {
    async fn get_watermark(&self) -> Result<HybridLogicalClock>;
    async fn get_changes_since(&self, hlc: HybridLogicalClock) -> Result<ChangeSet>;
    async fn push_changes(&self, changes: ChangeSet) -> Result<()>;
}
```

**Confidence: HIGH** - Trait design follows Rust idioms; allows easy testing via mocks.

---

## 4. Data Flow

### 4.1 Ingestion Flow

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Input  │────▶│   Loader    │────▶│  Chunker    │────▶│  Embedder   │
│ (URI)   │     │             │     │             │     │             │
└─────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                      │                   │                   │
                      ▼                   ▼                   ▼
                ┌───────────┐       ┌───────────┐       ┌───────────┐
                │ Document  │       │  Chunks   │       │ Embeddings│
                │  Record   │       │  Records  │       │  Records  │
                └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
                      │                   │                   │
                      └───────────────────┼───────────────────┘
                                          ▼
                                    ┌───────────┐
                                    │   Store   │
                                    │ (SQLite)  │
                                    └───────────┘
```

**Detailed Steps:**

1. **Load** (rag-store or rag-cli)
   - Parse URI scheme (file://, https://, raw:)
   - Read content into memory
   - Compute blake3 hash
   - Detect content type (extension + magic bytes)

2. **Deduplicate** (rag-store)
   - Check if document with same source_uri exists
   - If exists and content_hash matches: return existing doc (no-op)
   - If exists and content_hash differs: trigger incremental update
   - If not exists: proceed with full ingestion

3. **Chunk** (rag-chunk)
   - Select chunker based on content type
   - Generate chunks with metadata
   - Compute per-chunk content hashes

4. **Embed** (rag-embed)
   - Batch chunks (32 at a time)
   - Run through ONNX model
   - Return 768-dim vectors

5. **Persist** (rag-store)
   - Begin transaction
   - Insert/update document record
   - Insert chunks with FTS5 indexing
   - Insert embeddings into sqlite-vec
   - Commit transaction

**Confidence: HIGH** - Standard RAG pipeline; well-understood patterns.

### 4.2 Incremental Update Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Incremental Update                              │
└─────────────────────────────────────────────────────────────────────┘

     Existing Doc                         New Content
          │                                    │
          ▼                                    ▼
   ┌─────────────┐                      ┌─────────────┐
   │ Old Chunks  │                      │ New Chunks  │
   │ [A, B, C]   │                      │ [A, B', D]  │
   └─────────────┘                      └─────────────┘
          │                                    │
          └──────────────┬─────────────────────┘
                         ▼
                  ┌─────────────┐
                  │    Diff     │
                  │  (by hash)  │
                  └─────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │  Keep A   │   │ Delete C  │   │ Add B', D │
   │ (no-op)   │   │           │   │ (embed)   │
   └───────────┘   └───────────┘   └───────────┘
```

**Algorithm:**

```python
def incremental_update(doc_id, new_content):
    old_chunks = store.get_chunks(doc_id)
    new_chunks = chunker.chunk(new_content)

    old_hashes = {c.content_hash: c for c in old_chunks}
    new_hashes = {c.content_hash: c for c in new_chunks}

    to_keep = set(old_hashes.keys()) & set(new_hashes.keys())
    to_delete = set(old_hashes.keys()) - to_keep
    to_add = set(new_hashes.keys()) - to_keep

    # Delete removed chunks and their embeddings
    for h in to_delete:
        store.delete_chunk(old_hashes[h].id)

    # Embed and insert new chunks
    new_chunk_list = [new_hashes[h] for h in to_add]
    embeddings = embedder.embed([c.content for c in new_chunk_list])
    store.insert_chunks(new_chunk_list)
    store.insert_embeddings(zip([c.id for c in new_chunk_list], embeddings))

    # Update chunk indices for kept chunks (positions may have shifted)
    for h in to_keep:
        old_chunk = old_hashes[h]
        new_chunk = new_hashes[h]
        if old_chunk.chunk_index != new_chunk.chunk_index:
            store.update_chunk_index(old_chunk.id, new_chunk.chunk_index)
```

**Confidence: MEDIUM** - Logic is sound but edge cases exist (chunk boundary shifts).

### 4.3 Query Flow

```
┌─────────┐     ┌─────────────┐     ┌─────────────────────────────────┐
│  Query  │────▶│   Embed     │────▶│         Parallel Search         │
│ String  │     │   Query     │     │  ┌─────────┐    ┌─────────┐    │
└─────────┘     └─────────────┘     │  │ Vector  │    │  BM25   │    │
                                    │  │ Search  │    │ Search  │    │
                                    │  └────┬────┘    └────┬────┘    │
                                    │       │              │         │
                                    │       └──────┬───────┘         │
                                    │              ▼                 │
                                    │        ┌───────────┐           │
                                    │        │ RRF Merge │           │
                                    │        └─────┬─────┘           │
                                    └──────────────┼─────────────────┘
                                                   ▼
                                            ┌─────────────┐
                                            │  Context    │
                                            │  Assembly   │
                                            └─────────────┘
                                                   │
                                                   ▼
                                            ┌─────────────┐
                                            │   Results   │
                                            └─────────────┘
```

**Confidence: HIGH** - Hybrid search with RRF is a proven technique.

---

## 5. Storage Design

### 5.1 SQLite Schema

```sql
-- Enable required extensions
-- sqlite-vec must be loaded at runtime

-- Core tables
CREATE TABLE IF NOT EXISTS collections (
    name TEXT PRIMARY KEY,
    description TEXT,
    settings TEXT NOT NULL DEFAULT '{}',  -- JSON
    created_at INTEGER NOT NULL,
    hlc BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,                   -- ULID as text
    collection TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    source_uri TEXT NOT NULL,
    content_hash BLOB NOT NULL,            -- 32 bytes blake3
    content_type TEXT NOT NULL,
    raw_content TEXT,                      -- Optional: store original content
    metadata TEXT NOT NULL DEFAULT '{}',   -- JSON
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    hlc BLOB NOT NULL,

    UNIQUE(collection, source_uri)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,                   -- ULID
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash BLOB NOT NULL,
    token_count INTEGER NOT NULL,
    char_offset_start INTEGER NOT NULL,
    char_offset_end INTEGER NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',   -- JSON ChunkMetadata
    hlc BLOB NOT NULL,

    UNIQUE(doc_id, chunk_index)
);

-- sqlite-vec virtual table for vector search
-- Note: sqlite-vec uses vec0 module
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding float[768] distance_metric=cosine
);

-- FTS5 for keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', OLD.rowid, OLD.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', OLD.rowid, OLD.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
END;

-- Indices for common queries
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
CREATE INDEX IF NOT EXISTS idx_documents_source_uri ON documents(source_uri);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_hlc ON documents(hlc);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_hlc ON chunks(hlc);

-- Sync tracking table
CREATE TABLE IF NOT EXISTS sync_peers (
    peer_id TEXT PRIMARY KEY,
    endpoint TEXT NOT NULL,
    last_sync_hlc BLOB,
    last_sync_at INTEGER,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    peer_id TEXT NOT NULL REFERENCES sync_peers(peer_id),
    direction TEXT NOT NULL,  -- 'push' or 'pull'
    changes_count INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    status TEXT NOT NULL,     -- 'success', 'failed', 'partial'
    error_message TEXT
);
```

**Confidence: HIGH** - Schema is straightforward; sqlite-vec syntax verified against docs.

### 5.2 Storage Estimates

| Component | Per-Chunk | 100K Docs × 10 Chunks |
|-----------|-----------|----------------------|
| Chunk content (avg 300 chars) | 300 B | 300 MB |
| Chunk metadata | 200 B | 200 MB |
| Embedding (768 × f32) | 3,072 B | 3 GB |
| FTS5 index | ~100 B | 100 MB |
| sqlite-vec index | ~500 B | 500 MB |
| **Total** | ~4.2 KB | **~4.1 GB** |

**Confidence: MEDIUM** - Estimates based on typical data; actual varies with content.

### 5.3 sqlite-vec Considerations

**Strengths:**
- Single-file database (portable)
- No external server process
- Good performance up to ~1M vectors
- Supports multiple distance metrics

**Limitations:**
- No built-in sharding (single-node)
- Memory-maps index file (needs RAM ~= index size)
- Limited to SQLite's concurrency model (single writer)

**Mitigations:**
- Use WAL mode for better read concurrency
- Connection pooling for parallel reads
- Write batching to reduce transaction overhead

**Confidence: HIGH** - Well-documented limitations with known workarounds.

---

## 6. Embedding Pipeline

### 6.1 Model Selection

| Model | Dims | Context | Quality | Speed | Choice |
|-------|------|---------|---------|-------|--------|
| nomic-embed-text-v1.5 | 768 | 8192 | Good | Fast | **Selected** |
| bge-base-en-v1.5 | 768 | 512 | Good | Fast | Limited context |
| e5-large-v2 | 1024 | 512 | Better | Slower | Larger vectors |
| gte-large | 1024 | 512 | Better | Slower | Larger vectors |

**Why nomic-embed-text-v1.5:**
- 8192 token context handles most code files
- 768 dimensions balances quality vs storage
- Apache 2.0 licensed
- Good benchmark performance
- Available as ONNX

**Confidence: HIGH** - Nomic is well-regarded; 8K context is key differentiator.

### 6.2 ONNX Integration

```rust
// Pseudocode for embedding pipeline

struct OnnxEmbedder {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
    max_tokens: usize,
    dimension: usize,
}

impl OnnxEmbedder {
    fn new(model_path: &Path) -> Result<Self> {
        let session = ort::Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let tokenizer = tokenizers::Tokenizer::from_file(
            model_path.parent().unwrap().join("tokenizer.json")
        )?;

        Ok(Self {
            session,
            tokenizer,
            max_tokens: 8192,
            dimension: 768,
        })
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Tokenize
        let encodings = self.tokenizer.encode_batch(texts, true)?;

        // Prepare inputs
        let input_ids: Vec<Vec<i64>> = encodings.iter()
            .map(|e| e.get_ids().iter().map(|&id| id as i64).collect())
            .collect();

        let attention_mask: Vec<Vec<i64>> = encodings.iter()
            .map(|e| e.get_attention_mask().iter().map(|&m| m as i64).collect())
            .collect();

        // Pad to max length in batch
        let max_len = input_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);
        let padded_ids = pad_sequences(&input_ids, max_len, 0);
        let padded_mask = pad_sequences(&attention_mask, max_len, 0);

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input_ids" => padded_ids,
            "attention_mask" => padded_mask,
        ]?)?;

        // Extract embeddings (mean pooling over non-padding tokens)
        let embeddings = mean_pool(&outputs["last_hidden_state"], &padded_mask);

        // L2 normalize
        let normalized = l2_normalize(&embeddings);

        Ok(normalized)
    }
}
```

**Confidence: MEDIUM** - ONNX integration is straightforward but model-specific; may need adjustment.

### 6.3 Query vs Document Embedding

Nomic uses a prefix system to distinguish queries from documents:

```rust
impl Embedder for OnnxEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // For documents: prefix with "search_document: "
        let prefixed: Vec<String> = texts.iter()
            .map(|t| format!("search_document: {}", t))
            .collect();
        self.embed_batch(&prefixed.iter().map(|s| s.as_str()).collect::<Vec<_>>())
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        // For queries: prefix with "search_query: "
        let prefixed = format!("search_query: {}", query);
        let result = self.embed_batch(&[&prefixed])?;
        Ok(result.into_iter().next().unwrap())
    }
}
```

**Confidence: HIGH** - This is documented in nomic's model card.

### 6.4 Batching Strategy

```
Batch Size vs Throughput (estimated)

Batch Size | Tokens/Batch | Memory | Throughput
-----------|--------------|--------|------------
1          | 8192         | 200 MB | 5 chunks/sec
8          | 65536        | 500 MB | 25 chunks/sec
32         | 262144       | 1.5 GB | 50 chunks/sec
64         | 524288       | 3 GB   | 55 chunks/sec (diminishing returns)
```

**Recommendation:** Batch size 32 as default, configurable.

**Confidence: MEDIUM** - Numbers are estimates; actual depends on hardware.

---

## 7. Chunking Strategies

### 7.1 Strategy Selection Matrix

| Content Type | Strategy | Rationale |
|--------------|----------|-----------|
| Rust, Python, TS, JS, Go, Java, C, C++ | AST-aware | Preserve semantic boundaries |
| Markdown, RST | Heading-aware | Respect document structure |
| HTML | DOM-aware + text extraction | Handle nested structure |
| PlainText | Sentence-based recursive | Natural breakpoints |
| JSON, YAML, TOML | Record-based | Keep objects intact |
| ChatLog | Sliding window | Preserve conversation flow |
| PDF | Page + paragraph | Physical boundaries |
| Unknown | Recursive character | Fallback |

### 7.2 AST-Aware Chunking (Code)

```rust
struct AstChunker {
    max_tokens: usize,      // 512 default
    min_tokens: usize,      // 50 default
    overlap_tokens: usize,  // 0 for code (boundaries are semantic)
}

impl AstChunker {
    fn chunk(&self, code: &str, language: Language) -> Result<Vec<ChunkOutput>> {
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language.tree_sitter_grammar())?;

        let tree = parser.parse(code, None)
            .ok_or(ChunkError::ParseFailed)?;

        let mut chunks = Vec::new();
        self.visit_node(tree.root_node(), code, &mut chunks)?;

        // Merge small chunks
        self.merge_small_chunks(&mut chunks);

        Ok(chunks)
    }

    fn visit_node(&self, node: Node, source: &str, chunks: &mut Vec<ChunkOutput>) -> Result<()> {
        // Chunk boundaries: functions, classes, impl blocks, modules
        let boundary_kinds = [
            "function_item", "function_definition",  // Rust, Python
            "impl_item", "class_definition",         // Rust, Python
            "function_declaration", "class_declaration", // JS/TS
            "method_definition", "function_declaration", // Go, Java
        ];

        if boundary_kinds.contains(&node.kind()) {
            let text = &source[node.byte_range()];
            let token_count = count_tokens(text);

            if token_count <= self.max_tokens {
                // Whole node fits in one chunk
                chunks.push(ChunkOutput {
                    content: text.to_string(),
                    token_count,
                    char_offset_start: node.start_byte() as u64,
                    char_offset_end: node.end_byte() as u64,
                    metadata: ChunkMetadata {
                        syntax_node_type: Some(node.kind().to_string()),
                        syntax_node_name: extract_name(node, source),
                        line_start: node.start_position().row as u32,
                        line_end: node.end_position().row as u32,
                        ..Default::default()
                    },
                });
            } else {
                // Node too large; recurse into children
                for child in node.children(&mut node.walk()) {
                    self.visit_node(child, source, chunks)?;
                }
            }
        } else {
            // Not a boundary; recurse
            for child in node.children(&mut node.walk()) {
                self.visit_node(child, source, chunks)?;
            }
        }

        Ok(())
    }
}
```

**Confidence: HIGH** - tree-sitter is battle-tested; this pattern is used by many tools.

### 7.3 Semantic Chunking (Markdown)

```rust
struct SemanticChunker {
    max_tokens: usize,      // 512 default
    min_tokens: usize,      // 100 default
    overlap_tokens: usize,  // 50 default
}

impl SemanticChunker {
    fn chunk(&self, markdown: &str) -> Result<Vec<ChunkOutput>> {
        let ast = pulldown_cmark::Parser::new(markdown);

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_headings: Vec<String> = Vec::new();
        let mut chunk_start = 0;

        for (event, range) in ast.into_offset_iter() {
            match event {
                Event::Start(Tag::Heading(level, _, _)) => {
                    // Flush current chunk
                    if !current_chunk.is_empty() {
                        chunks.push(self.make_chunk(
                            &current_chunk,
                            chunk_start,
                            range.start,
                            &current_headings,
                        ));
                        current_chunk.clear();
                    }

                    // Update heading hierarchy
                    let level = level as usize;
                    current_headings.truncate(level - 1);
                    chunk_start = range.start;
                }
                Event::Text(text) => {
                    current_chunk.push_str(&text);
                }
                Event::End(Tag::Heading(..)) => {
                    current_headings.push(current_chunk.trim().to_string());
                    current_chunk.clear();
                }
                Event::SoftBreak | Event::HardBreak => {
                    current_chunk.push('\n');
                }
                _ => {}
            }

            // Check if chunk exceeds max tokens
            if count_tokens(&current_chunk) > self.max_tokens {
                // Split at sentence boundary
                let (chunk, remainder) = split_at_sentence(&current_chunk, self.max_tokens);
                chunks.push(self.make_chunk(&chunk, chunk_start, range.start, &current_headings));
                current_chunk = remainder;
                chunk_start = range.start;
            }
        }

        // Flush final chunk
        if !current_chunk.is_empty() {
            chunks.push(self.make_chunk(
                &current_chunk,
                chunk_start,
                markdown.len(),
                &current_headings,
            ));
        }

        Ok(chunks)
    }
}
```

**Confidence: MEDIUM** - Markdown parsing is well-understood but edge cases exist (code blocks, nested lists).

### 7.4 Sliding Window (Chat)

```rust
struct WindowChunker {
    window_tokens: usize,   // 128 default
    overlap_tokens: usize,  // 64 default (50% overlap)
}

impl WindowChunker {
    fn chunk(&self, text: &str) -> Result<Vec<ChunkOutput>> {
        let tokens = tokenize(text);
        let stride = self.window_tokens - self.overlap_tokens;

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < tokens.len() {
            let end = (start + self.window_tokens).min(tokens.len());
            let chunk_tokens = &tokens[start..end];
            let content = detokenize(chunk_tokens);

            chunks.push(ChunkOutput {
                content,
                token_count: chunk_tokens.len() as u32,
                metadata: ChunkMetadata {
                    overlaps_previous: start > 0,
                    overlaps_next: end < tokens.len(),
                    ..Default::default()
                },
                ..Default::default()
            });

            start += stride;
        }

        Ok(chunks)
    }
}
```

**Confidence: HIGH** - Sliding window is simple and well-understood.

---

## 8. Query Engine

### 8.1 Hybrid Search Architecture

```rust
struct QueryEngine {
    store: Arc<dyn Store>,
    embedder: Arc<dyn Embedder>,
    config: QueryConfig,
}

struct QueryConfig {
    vector_k: usize,        // 50 default - candidates from vector search
    keyword_k: usize,       // 50 default - candidates from BM25
    rrf_k: f32,             // 60.0 - RRF constant
    final_k: usize,         // 10 default - results to return
    hybrid_alpha: f32,      // 0.5 - weight between vector (1.0) and keyword (0.0)
    expand_context: bool,   // true - include adjacent chunks
    max_context_tokens: usize, // 4000 - max tokens in result
}

impl QueryEngine {
    async fn search(&self, query: &str, collection: Option<&str>) -> Result<SearchResults> {
        // 1. Embed query
        let query_embedding = self.embedder.embed_query(query)?;

        // 2. Parallel search
        let (vector_results, keyword_results) = tokio::join!(
            self.store.vector_search(&query_embedding, self.config.vector_k, collection),
            self.store.keyword_search(query, self.config.keyword_k, collection),
        );

        let vector_results = vector_results?;
        let keyword_results = keyword_results?;

        // 3. RRF fusion
        let fused = self.reciprocal_rank_fusion(&vector_results, &keyword_results);

        // 4. Fetch chunk content
        let mut results = Vec::new();
        for (chunk_id, score) in fused.into_iter().take(self.config.final_k) {
            let chunk = self.store.get_chunk(chunk_id).await?;
            results.push(SearchResult { chunk, score });
        }

        // 5. Expand context
        if self.config.expand_context {
            results = self.expand_with_adjacent(results).await?;
        }

        // 6. Deduplicate overlapping chunks
        results = self.deduplicate_overlaps(results);

        // 7. Truncate to token limit
        results = self.truncate_to_limit(results, self.config.max_context_tokens);

        Ok(SearchResults { results })
    }

    fn reciprocal_rank_fusion(
        &self,
        vector: &[(Ulid, f32)],
        keyword: &[(Ulid, f32)],
    ) -> Vec<(Ulid, f32)> {
        let mut scores: HashMap<Ulid, f32> = HashMap::new();
        let k = self.config.rrf_k;

        // Vector scores
        for (rank, (id, _)) in vector.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f32 + 1.0);
            *scores.entry(*id).or_default() += rrf_score * self.config.hybrid_alpha;
        }

        // Keyword scores
        for (rank, (id, _)) in keyword.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f32 + 1.0);
            *scores.entry(*id).or_default() += rrf_score * (1.0 - self.config.hybrid_alpha);
        }

        // Sort by fused score
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        results
    }
}
```

**Confidence: HIGH** - RRF is a standard technique; implementation is straightforward.

### 8.2 BM25 via FTS5

```sql
-- Query with BM25 ranking
SELECT
    c.id,
    c.content,
    bm25(chunks_fts) AS score
FROM chunks_fts
JOIN chunks c ON chunks_fts.rowid = c.rowid
WHERE chunks_fts MATCH ?
ORDER BY score
LIMIT ?;
```

**FTS5 Query Syntax:**
- Simple: `"error handling"`
- Boolean: `rust AND error`
- Phrase: `"async fn"`
- Prefix: `config*`
- Column filter: `content:error`

**Confidence: HIGH** - FTS5 is well-documented and widely used.

### 8.3 Vector Search via sqlite-vec

```sql
-- KNN query with cosine similarity
SELECT
    chunk_id,
    distance
FROM vec_chunks
WHERE embedding MATCH ?
ORDER BY distance
LIMIT ?;
```

**Note:** sqlite-vec returns distance (lower is better), not similarity (higher is better).
For cosine, `similarity = 1 - distance`.

**Confidence: HIGH** - sqlite-vec syntax verified against documentation.

### 8.4 Context Expansion

```rust
async fn expand_with_adjacent(&self, results: Vec<SearchResult>) -> Result<Vec<SearchResult>> {
    let mut expanded = Vec::new();

    for result in results {
        let doc_id = result.chunk.doc_id;
        let chunk_index = result.chunk.chunk_index;

        // Fetch adjacent chunks
        let prev = if chunk_index > 0 {
            self.store.get_chunk_by_index(doc_id, chunk_index - 1).await?
        } else {
            None
        };

        let next = self.store.get_chunk_by_index(doc_id, chunk_index + 1).await?;

        // Add with reduced scores
        if let Some(prev) = prev {
            expanded.push(SearchResult {
                chunk: prev,
                score: result.score * 0.5,  // Lower weight for adjacent
                is_context: true,
            });
        }

        expanded.push(result);

        if let Some(next) = next {
            expanded.push(SearchResult {
                chunk: next,
                score: result.score * 0.5,
                is_context: true,
            });
        }
    }

    // Sort by doc_id, chunk_index to maintain document order
    expanded.sort_by(|a, b| {
        (a.chunk.doc_id, a.chunk.chunk_index).cmp(&(b.chunk.doc_id, b.chunk.chunk_index))
    });

    Ok(expanded)
}
```

**Confidence: MEDIUM** - Context expansion is valuable but scoring heuristics may need tuning.

---

## 9. Synchronization

### 9.1 Hybrid Logical Clocks

HLC combines physical time with logical counters to provide:
- Causality tracking (if A happens before B, HLC(A) < HLC(B))
- Bounded clock drift (within epsilon of physical time)
- Unique ordering across nodes

```rust
struct HybridLogicalClock {
    wall_time: u64,      // Unix millis
    logical: u32,        // Logical counter
    node_id: u16,        // Unique node identifier
}

impl HybridLogicalClock {
    fn new(node_id: u16) -> Self {
        Self {
            wall_time: current_time_millis(),
            logical: 0,
            node_id,
        }
    }

    fn tick(&mut self) -> HybridLogicalClock {
        let now = current_time_millis();

        if now > self.wall_time {
            self.wall_time = now;
            self.logical = 0;
        } else {
            self.logical += 1;
        }

        self.clone()
    }

    fn update(&mut self, remote: &HybridLogicalClock) -> HybridLogicalClock {
        let now = current_time_millis();

        if now > self.wall_time && now > remote.wall_time {
            self.wall_time = now;
            self.logical = 0;
        } else if self.wall_time == remote.wall_time {
            self.logical = self.logical.max(remote.logical) + 1;
        } else if self.wall_time > remote.wall_time {
            self.logical += 1;
        } else {
            self.wall_time = remote.wall_time;
            self.logical = remote.logical + 1;
        }

        self.clone()
    }

    fn to_bytes(&self) -> [u8; 14] {
        let mut buf = [0u8; 14];
        buf[0..8].copy_from_slice(&self.wall_time.to_be_bytes());
        buf[8..12].copy_from_slice(&self.logical.to_be_bytes());
        buf[12..14].copy_from_slice(&self.node_id.to_be_bytes());
        buf
    }
}
```

**Confidence: HIGH** - HLC is a well-studied algorithm; implementation is standard.

### 9.2 Sync Protocol

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Sync Protocol                                   │
└─────────────────────────────────────────────────────────────────────────┘

Node A                                                          Node B
   │                                                               │
   │  GET /sync/watermark                                          │
   │──────────────────────────────────────────────────────────────▶│
   │                                                               │
   │  { "hlc": "00018d9e8f3c00000001000a" }                       │
   │◀──────────────────────────────────────────────────────────────│
   │                                                               │
   │  GET /sync/changes?since=00018d9e8f3c00000001000a            │
   │──────────────────────────────────────────────────────────────▶│
   │                                                               │
   │  { "changes": [...], "watermark": "00018d9e900000000003000a" }│
   │◀──────────────────────────────────────────────────────────────│
   │                                                               │
   │  [Apply changes locally]                                      │
   │                                                               │
   │  POST /sync/changes                                           │
   │  { "changes": [...], "watermark": "00018d9e8ff000000002000b" }│
   │──────────────────────────────────────────────────────────────▶│
   │                                                               │
   │  { "accepted": true, "conflicts": [] }                        │
   │◀──────────────────────────────────────────────────────────────│
   │                                                               │
```

### 9.3 Change Set Format

```rust
struct ChangeSet {
    changes: Vec<Change>,
    watermark: HybridLogicalClock,
}

enum Change {
    DocumentInsert {
        document: Document,
        chunks: Vec<Chunk>,
        embeddings: Vec<(Ulid, Vec<f32>)>,
    },
    DocumentUpdate {
        document: Document,
        chunks_to_delete: Vec<Ulid>,
        chunks_to_insert: Vec<Chunk>,
        embeddings_to_delete: Vec<Ulid>,
        embeddings_to_insert: Vec<(Ulid, Vec<f32>)>,
    },
    DocumentDelete {
        doc_id: Ulid,
    },
    CollectionCreate {
        collection: Collection,
    },
    CollectionDelete {
        name: String,
    },
}
```

### 9.4 Conflict Resolution

**Strategy: Last-Writer-Wins by HLC**

```rust
fn resolve_conflict(local: &Change, remote: &Change) -> Resolution {
    match (local, remote) {
        // Same document modified
        (Change::DocumentUpdate { document: l, .. }, Change::DocumentUpdate { document: r, .. })
            if l.id == r.id =>
        {
            if l.hlc > r.hlc {
                Resolution::KeepLocal
            } else {
                Resolution::TakeRemote
            }
        }

        // Document deleted vs updated
        (Change::DocumentDelete { doc_id }, Change::DocumentUpdate { document, .. })
            if *doc_id == document.id =>
        {
            // Delete wins if it's newer
            if get_hlc_for_delete(doc_id) > document.hlc {
                Resolution::KeepLocal
            } else {
                Resolution::TakeRemote
            }
        }

        // No conflict
        _ => Resolution::ApplyBoth,
    }
}
```

**Confidence: MEDIUM** - LWW is simple but may lose data in rare cases; acceptable for this use case.

### 9.5 Sync HTTP API

```yaml
# GET /sync/watermark
# Returns the current HLC watermark for this node
Response:
  hlc: string (hex-encoded HLC)
  node_id: string

# GET /sync/changes?since={hlc}&limit={n}
# Returns changes since the given HLC
Query Parameters:
  since: string (hex-encoded HLC)
  limit: integer (default 1000)
Response:
  changes: Change[]
  watermark: string (current HLC after these changes)
  has_more: boolean

# POST /sync/changes
# Push changes to this node
Request:
  changes: Change[]
  source_watermark: string
Response:
  accepted: boolean
  conflicts: ConflictReport[]
  new_watermark: string
```

**Confidence: HIGH** - Simple REST API; well-understood pattern.

---

## 10. MCP Interface

### 10.1 Tool Definitions

```json
{
  "tools": [
    {
      "name": "rag_search",
      "description": "Search the RAG knowledge base using hybrid semantic and keyword search",
      "inputSchema": {
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
            "description": "Use hybrid search (default: true)",
            "default": true
          }
        },
        "required": ["query"]
      }
    },
    {
      "name": "rag_ingest",
      "description": "Ingest a document into the RAG knowledge base",
      "inputSchema": {
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
      }
    },
    {
      "name": "rag_delete",
      "description": "Delete a document from the RAG knowledge base",
      "inputSchema": {
        "type": "object",
        "properties": {
          "doc_id": {
            "type": "string",
            "description": "The document ID to delete"
          }
        },
        "required": ["doc_id"]
      }
    },
    {
      "name": "rag_list_collections",
      "description": "List all collections in the RAG knowledge base",
      "inputSchema": {
        "type": "object",
        "properties": {}
      }
    },
    {
      "name": "rag_list_documents",
      "description": "List documents in a collection",
      "inputSchema": {
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
      }
    }
  ]
}
```

### 10.2 Resource Definitions

```json
{
  "resources": [
    {
      "uri": "rag://collections",
      "name": "RAG Collections",
      "description": "List of all collections with statistics",
      "mimeType": "application/json"
    },
    {
      "uri": "rag://collections/{name}",
      "name": "Collection Details",
      "description": "Details and document list for a specific collection",
      "mimeType": "application/json"
    },
    {
      "uri": "rag://documents/{id}",
      "name": "Document Content",
      "description": "Full content and metadata for a specific document",
      "mimeType": "application/json"
    }
  ]
}
```

### 10.3 MCP Server Implementation

```rust
// Pseudocode for MCP server

struct RagMcpServer {
    query_engine: Arc<QueryEngine>,
    store: Arc<dyn Store>,
    chunker: Arc<dyn Chunker>,
    embedder: Arc<dyn Embedder>,
}

impl McpServer for RagMcpServer {
    async fn handle_tool_call(&self, name: &str, args: Value) -> Result<Value> {
        match name {
            "rag_search" => {
                let query: String = args["query"].as_str().unwrap().to_string();
                let collection = args.get("collection").and_then(|v| v.as_str());
                let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

                let results = self.query_engine.search(&query, collection).await?;

                // Format results for LLM consumption
                let formatted = results.results.iter()
                    .map(|r| json!({
                        "content": r.chunk.content,
                        "source": r.chunk.doc_id,
                        "score": r.score,
                        "metadata": r.chunk.metadata,
                    }))
                    .collect::<Vec<_>>();

                Ok(json!({ "results": formatted }))
            }

            "rag_ingest" => {
                let uri: String = args["uri"].as_str().unwrap().to_string();
                let collection: String = args["collection"].as_str().unwrap().to_string();
                let content_type = args.get("content_type").and_then(|v| v.as_str());
                let metadata = args.get("metadata").cloned().unwrap_or(json!({}));

                // Load content
                let content = load_uri(&uri).await?;
                let detected_type = content_type
                    .map(|s| s.parse())
                    .transpose()?
                    .unwrap_or_else(|| detect_content_type(&uri, &content));

                // Create document
                let doc_id = Ulid::new();
                let document = Document {
                    id: doc_id,
                    collection: collection.clone(),
                    source_uri: uri,
                    content_hash: blake3::hash(&content).into(),
                    content_type: detected_type,
                    metadata,
                    ..Default::default()
                };

                // Chunk
                let chunks = self.chunker.chunk(&content, detected_type)?;

                // Embed
                let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
                let embeddings = self.embedder.embed(&texts)?;

                // Store
                self.store.insert_document(document).await?;
                self.store.insert_chunks(chunks.into_iter().map(|c| /* ... */).collect()).await?;
                self.store.insert_embeddings(/* ... */).await?;

                Ok(json!({ "doc_id": doc_id.to_string(), "chunks": chunks.len() }))
            }

            // ... other tools

            _ => Err(anyhow!("Unknown tool: {}", name)),
        }
    }

    async fn handle_resource_read(&self, uri: &str) -> Result<(String, String)> {
        // Parse URI and return content
        // ...
    }
}
```

**Confidence: HIGH** - MCP is a well-defined protocol; implementation is straightforward.

---

## 11. Error Handling

### 11.1 Error Taxonomy

```rust
#[derive(Debug, thiserror::Error)]
pub enum RagError {
    // Storage errors
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Document not found: {0}")]
    DocumentNotFound(Ulid),

    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Duplicate document: {uri} already exists in {collection}")]
    DuplicateDocument { uri: String, collection: String },

    // Embedding errors
    #[error("Embedding model error: {0}")]
    EmbeddingModel(#[from] ort::Error),

    #[error("Text too long for embedding: {tokens} tokens exceeds max {max}")]
    TextTooLong { tokens: usize, max: usize },

    // Chunking errors
    #[error("Failed to parse {content_type}: {reason}")]
    ParseError { content_type: String, reason: String },

    #[error("Unsupported content type: {0}")]
    UnsupportedContentType(String),

    // Sync errors
    #[error("Sync failed with peer {peer}: {reason}")]
    SyncFailed { peer: String, reason: String },

    #[error("Conflict resolution failed: {0}")]
    ConflictResolution(String),

    // IO errors
    #[error("Failed to load URI {uri}: {reason}")]
    LoadFailed { uri: String, reason: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    // MCP errors
    #[error("Invalid tool arguments: {0}")]
    InvalidArguments(String),
}
```

### 11.2 Error Recovery Strategies

| Error Type | Recovery Strategy | Confidence |
|------------|-------------------|------------|
| Database corruption | Rebuild from WAL; fallback to backup | MEDIUM |
| Embedding OOM | Reduce batch size; retry | HIGH |
| Parse failure | Skip document; log warning | HIGH |
| Sync network error | Exponential backoff; queue for retry | HIGH |
| Conflict | Apply LWW; log for review | MEDIUM |

**Confidence: MEDIUM** - Error handling is comprehensive but edge cases may emerge.

---

## 12. Performance Considerations

### 12.1 Bottleneck Analysis

| Operation | Bottleneck | Mitigation | Confidence |
|-----------|------------|------------|------------|
| Ingestion | Embedding (GPU/CPU bound) | Batch processing; async pipeline | HIGH |
| Query | Vector search (I/O bound) | Index in RAM; SSD storage | HIGH |
| Query | Keyword search (I/O bound) | FTS5 optimization; limit results | HIGH |
| Sync | Network (bandwidth bound) | Delta sync; compression | MEDIUM |

### 12.2 Optimization Strategies

**Embedding Throughput:**
```
Baseline: 5 chunks/sec (single, no batching)
Batched (32): 50 chunks/sec (10x improvement)
Quantized INT8: +20% throughput, -5% quality
GPU acceleration: 200+ chunks/sec (if available)
```

**Query Latency:**
```
Vector search (100K vectors): ~50ms
FTS5 search (100K docs): ~20ms
RRF fusion: ~1ms
Context fetch: ~30ms
Total: ~100ms (well under 1s target)
```

**Storage I/O:**
```
SQLite WAL mode: Allows concurrent reads during writes
Connection pool (8 readers): Parallel query execution
Batch inserts (100 rows): Reduce transaction overhead
```

**Confidence: MEDIUM** - Numbers are estimates based on similar systems; actual may vary.

### 12.3 Memory Budget

```
Component           | Memory Usage
--------------------|-------------
SQLite connection   | 10 MB
Embedding model     | 500 MB - 1.5 GB
sqlite-vec index    | ~4 bytes/dim × vectors = ~3 GB for 1M chunks
FTS5 index          | ~100 MB for 1M chunks
Working set         | 200 MB
--------------------|-------------
Total (1M chunks)   | ~5 GB
```

**Recommendation:** 8 GB RAM minimum for 100K documents; 16 GB for 1M.

**Confidence: MEDIUM** - Memory estimates are rough; profiling needed.

---

## 13. Security Model

### 13.1 Threat Model

| Threat | Mitigation | Implemented |
|--------|------------|-------------|
| Malicious input (injection) | Parameterized queries; input validation | Yes |
| Path traversal | URI validation; sandboxed file access | Yes |
| DoS via large documents | Size limits; timeout on operations | Yes |
| Data exfiltration via MCP | Local-only by default; auth for network | Partial |
| Model poisoning | Content hash verification; source tracking | Yes |

### 13.2 Input Validation

```rust
fn validate_uri(uri: &str) -> Result<ValidatedUri> {
    let parsed = Url::parse(uri)?;

    match parsed.scheme() {
        "file" => {
            let path = parsed.to_file_path()
                .map_err(|_| RagError::InvalidArguments("Invalid file path".into()))?;

            // Prevent path traversal
            let canonical = path.canonicalize()?;
            if !canonical.starts_with(allowed_base_path()) {
                return Err(RagError::InvalidArguments("Path outside allowed directory".into()));
            }

            Ok(ValidatedUri::File(canonical))
        }
        "https" | "http" => {
            // Validate domain against allowlist if configured
            Ok(ValidatedUri::Http(parsed))
        }
        "data" => {
            // Validate data URI format
            Ok(ValidatedUri::Data(parsed))
        }
        _ => Err(RagError::InvalidArguments(format!("Unsupported scheme: {}", parsed.scheme()))),
    }
}
```

**Confidence: MEDIUM** - Security requires ongoing attention; this covers basics.

---

## 14. Confidence Assessment

### Overall System Confidence

| Component | Confidence | Rationale |
|-----------|------------|-----------|
| Storage (sqlite-vec) | HIGH | Well-documented, tested at scale |
| Embedding (ONNX) | HIGH | Mature runtime, standard integration |
| Chunking (tree-sitter) | HIGH | Battle-tested in many editors |
| Query engine | HIGH | Standard hybrid search techniques |
| Sync protocol | MEDIUM | HLC is proven; implementation complexity |
| MCP interface | HIGH | Well-defined protocol |
| Performance targets | MEDIUM | Based on estimates; needs validation |
| Error handling | MEDIUM | Comprehensive but edge cases possible |

### Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| sqlite-vec performance degrades at scale | Low | High | Benchmark early; have fallback to pgvector |
| ONNX model compatibility issues | Medium | Medium | Pin versions; test on target platforms |
| Sync conflicts cause data loss | Low | Medium | Conflict logging; manual review option |
| MCP protocol changes | Low | Low | Abstract behind interface; version negotiation |
| Memory usage exceeds estimates | Medium | Medium | Configurable limits; streaming where possible |

### Implementation Confidence by Phase

| Phase | Confidence | Notes |
|-------|------------|-------|
| 1. Core + Storage | HIGH | Standard Rust + SQLite |
| 2. Embedding | HIGH | ONNX well-documented |
| 3. Chunking | MEDIUM | tree-sitter integration may have edge cases |
| 4. Query | HIGH | Well-understood algorithms |
| 5. MCP | HIGH | Protocol is straightforward |
| 6. Sync | MEDIUM | Most complex component; may need iteration |

---

## Appendix A: Alternative Approaches Considered

### Vector Store Alternatives

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| sqlite-vec | Embedded, portable, single file | Limited scale, single writer | **Selected** |
| pgvector | Battle-tested, SQL integration | External process, setup complexity | Rejected |
| Qdrant | Purpose-built, high performance | External process, memory hungry | Rejected |
| LanceDB | Embedded, columnar | Newer, less proven | Rejected |
| FAISS | Very fast, GPU support | C++ complexity, no persistence | Rejected |

### Embedding Model Alternatives

| Option | Dims | Context | Quality | Speed | Decision |
|--------|------|---------|---------|-------|----------|
| nomic-embed-text-v1.5 | 768 | 8192 | Good | Fast | **Selected** |
| OpenAI text-embedding-3-small | 1536 | 8192 | Better | API latency | Rejected (not local) |
| Cohere embed-v3 | 1024 | 512 | Better | API latency | Rejected (not local) |
| bge-m3 | 1024 | 8192 | Good | Slower | Considered |
| e5-mistral-7b-instruct | 4096 | 32K | Best | Very slow | Rejected (too slow) |

### Sync Alternatives

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Custom HTTP/HLC | Simple, controllable | More code to write | **Selected** |
| cr-sqlite | Built-in CRDTs | Less control, dependency | Considered |
| Syncthing | No code needed | No fine-grained control | Rejected |
| Git (LFS) | Familiar | Poor for binary, no partial sync | Rejected |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| BM25 | Best Match 25; probabilistic ranking function for keyword search |
| CRDT | Conflict-free Replicated Data Type; data structure for eventual consistency |
| FTS5 | Full-Text Search 5; SQLite extension for text search |
| HLC | Hybrid Logical Clock; timestamp combining physical and logical time |
| MCP | Model Context Protocol; standard for AI tool integration |
| ONNX | Open Neural Network Exchange; portable ML model format |
| RAG | Retrieval-Augmented Generation; technique combining search with LLMs |
| RRF | Reciprocal Rank Fusion; algorithm for combining ranked lists |
| ULID | Universally Unique Lexicographically Sortable Identifier |
| WAL | Write-Ahead Logging; SQLite durability mode |
