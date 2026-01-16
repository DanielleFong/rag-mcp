# IMPLEMENTATION_NOTES.md

Implementation guidance, gotchas, and decisions rationale.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Crate Build Order](#crate-build-order)
3. [Critical Implementation Details](#critical-implementation-details)
4. [Common Pitfalls](#common-pitfalls)
5. [Testing Strategies](#testing-strategies)
6. [Performance Optimization](#performance-optimization)
7. [Decision Rationale](#decision-rationale)

---

## Development Setup

### Prerequisites

```bash
# Rust toolchain
rustup default stable
rustup component add clippy rustfmt

# System dependencies (macOS)
brew install sqlite sqlite-vec

# System dependencies (Ubuntu)
sudo apt install libsqlite3-dev
# Build sqlite-vec from source

# ONNX Runtime
# Download from https://github.com/microsoft/onnxruntime/releases
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

### Workspace Setup

```bash
# Clone and setup
git clone <repo>
cd rag-mcp

# Download embedding model
./scripts/download-model.sh

# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Run clippy
cargo clippy --workspace -- -D warnings
```

### Environment Variables

```bash
# Required
export RAG_DATABASE_PATH=~/.local/share/rag-mcp/rag.db
export RAG_MODEL_PATH=~/.local/share/rag-mcp/models/nomic-embed-text-v1.5

# Optional
export RUST_LOG=info  # debug, trace for more
export RAG_NODE_ID=1234  # For sync
```

---

## Crate Build Order

Due to dependencies, build in this order:

```
1. rag-core       (no internal deps)
2. rag-chunk      (depends on rag-core)
3. rag-embed      (depends on rag-core)
4. rag-store      (depends on rag-core)
5. rag-query      (depends on rag-core, rag-store, rag-embed)
6. rag-sync       (depends on rag-core, rag-store)
7. rag-mcp        (depends on rag-core, rag-query)
8. rag-cli        (depends on all)
9. rag-tui        (depends on all)
```

### Minimal Dependency Graph

```
rag-core
├── rag-chunk
├── rag-embed
├── rag-store
│   └── rag-sync
└── rag-query
    └── rag-mcp
        └── rag-cli
            └── rag-tui
```

---

## Critical Implementation Details

### 1. sqlite-vec Integration

**Loading the Extension:**

```rust
// MUST be loaded before creating the virtual table
unsafe {
    conn.load_extension_enable()?;

    // Try different paths based on platform
    let result = conn.load_extension("vec0", None)
        .or_else(|_| conn.load_extension("libsqlite_vec", None))
        .or_else(|_| conn.load_extension("/usr/local/lib/libsqlite_vec", None));

    conn.load_extension_disable()?;
    result?;
}
```

**Vector Format:**

```rust
// sqlite-vec expects little-endian f32 bytes
fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
```

**Query Syntax:**

```sql
-- MATCH clause for KNN search
SELECT chunk_id, distance
FROM vec_chunks
WHERE embedding MATCH ?  -- ? is the query vector as bytes
ORDER BY distance
LIMIT ?;
```

### 2. Nomic Embedding Prefixes

**CRITICAL:** Nomic-embed requires different prefixes for documents vs queries.

```rust
// For documents being indexed
fn embed_document(text: &str) -> Vec<f32> {
    let prefixed = format!("search_document: {}", text);
    embed_internal(&prefixed)
}

// For search queries
fn embed_query(text: &str) -> Vec<f32> {
    let prefixed = format!("search_query: {}", text);
    embed_internal(&prefixed)
}
```

**Why it matters:**
- Without prefixes, retrieval quality drops significantly
- Documents describe content; queries ask questions
- The model was trained with asymmetric prefixes

### 3. HLC Byte Ordering

**Format:** 14 bytes total
- Bytes 0-7: wall_time (big-endian u64)
- Bytes 8-11: logical (big-endian u32)
- Bytes 12-13: node_id (big-endian u16)

```rust
fn to_bytes(&self) -> [u8; 14] {
    let mut buf = [0u8; 14];
    // BIG-endian for lexicographic ordering
    buf[0..8].copy_from_slice(&self.wall_time.to_be_bytes());
    buf[8..12].copy_from_slice(&self.logical.to_be_bytes());
    buf[12..14].copy_from_slice(&self.node_id.to_be_bytes());
    buf
}
```

**Why big-endian:**
- Allows lexicographic comparison in SQLite
- `ORDER BY hlc` works correctly as blob comparison

### 4. FTS5 Trigger Synchronization

**The triggers MUST stay in sync:**

```sql
-- These three triggers keep FTS5 synchronized with chunks table
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
END;

CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', OLD.rowid, OLD.content);
END;

CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', OLD.rowid, OLD.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
END;
```

**Common bug:** Forgetting the delete trigger causes stale FTS entries.

### 5. Transaction Boundaries

**Rule:** Batch operations MUST be transactional.

```rust
// CORRECT: Single transaction for all inserts
async fn insert_document_with_chunks(&self, doc: Document, chunks: Vec<Chunk>) -> Result<()> {
    let conn = self.write_conn.write().await;
    let tx = conn.transaction()?;

    // Insert document
    tx.execute("INSERT INTO documents ...", params![...])?;

    // Insert all chunks
    for chunk in &chunks {
        tx.execute("INSERT INTO chunks ...", params![...])?;
    }

    // Insert all embeddings
    for (id, vec) in &embeddings {
        tx.execute("INSERT INTO vec_chunks ...", params![...])?;
    }

    tx.commit()?;  // Atomic commit
    Ok(())
}

// WRONG: Separate transactions (can leave inconsistent state)
async fn insert_document_bad(&self, doc: Document, chunks: Vec<Chunk>) -> Result<()> {
    self.store.insert_document(doc).await?;  // Transaction 1
    self.store.insert_chunks(chunks).await?;  // Transaction 2 - what if this fails?
    self.store.insert_embeddings(emb).await?; // Transaction 3
}
```

---

## Common Pitfalls

### 1. ULID String vs Bytes

**ULID can be stored as string or bytes:**

```rust
// We use STRING for readability in SQLite
// Pro: Easy to query manually
// Con: 26 bytes vs 16 bytes

// Store as string
let id_str = ulid.to_string();  // "01HRE4KXQN..."
conn.execute("INSERT INTO docs (id) VALUES (?)", params![id_str])?;

// Parse back
let id = Ulid::from_string(&id_str)?;
```

**Consistency:** Pick one format and stick with it. This design uses strings everywhere.

### 2. Async in Traits

**async-trait macro is required:**

```rust
use async_trait::async_trait;

#[async_trait]
pub trait Store: Send + Sync {
    async fn get_document(&self, id: Ulid) -> Result<Option<Document>>;
}

#[async_trait]
impl Store for SqliteStore {
    async fn get_document(&self, id: Ulid) -> Result<Option<Document>> {
        // Implementation
    }
}
```

**Why:** Rust doesn't have native async traits yet (as of 1.75).

### 3. Connection Pool Deadlocks

**Avoid holding connection across await points:**

```rust
// WRONG: Holds connection across await
async fn bad_pattern(&self) -> Result<()> {
    let conn = self.read_pool.get().await?;
    let data = conn.query(...)?;

    // This await while holding conn can cause deadlocks
    let processed = some_async_operation(data).await?;

    Ok(())
}

// CORRECT: Release connection before await
async fn good_pattern(&self) -> Result<()> {
    let data = {
        let conn = self.read_pool.get().await?;
        conn.query(...)?
        // conn dropped here
    };

    let processed = some_async_operation(data).await?;
    Ok(())
}
```

### 4. Tree-sitter Parser Reuse

**Parser is NOT thread-safe:**

```rust
// WRONG: Shared parser
struct BadChunker {
    parser: Parser,  // Will panic if used from multiple threads
}

// CORRECT: Parser per thread or mutex
struct GoodChunker {
    parser: Mutex<Parser>,
}

// Or create per-call (slight overhead)
fn chunk(&self, content: &str) -> Result<Vec<Chunk>> {
    let mut parser = Parser::new();
    parser.set_language(self.language)?;
    // ...
}
```

### 5. Embedding Batch Size vs Memory

**Trade-off:**

| Batch Size | Memory | Throughput | Latency |
|------------|--------|------------|---------|
| 1 | Low | Low | High |
| 32 | Medium | Good | Medium |
| 128 | High | Best | Low |

**Recommendation:** Start with 32, adjust based on available memory.

---

## Testing Strategies

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Test pure functions
    #[test]
    fn test_hlc_ordering() {
        let a = HybridLogicalClock::new(1);
        let b = HybridLogicalClock::new(1);
        assert!(a < b.tick());
    }

    // Test with mocks
    #[tokio::test]
    async fn test_search_with_mock_store() {
        let store = MockStore::new();
        store.expect_vector_search().returning(|_, _, _| {
            Ok(vec![(Ulid::new(), 0.5)])
        });

        let engine = QueryEngine::new(Arc::new(store), embedder, config);
        let results = engine.search("test", None).await.unwrap();
        assert_eq!(results.results.len(), 1);
    }
}
```

### Integration Tests

```rust
// tests/integration_test.rs

#[tokio::test]
async fn test_full_pipeline() {
    // Setup
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");

    let store = SqliteStore::open(&db_path, 1).await.unwrap();
    let embedder = OnnxEmbedder::new("./models/nomic-embed-text-v1.5").unwrap();
    let chunker = AdaptiveChunker::new("./models/nomic-embed-text-v1.5/tokenizer.json").unwrap();

    // Create collection
    store.create_collection(Collection {
        name: "test".to_string(),
        ..Default::default()
    }).await.unwrap();

    // Ingest document
    let doc = Document::new("test", "file://test.rs", "fn main() {}", ContentType::Rust);
    let chunks = chunker.chunk(&doc.raw_content.unwrap(), doc.content_type, &Default::default()).unwrap();
    // ... continue with embeddings and storage

    // Search
    let engine = QueryEngine::new(Arc::new(store), Arc::new(embedder), Default::default());
    let results = engine.search("main function", Some("test")).await.unwrap();

    // Verify
    assert!(!results.results.is_empty());
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn hlc_always_increases(ops in prop::collection::vec(0..100u32, 1..100)) {
        let mut hlc = HybridLogicalClock::new(1);
        let mut prev = hlc.to_bytes();

        for _ in ops {
            hlc.tick();
            let current = hlc.to_bytes();
            assert!(current > prev);
            prev = current;
        }
    }
}
```

---

## Performance Optimization

### 1. Batch Everything

```rust
// Instead of:
for doc in documents {
    store.insert_document(doc).await?;
}

// Do:
store.batch_insert_documents(documents).await?;
```

### 2. Parallel Where Possible

```rust
// Search both indices in parallel
let (vector_results, keyword_results) = tokio::join!(
    store.vector_search(&embedding, k, collection),
    store.keyword_search(&query, k, collection),
);
```

### 3. Lazy Loading

```rust
// Don't load content until needed
struct LazyChunk {
    id: Ulid,
    content: OnceCell<String>,
}

impl LazyChunk {
    async fn content(&self, store: &Store) -> &str {
        self.content.get_or_init(|| async {
            store.get_chunk_content(self.id).await.unwrap()
        }).await
    }
}
```

### 4. Connection Pooling Tuning

```rust
// Read-heavy workload: more readers
let pool = Pool::builder()
    .max_size(16)  // Many concurrent reads
    .build(manager)?;

// Write-heavy workload: fewer readers (WAL contention)
let pool = Pool::builder()
    .max_size(4)
    .build(manager)?;
```

### 5. SQLite Pragmas

```sql
-- For SSD storage
PRAGMA page_size = 4096;

-- For bulk inserts
PRAGMA synchronous = OFF;  -- DANGER: data loss on crash
PRAGMA journal_mode = MEMORY;

-- For read-heavy
PRAGMA cache_size = -64000;  -- 64MB
PRAGMA mmap_size = 268435456;  -- 256MB
```

---

## Decision Rationale

### Why sqlite-vec over pgvector?

| Factor | sqlite-vec | pgvector |
|--------|-----------|----------|
| Deployment | Single binary | Requires PostgreSQL |
| Performance | Good to 1M vectors | Excellent |
| Portability | File-based | Client-server |
| Features | Basic KNN | Indexes, filtering |

**Decision:** sqlite-vec for simplicity. Can migrate to pgvector if scale requires.

### Why nomic-embed over OpenAI?

| Factor | nomic-embed | OpenAI |
|--------|-------------|--------|
| Cost | Free | $0.0001/1K tokens |
| Privacy | Local | Data sent to API |
| Latency | ~20ms | ~200ms |
| Quality | Good | Better |
| Context | 8192 tokens | 8192 tokens |

**Decision:** Local-first is a core requirement. Nomic quality is sufficient.

### Why HLC over Lamport Clocks?

| Factor | HLC | Lamport |
|--------|-----|---------|
| Wall-clock bound | Yes | No |
| Debugging | Easier (readable time) | Harder |
| Complexity | Medium | Low |
| Drift handling | Built-in | N/A |

**Decision:** HLC provides better debugging and bounded drift.

### Why RRF over Learned Fusion?

| Factor | RRF | Learned |
|--------|-----|---------|
| Training | None | Required |
| Performance | Good | Potentially better |
| Interpretability | High | Low |
| Maintenance | None | Ongoing |

**Decision:** RRF is simple, robust, and works well without training data.

### Why tree-sitter over Regex?

| Factor | tree-sitter | Regex |
|--------|-------------|-------|
| Accuracy | Excellent | Poor for nested |
| Languages | Many supported | Custom per language |
| Performance | Fast | Varies |
| Maintenance | Grammar updates | Pattern updates |

**Decision:** Tree-sitter provides semantic chunking that regex cannot achieve.

---

## Code Style Guidelines

### Error Handling

```rust
// Use ? for propagation
fn example() -> Result<()> {
    let data = load_data()?;
    process(data)?;
    Ok(())
}

// Add context for debugging
fn example_with_context() -> Result<()> {
    let data = load_data()
        .map_err(|e| RagError::LoadFailed {
            uri: "file://data.txt".to_string(),
            reason: e.to_string(),
        })?;
    Ok(())
}
```

### Logging

```rust
use tracing::{info, warn, error, debug, trace};

// Info: User-visible events
info!("Ingested {} documents", count);

// Warn: Recoverable issues
warn!("Peer {} unreachable, will retry", peer_id);

// Error: Failures
error!("Database corruption detected: {}", err);

// Debug: Developer info
debug!("Query embedding: {:?}", &embedding[..5]);

// Trace: Verbose
trace!("Processing chunk {} of {}", i, total);
```

### Documentation

```rust
/// Search the knowledge base.
///
/// # Arguments
///
/// * `query` - The search query string
/// * `collection` - Optional collection to filter results
///
/// # Returns
///
/// Search results ordered by relevance score.
///
/// # Errors
///
/// Returns `RagError::Database` if the database query fails.
///
/// # Example
///
/// ```
/// let results = engine.search("error handling", None).await?;
/// for r in results.results {
///     println!("{}: {}", r.score, r.chunk.content);
/// }
/// ```
pub async fn search(&self, query: &str, collection: Option<&str>) -> Result<SearchResults> {
    // ...
}
```
