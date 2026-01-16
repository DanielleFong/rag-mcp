# rag-store

SQLite-based storage layer with sqlite-vec for vector search and FTS5 for keyword search.

## Overview

| Attribute | Value |
|-----------|-------|
| Purpose | Persistent storage for documents, chunks, and embeddings |
| Dependencies | rusqlite, sqlite-vec, rag-core |
| Est. Lines | ~2,000 |
| Confidence | HIGH |

This crate implements the `Store` trait from rag-core using SQLite with extensions.

---

## Module Structure

```
rag-store/
├── src/
│   ├── lib.rs           # Public exports
│   ├── store.rs         # SqliteStore implementation
│   ├── schema.rs        # Schema creation and migrations
│   ├── queries/
│   │   ├── mod.rs
│   │   ├── collections.rs
│   │   ├── documents.rs
│   │   ├── chunks.rs
│   │   ├── embeddings.rs
│   │   └── search.rs
│   ├── pool.rs          # Connection pooling
│   └── migrations/
│       └── mod.rs       # Migration system
└── Cargo.toml
```

---

## SqliteStore Implementation

### Construction

```rust
use rusqlite::{Connection, OpenFlags};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// SQLite-based implementation of the Store trait.
pub struct SqliteStore {
    /// Write connection (single writer).
    write_conn: Arc<RwLock<Connection>>,

    /// Read connection pool.
    read_pool: Pool<Connection>,

    /// Database path.
    path: PathBuf,

    /// Node ID for HLC.
    node_id: u16,

    /// Current HLC state.
    hlc: Arc<RwLock<HybridLogicalClock>>,
}

impl SqliteStore {
    /// Open or create a database at the given path.
    pub async fn open(path: impl AsRef<Path>, node_id: u16) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Open write connection
        let write_conn = Connection::open_with_flags(
            &path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        // Configure connection
        Self::configure_connection(&write_conn)?;

        // Load sqlite-vec extension
        unsafe {
            write_conn.load_extension_enable()?;
            write_conn.load_extension("vec0", None)?;
            write_conn.load_extension_disable()?;
        }

        // Run migrations
        Self::run_migrations(&write_conn)?;

        // Create read pool
        let read_pool = Pool::builder()
            .max_size(8)
            .build(ConnectionManager::new(&path))?;

        Ok(Self {
            write_conn: Arc::new(RwLock::new(write_conn)),
            read_pool,
            path,
            node_id,
            hlc: Arc::new(RwLock::new(HybridLogicalClock::new(node_id))),
        })
    }

    /// Configure connection pragmas for performance.
    fn configure_connection(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -64000;  -- 64MB cache
            PRAGMA temp_store = MEMORY;
            PRAGMA mmap_size = 268435456;  -- 256MB mmap
            PRAGMA busy_timeout = 5000;
            "
        )?;
        Ok(())
    }

    /// Run pending migrations.
    fn run_migrations(conn: &Connection) -> Result<()> {
        // Migration system implementation
        migrations::run_all(conn)
    }

    /// Get the next HLC timestamp.
    async fn next_hlc(&self) -> HybridLogicalClock {
        let mut hlc = self.hlc.write().await;
        hlc.tick()
    }
}
```

### Collection Operations

```rust
#[async_trait]
impl Store for SqliteStore {
    async fn create_collection(&self, collection: Collection) -> Result<()> {
        let conn = self.write_conn.write().await;
        let hlc = self.next_hlc().await;

        conn.execute(
            "INSERT INTO collections (name, description, settings, created_at, hlc)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                &collection.name,
                &collection.description,
                serde_json::to_string(&collection.settings)?,
                collection.created_at,
                hlc.to_bytes().as_slice(),
            ],
        ).map_err(|e| {
            if e.to_string().contains("UNIQUE constraint") {
                RagError::CollectionExists(collection.name.clone())
            } else {
                RagError::Database(e)
            }
        })?;

        Ok(())
    }

    async fn get_collection(&self, name: &str) -> Result<Option<Collection>> {
        let conn = self.read_pool.get().await?;

        let result = conn.query_row(
            "SELECT name, description, settings, created_at, hlc
             FROM collections WHERE name = ?1",
            params![name],
            |row| {
                Ok(Collection {
                    name: row.get(0)?,
                    description: row.get(1)?,
                    settings: serde_json::from_str(&row.get::<_, String>(2)?).unwrap_or_default(),
                    created_at: row.get(3)?,
                    hlc: HybridLogicalClock::from_bytes(&row.get::<_, Vec<u8>>(4)?.try_into().unwrap()),
                })
            },
        );

        match result {
            Ok(collection) => Ok(Some(collection)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(RagError::Database(e)),
        }
    }

    async fn list_collections(&self) -> Result<Vec<Collection>> {
        let conn = self.read_pool.get().await?;

        let mut stmt = conn.prepare(
            "SELECT name, description, settings, created_at, hlc
             FROM collections ORDER BY name"
        )?;

        let collections = stmt.query_map([], |row| {
            Ok(Collection {
                name: row.get(0)?,
                description: row.get(1)?,
                settings: serde_json::from_str(&row.get::<_, String>(2)?).unwrap_or_default(),
                created_at: row.get(3)?,
                hlc: HybridLogicalClock::from_bytes(&row.get::<_, Vec<u8>>(4)?.try_into().unwrap()),
            })
        })?.collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(collections)
    }

    async fn delete_collection(&self, name: &str) -> Result<()> {
        let conn = self.write_conn.write().await;

        // CASCADE will delete documents and chunks
        let deleted = conn.execute(
            "DELETE FROM collections WHERE name = ?1",
            params![name],
        )?;

        if deleted == 0 {
            return Err(RagError::CollectionNotFound(name.to_string()));
        }

        Ok(())
    }
}
```

### Document Operations

```rust
impl SqliteStore {
    async fn insert_document(&self, document: Document) -> Result<Ulid> {
        let conn = self.write_conn.write().await;
        let hlc = self.next_hlc().await;

        conn.execute(
            "INSERT INTO documents
             (id, collection, source_uri, content_hash, content_type, raw_content,
              metadata, created_at, updated_at, hlc)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                document.id.to_string(),
                &document.collection,
                &document.source_uri,
                document.content_hash.as_slice(),
                format!("{:?}", document.content_type),
                &document.raw_content,
                serde_json::to_string(&document.metadata)?,
                document.created_at,
                document.updated_at,
                hlc.to_bytes().as_slice(),
            ],
        )?;

        Ok(document.id)
    }

    async fn get_document(&self, id: Ulid) -> Result<Option<Document>> {
        let conn = self.read_pool.get().await?;

        let result = conn.query_row(
            "SELECT id, collection, source_uri, content_hash, content_type,
                    raw_content, metadata, created_at, updated_at, hlc
             FROM documents WHERE id = ?1",
            params![id.to_string()],
            Self::row_to_document,
        );

        match result {
            Ok(doc) => Ok(Some(doc)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(RagError::Database(e)),
        }
    }

    async fn get_document_by_uri(&self, collection: &str, uri: &str) -> Result<Option<Document>> {
        let conn = self.read_pool.get().await?;

        let result = conn.query_row(
            "SELECT id, collection, source_uri, content_hash, content_type,
                    raw_content, metadata, created_at, updated_at, hlc
             FROM documents WHERE collection = ?1 AND source_uri = ?2",
            params![collection, uri],
            Self::row_to_document,
        );

        match result {
            Ok(doc) => Ok(Some(doc)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(RagError::Database(e)),
        }
    }

    async fn list_documents(
        &self,
        collection: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<Document>> {
        let conn = self.read_pool.get().await?;

        let mut stmt = conn.prepare(
            "SELECT id, collection, source_uri, content_hash, content_type,
                    raw_content, metadata, created_at, updated_at, hlc
             FROM documents
             WHERE collection = ?1
             ORDER BY created_at DESC
             LIMIT ?2 OFFSET ?3"
        )?;

        let documents = stmt.query_map(
            params![collection, limit, offset],
            Self::row_to_document,
        )?.collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(documents)
    }

    async fn delete_document(&self, id: Ulid) -> Result<()> {
        let conn = self.write_conn.write().await;

        // Delete embeddings first (no FK to vec_chunks)
        let chunk_ids: Vec<String> = conn.prepare(
            "SELECT id FROM chunks WHERE doc_id = ?1"
        )?.query_map(params![id.to_string()], |row| row.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        for chunk_id in &chunk_ids {
            conn.execute(
                "DELETE FROM vec_chunks WHERE chunk_id = ?1",
                params![chunk_id],
            )?;
        }

        // Delete document (CASCADE handles chunks via FK)
        let deleted = conn.execute(
            "DELETE FROM documents WHERE id = ?1",
            params![id.to_string()],
        )?;

        if deleted == 0 {
            return Err(RagError::DocumentNotFound(id));
        }

        Ok(())
    }

    fn row_to_document(row: &rusqlite::Row) -> rusqlite::Result<Document> {
        let id_str: String = row.get(0)?;
        let content_type_str: String = row.get(4)?;
        let hlc_bytes: Vec<u8> = row.get(9)?;

        Ok(Document {
            id: Ulid::from_string(&id_str).unwrap(),
            collection: row.get(1)?,
            source_uri: row.get(2)?,
            content_hash: row.get::<_, Vec<u8>>(3)?.try_into().unwrap(),
            content_type: content_type_str.parse().unwrap_or(ContentType::Unknown),
            raw_content: row.get(5)?,
            metadata: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
            created_at: row.get(7)?,
            updated_at: row.get(8)?,
            hlc: HybridLogicalClock::from_bytes(&hlc_bytes.try_into().unwrap()),
        })
    }
}
```

### Chunk Operations

```rust
impl SqliteStore {
    async fn insert_chunks(&self, chunks: Vec<Chunk>) -> Result<()> {
        let conn = self.write_conn.write().await;

        let tx = conn.transaction()?;

        {
            let mut stmt = tx.prepare(
                "INSERT INTO chunks
                 (id, doc_id, chunk_index, content, content_hash, token_count,
                  char_offset_start, char_offset_end, metadata, hlc)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)"
            )?;

            for chunk in &chunks {
                let hlc = self.next_hlc().await;
                stmt.execute(params![
                    chunk.id.to_string(),
                    chunk.doc_id.to_string(),
                    chunk.chunk_index,
                    &chunk.content,
                    chunk.content_hash.as_slice(),
                    chunk.token_count,
                    chunk.char_offset_start,
                    chunk.char_offset_end,
                    serde_json::to_string(&chunk.metadata)?,
                    hlc.to_bytes().as_slice(),
                ])?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    async fn get_chunks(&self, doc_id: Ulid) -> Result<Vec<Chunk>> {
        let conn = self.read_pool.get().await?;

        let mut stmt = conn.prepare(
            "SELECT id, doc_id, chunk_index, content, content_hash, token_count,
                    char_offset_start, char_offset_end, metadata, hlc
             FROM chunks
             WHERE doc_id = ?1
             ORDER BY chunk_index"
        )?;

        let chunks = stmt.query_map(
            params![doc_id.to_string()],
            Self::row_to_chunk,
        )?.collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(chunks)
    }

    async fn get_chunk(&self, id: Ulid) -> Result<Option<Chunk>> {
        let conn = self.read_pool.get().await?;

        let result = conn.query_row(
            "SELECT id, doc_id, chunk_index, content, content_hash, token_count,
                    char_offset_start, char_offset_end, metadata, hlc
             FROM chunks WHERE id = ?1",
            params![id.to_string()],
            Self::row_to_chunk,
        );

        match result {
            Ok(chunk) => Ok(Some(chunk)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(RagError::Database(e)),
        }
    }

    async fn get_chunk_by_index(&self, doc_id: Ulid, index: u32) -> Result<Option<Chunk>> {
        let conn = self.read_pool.get().await?;

        let result = conn.query_row(
            "SELECT id, doc_id, chunk_index, content, content_hash, token_count,
                    char_offset_start, char_offset_end, metadata, hlc
             FROM chunks WHERE doc_id = ?1 AND chunk_index = ?2",
            params![doc_id.to_string(), index],
            Self::row_to_chunk,
        );

        match result {
            Ok(chunk) => Ok(Some(chunk)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(RagError::Database(e)),
        }
    }

    fn row_to_chunk(row: &rusqlite::Row) -> rusqlite::Result<Chunk> {
        let id_str: String = row.get(0)?;
        let doc_id_str: String = row.get(1)?;
        let hlc_bytes: Vec<u8> = row.get(9)?;

        Ok(Chunk {
            id: Ulid::from_string(&id_str).unwrap(),
            doc_id: Ulid::from_string(&doc_id_str).unwrap(),
            chunk_index: row.get(2)?,
            content: row.get(3)?,
            content_hash: row.get::<_, Vec<u8>>(4)?.try_into().unwrap(),
            token_count: row.get(5)?,
            char_offset_start: row.get(6)?,
            char_offset_end: row.get(7)?,
            metadata: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
            hlc: HybridLogicalClock::from_bytes(&hlc_bytes.try_into().unwrap()),
        })
    }
}
```

### Embedding Operations

```rust
impl SqliteStore {
    async fn insert_embeddings(&self, embeddings: Vec<(Ulid, Vec<f32>)>) -> Result<()> {
        let conn = self.write_conn.write().await;

        let tx = conn.transaction()?;

        {
            let mut stmt = tx.prepare(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)"
            )?;

            for (chunk_id, embedding) in &embeddings {
                // Convert f32 vector to bytes for sqlite-vec
                let embedding_bytes: Vec<u8> = embedding.iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();

                stmt.execute(params![
                    chunk_id.to_string(),
                    embedding_bytes,
                ])?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    async fn delete_embeddings(&self, chunk_ids: &[Ulid]) -> Result<()> {
        let conn = self.write_conn.write().await;

        let tx = conn.transaction()?;

        {
            let mut stmt = tx.prepare("DELETE FROM vec_chunks WHERE chunk_id = ?1")?;
            for id in chunk_ids {
                stmt.execute(params![id.to_string()])?;
            }
        }

        tx.commit()?;
        Ok(())
    }
}
```

### Search Operations

```rust
impl SqliteStore {
    async fn vector_search(
        &self,
        query: &[f32],
        k: usize,
        collection: Option<&str>,
    ) -> Result<Vec<(Ulid, f32)>> {
        let conn = self.read_pool.get().await?;

        // Convert query to bytes
        let query_bytes: Vec<u8> = query.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let sql = if let Some(coll) = collection {
            // Filter by collection via JOIN
            "SELECT vc.chunk_id, vc.distance
             FROM vec_chunks vc
             JOIN chunks c ON vc.chunk_id = c.id
             JOIN documents d ON c.doc_id = d.id
             WHERE d.collection = ?2
             AND vc.embedding MATCH ?1
             ORDER BY vc.distance
             LIMIT ?3"
        } else {
            "SELECT chunk_id, distance
             FROM vec_chunks
             WHERE embedding MATCH ?1
             ORDER BY distance
             LIMIT ?2"
        };

        let results = if let Some(coll) = collection {
            let mut stmt = conn.prepare(sql)?;
            stmt.query_map(params![query_bytes, coll, k as i64], |row| {
                let id_str: String = row.get(0)?;
                let distance: f32 = row.get(1)?;
                Ok((Ulid::from_string(&id_str).unwrap(), distance))
            })?.collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            let mut stmt = conn.prepare(sql)?;
            stmt.query_map(params![query_bytes, k as i64], |row| {
                let id_str: String = row.get(0)?;
                let distance: f32 = row.get(1)?;
                Ok((Ulid::from_string(&id_str).unwrap(), distance))
            })?.collect::<std::result::Result<Vec<_>, _>>()?
        };

        Ok(results)
    }

    async fn keyword_search(
        &self,
        query: &str,
        k: usize,
        collection: Option<&str>,
    ) -> Result<Vec<(Ulid, f32)>> {
        let conn = self.read_pool.get().await?;

        let sql = if let Some(coll) = collection {
            "SELECT c.id, bm25(chunks_fts) as score
             FROM chunks_fts
             JOIN chunks c ON chunks_fts.rowid = c.rowid
             JOIN documents d ON c.doc_id = d.id
             WHERE d.collection = ?2
             AND chunks_fts MATCH ?1
             ORDER BY score
             LIMIT ?3"
        } else {
            "SELECT c.id, bm25(chunks_fts) as score
             FROM chunks_fts
             JOIN chunks c ON chunks_fts.rowid = c.rowid
             WHERE chunks_fts MATCH ?1
             ORDER BY score
             LIMIT ?2"
        };

        // Convert query to FTS5 format (escape special chars)
        let fts_query = Self::escape_fts_query(query);

        let results = if let Some(coll) = collection {
            let mut stmt = conn.prepare(sql)?;
            stmt.query_map(params![fts_query, coll, k as i64], |row| {
                let id_str: String = row.get(0)?;
                let score: f32 = row.get(1)?;
                Ok((Ulid::from_string(&id_str).unwrap(), score))
            })?.collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            let mut stmt = conn.prepare(sql)?;
            stmt.query_map(params![fts_query, k as i64], |row| {
                let id_str: String = row.get(0)?;
                let score: f32 = row.get(1)?;
                Ok((Ulid::from_string(&id_str).unwrap(), score))
            })?.collect::<std::result::Result<Vec<_>, _>>()?
        };

        Ok(results)
    }

    /// Escape special FTS5 characters.
    fn escape_fts_query(query: &str) -> String {
        // Simple tokenization - split on whitespace, wrap each in quotes
        query.split_whitespace()
            .map(|word| format!("\"{}\"", word.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" ")
    }
}
```

---

## Schema

```rust
// schema.rs

pub const SCHEMA_SQL: &str = r#"
-- Collections
CREATE TABLE IF NOT EXISTS collections (
    name TEXT PRIMARY KEY,
    description TEXT,
    settings TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    hlc BLOB NOT NULL
);

-- Documents
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    source_uri TEXT NOT NULL,
    content_hash BLOB NOT NULL,
    content_type TEXT NOT NULL,
    raw_content TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    hlc BLOB NOT NULL,
    UNIQUE(collection, source_uri)
);

-- Chunks
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash BLOB NOT NULL,
    token_count INTEGER NOT NULL,
    char_offset_start INTEGER NOT NULL,
    char_offset_end INTEGER NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    hlc BLOB NOT NULL,
    UNIQUE(doc_id, chunk_index)
);

-- Vector embeddings (sqlite-vec)
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding float[768] distance_metric=cosine
);

-- Full-text search (FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- FTS5 sync triggers
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

-- Indices
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
CREATE INDEX IF NOT EXISTS idx_documents_source_uri ON documents(source_uri);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_hlc ON documents(hlc);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_hlc ON chunks(hlc);

-- Sync tracking
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
    direction TEXT NOT NULL,
    changes_count INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    status TEXT NOT NULL,
    error_message TEXT
);
"#;
```

---

## Connection Pooling

```rust
// pool.rs

use deadpool::managed::{Manager, Pool, RecycleResult};
use rusqlite::Connection;

pub struct ConnectionManager {
    path: PathBuf,
}

impl ConnectionManager {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

impl Manager for ConnectionManager {
    type Type = Connection;
    type Error = rusqlite::Error;

    async fn create(&self) -> Result<Connection, Self::Error> {
        let conn = Connection::open_with_flags(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        // Configure for read-only operations
        conn.execute_batch(
            "
            PRAGMA query_only = ON;
            PRAGMA cache_size = -16000;
            PRAGMA mmap_size = 268435456;
            "
        )?;

        // Load sqlite-vec for vector search
        unsafe {
            conn.load_extension_enable()?;
            conn.load_extension("vec0", None)?;
            conn.load_extension_disable()?;
        }

        Ok(conn)
    }

    async fn recycle(&self, conn: &mut Connection) -> RecycleResult<Self::Error> {
        // Verify connection is still valid
        conn.execute_batch("SELECT 1")?;
        Ok(())
    }
}
```

---

## Performance Considerations

### Write Batching

```rust
impl SqliteStore {
    /// Batch insert for improved throughput.
    pub async fn batch_insert(
        &self,
        documents: Vec<Document>,
        chunks: Vec<Vec<Chunk>>,
        embeddings: Vec<Vec<(Ulid, Vec<f32>)>>,
    ) -> Result<()> {
        let conn = self.write_conn.write().await;
        let tx = conn.transaction()?;

        // Insert all documents
        {
            let mut stmt = tx.prepare(/* ... */)?;
            for doc in &documents {
                stmt.execute(/* ... */)?;
            }
        }

        // Insert all chunks
        {
            let mut stmt = tx.prepare(/* ... */)?;
            for doc_chunks in &chunks {
                for chunk in doc_chunks {
                    stmt.execute(/* ... */)?;
                }
            }
        }

        // Insert all embeddings
        {
            let mut stmt = tx.prepare(/* ... */)?;
            for doc_embeddings in &embeddings {
                for (id, vec) in doc_embeddings {
                    stmt.execute(/* ... */)?;
                }
            }
        }

        tx.commit()?;
        Ok(())
    }
}
```

### Optimization

```rust
impl SqliteStore {
    /// Optimize database indices.
    pub async fn optimize(&self) -> Result<()> {
        let conn = self.write_conn.write().await;

        // Optimize FTS5 index
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')", [])?;

        // Analyze for query planner
        conn.execute("ANALYZE", [])?;

        // Vacuum to reclaim space
        conn.execute("VACUUM", [])?;

        Ok(())
    }

    /// Get storage statistics.
    pub async fn stats(&self) -> Result<StoreStats> {
        let conn = self.read_pool.get().await?;

        let collections: u64 = conn.query_row(
            "SELECT COUNT(*) FROM collections", [], |r| r.get(0)
        )?;

        let documents: u64 = conn.query_row(
            "SELECT COUNT(*) FROM documents", [], |r| r.get(0)
        )?;

        let chunks: u64 = conn.query_row(
            "SELECT COUNT(*) FROM chunks", [], |r| r.get(0)
        )?;

        let embeddings: u64 = conn.query_row(
            "SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0)
        )?;

        // Get database file size
        let storage_bytes = std::fs::metadata(&self.path)?.len();

        Ok(StoreStats {
            collections,
            documents,
            chunks,
            embeddings,
            storage_bytes,
            index_bytes: 0, // Would need PRAGMA to get this
        })
    }
}
```

---

## Confidence Assessment

| Component | Confidence | Notes |
|-----------|------------|-------|
| SQLite integration | HIGH | rusqlite is mature |
| sqlite-vec integration | HIGH | Well-documented extension |
| FTS5 integration | HIGH | Standard SQLite feature |
| Connection pooling | HIGH | Standard pattern |
| Transaction handling | HIGH | Standard SQLite patterns |
| Vector search SQL | MEDIUM | sqlite-vec syntax may vary |
| Performance tuning | MEDIUM | May need benchmarking |

---

## Cargo.toml

```toml
[package]
name = "rag-store"
version = "0.1.0"
edition = "2021"
description = "SQLite-based storage for the RAG system"
license = "MIT"

[dependencies]
rag-core = { path = "../rag-core" }
rusqlite = { version = "0.31", features = ["bundled", "vtab", "load_extension"] }
deadpool = { version = "0.10", features = ["managed"] }
tokio = { version = "1.0", features = ["sync"] }
serde_json = "1.0"
async-trait = "0.1"
tracing = "0.1"

[dev-dependencies]
tokio = { version = "1.0", features = ["macros", "rt-multi-thread"] }
tempfile = "3.10"
```
