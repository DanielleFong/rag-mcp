//! SQLite-based storage implementation.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use tracing::{debug, info, warn};
use ulid::Ulid;

use rag_core::{
    Collection, Chunk, ContentType, Document, HybridLogicalClock, RagError, Result,
    Stats, Store, SyncChange,
};

use crate::schema::{SCHEMA, VEC_SCHEMA};

/// SQLite-based store implementation.
///
/// Uses a blocking Mutex for thread-safe access and runs SQLite operations
/// on the blocking thread pool via `spawn_blocking`.
pub struct SqliteStore {
    /// Connection wrapped in blocking Mutex.
    conn: Arc<Mutex<Connection>>,

    /// Node ID for HLC.
    node_id: u16,

    /// Current HLC state.
    hlc: Arc<Mutex<HybridLogicalClock>>,

    /// Whether sqlite-vec extension is loaded.
    vec_enabled: bool,
}

// Manually implement Send + Sync since Connection is protected by Mutex
unsafe impl Send for SqliteStore {}
unsafe impl Sync for SqliteStore {}

impl SqliteStore {
    /// Open or create a database at the given path.
    pub fn open(path: impl AsRef<Path>, node_id: u16) -> Result<Self> {
        let path = path.as_ref();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Open connection with appropriate flags
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|e| RagError::database(format!("Failed to open database: {}", e)))?;

        Self::init(conn, node_id, path)
    }

    /// Open an in-memory database (for testing).
    pub fn open_memory(node_id: u16) -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| RagError::database(format!("Failed to open in-memory database: {}", e)))?;

        Self::init(conn, node_id, Path::new(":memory:"))
    }

    /// Initialize the store with a connection.
    fn init(conn: Connection, node_id: u16, path: &Path) -> Result<Self> {
        // Configure SQLite for performance
        Self::configure_connection(&conn)?;

        // Initialize schema
        conn.execute_batch(SCHEMA)
            .map_err(|e| RagError::database(format!("Failed to initialize schema: {}", e)))?;

        // Try to load sqlite-vec extension
        let vec_enabled = Self::try_load_vec_extension(&conn);

        if vec_enabled {
            conn.execute_batch(VEC_SCHEMA)
                .map_err(|e| RagError::database(format!("Failed to create vec table: {}", e)))?;
            info!("sqlite-vec extension loaded successfully");
        } else {
            warn!("sqlite-vec extension not available - vector search disabled");
        }

        // Initialize HLC
        let hlc = HybridLogicalClock::new(node_id);

        info!("Database opened at {:?}", path);

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            node_id,
            hlc: Arc::new(Mutex::new(hlc)),
            vec_enabled,
        })
    }

    /// Configure SQLite connection for optimal performance.
    fn configure_connection(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -64000;
            PRAGMA busy_timeout = 30000;
            PRAGMA temp_store = MEMORY;
            PRAGMA mmap_size = 268435456;
            PRAGMA foreign_keys = ON;
            "#,
        )
        .map_err(|e| RagError::database(format!("Failed to configure connection: {}", e)))?;

        Ok(())
    }

    /// Try to load the sqlite-vec extension.
    fn try_load_vec_extension(conn: &Connection) -> bool {
        // Try common extension paths
        let paths = [
            "vec0",
            "libsqlite_vec",
            "/usr/local/lib/libsqlite_vec",
            "/opt/homebrew/lib/libsqlite_vec",
        ];

        unsafe {
            if conn.load_extension_enable().is_err() {
                return false;
            }

            for path in paths {
                if conn.load_extension(path, None).is_ok() {
                    let _ = conn.load_extension_disable();
                    return true;
                }
            }

            let _ = conn.load_extension_disable();
        }

        false
    }

    /// Get the next HLC value.
    fn next_hlc(&self) -> HybridLogicalClock {
        let mut hlc = self.hlc.lock().unwrap();
        *hlc = hlc.tick();
        *hlc
    }

    /// Check if vector search is available.
    pub fn vec_enabled(&self) -> bool {
        self.vec_enabled
    }

    /// Execute a blocking operation on the connection.
    fn with_conn<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&Connection) -> Result<R>,
    {
        let conn = self.conn.lock().map_err(|e| RagError::database(e.to_string()))?;
        f(&conn)
    }

    /// Execute a mutable blocking operation on the connection.
    fn with_conn_mut<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Connection) -> Result<R>,
    {
        let mut conn = self.conn.lock().map_err(|e| RagError::database(e.to_string()))?;
        f(&mut conn)
    }
}

#[async_trait]
impl Store for SqliteStore {
    // Collection operations

    async fn create_collection(&self, mut collection: Collection) -> Result<()> {
        collection.hlc = self.next_hlc();

        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO collections (name, description, created_at, hlc) VALUES (?1, ?2, ?3, ?4)",
                params![
                    collection.name,
                    collection.description,
                    collection.created_at as i64,
                    collection.hlc.to_bytes().as_slice(),
                ],
            )
            .map_err(|e| {
                if e.to_string().contains("UNIQUE constraint") {
                    RagError::CollectionExists {
                        name: collection.name.clone(),
                    }
                } else {
                    RagError::database(format!("Failed to create collection: {}", e))
                }
            })?;

            debug!("Created collection: {}", collection.name);
            Ok(())
        })
    }

    async fn get_collection(&self, name: &str) -> Result<Option<Collection>> {
        let name = name.to_string();
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare("SELECT name, description, created_at, hlc FROM collections WHERE name = ?1")
                .map_err(|e| RagError::database(e.to_string()))?;

            let result = stmt
                .query_row(params![name], |row| {
                    let hlc_bytes: Vec<u8> = row.get(3)?;
                    Ok(Collection {
                        name: row.get(0)?,
                        description: row.get(1)?,
                        created_at: row.get::<_, i64>(2)? as u64,
                        hlc: HybridLogicalClock::from_bytes(&hlc_bytes)
                            .unwrap_or_else(HybridLogicalClock::zero),
                    })
                })
                .optional()
                .map_err(|e| RagError::database(e.to_string()))?;

            Ok(result)
        })
    }

    async fn list_collections(&self) -> Result<Vec<Collection>> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare("SELECT name, description, created_at, hlc FROM collections ORDER BY name")
                .map_err(|e| RagError::database(e.to_string()))?;

            let collections = stmt
                .query_map([], |row| {
                    let hlc_bytes: Vec<u8> = row.get(3)?;
                    Ok(Collection {
                        name: row.get(0)?,
                        description: row.get(1)?,
                        created_at: row.get::<_, i64>(2)? as u64,
                        hlc: HybridLogicalClock::from_bytes(&hlc_bytes)
                            .unwrap_or_else(HybridLogicalClock::zero),
                    })
                })
                .map_err(|e| RagError::database(e.to_string()))?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| RagError::database(e.to_string()))?;

            Ok(collections)
        })
    }

    async fn delete_collection(&self, name: &str) -> Result<()> {
        let name = name.to_string();
        self.with_conn(|conn| {
            let deleted = conn
                .execute("DELETE FROM collections WHERE name = ?1", params![name])
                .map_err(|e| RagError::database(e.to_string()))?;

            if deleted == 0 {
                return Err(RagError::CollectionNotFound { name });
            }

            debug!("Deleted collection: {}", name);
            Ok(())
        })
    }

    // Document operations

    async fn insert_document(&self, mut doc: Document) -> Result<()> {
        doc.hlc = self.next_hlc();

        let content_hash = doc.content_hash.map(|h| h.to_vec());
        let metadata = serde_json::to_string(&doc.metadata)?;

        self.with_conn(|conn| {
            conn.execute(
                r#"
                INSERT INTO documents (id, collection, source_uri, content_hash, raw_content,
                                       content_type, metadata, created_at, updated_at, hlc)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                "#,
                params![
                    doc.id.to_string(),
                    doc.collection,
                    doc.source_uri,
                    content_hash,
                    doc.raw_content,
                    doc.content_type.to_string(),
                    metadata,
                    doc.created_at as i64,
                    doc.updated_at as i64,
                    doc.hlc.to_bytes().as_slice(),
                ],
            )
            .map_err(|e| RagError::database(format!("Failed to insert document: {}", e)))?;

            debug!("Inserted document: {}", doc.id);
            Ok(())
        })
    }

    async fn get_document(&self, id: Ulid) -> Result<Option<Document>> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    r#"
                    SELECT id, collection, source_uri, content_hash, raw_content,
                           content_type, metadata, created_at, updated_at, hlc
                    FROM documents WHERE id = ?1
                    "#,
                )
                .map_err(|e| RagError::database(e.to_string()))?;

            let result = stmt
                .query_row(params![id.to_string()], |row| {
                    Self::row_to_document(row)
                })
                .optional()
                .map_err(|e| RagError::database(e.to_string()))?;

            Ok(result)
        })
    }

    async fn get_document_by_uri(&self, uri: &str) -> Result<Option<Document>> {
        let uri = uri.to_string();
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    r#"
                    SELECT id, collection, source_uri, content_hash, raw_content,
                           content_type, metadata, created_at, updated_at, hlc
                    FROM documents WHERE source_uri = ?1
                    "#,
                )
                .map_err(|e| RagError::database(e.to_string()))?;

            let result = stmt
                .query_row(params![uri], |row| Self::row_to_document(row))
                .optional()
                .map_err(|e| RagError::database(e.to_string()))?;

            Ok(result)
        })
    }

    async fn list_documents(&self, collection: &str, limit: u32, offset: u32) -> Result<Vec<Document>> {
        let collection = collection.to_string();
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    r#"
                    SELECT id, collection, source_uri, content_hash, raw_content,
                           content_type, metadata, created_at, updated_at, hlc
                    FROM documents
                    WHERE collection = ?1
                    ORDER BY created_at DESC
                    LIMIT ?2 OFFSET ?3
                    "#,
                )
                .map_err(|e| RagError::database(e.to_string()))?;

            let documents = stmt
                .query_map(params![collection, limit, offset], |row| {
                    Self::row_to_document(row)
                })
                .map_err(|e| RagError::database(e.to_string()))?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| RagError::database(e.to_string()))?;

            Ok(documents)
        })
    }

    async fn delete_document(&self, id: Ulid) -> Result<()> {
        let vec_enabled = self.vec_enabled;
        self.with_conn(|conn| {
            // Delete embeddings first (if vec enabled)
            if vec_enabled {
                conn.execute(
                    "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?1)",
                    params![id.to_string()],
                )
                .map_err(|e| RagError::database(e.to_string()))?;
            }

            // Chunks are deleted by CASCADE
            let deleted = conn
                .execute("DELETE FROM documents WHERE id = ?1", params![id.to_string()])
                .map_err(|e| RagError::database(e.to_string()))?;

            if deleted == 0 {
                return Err(RagError::DocumentNotFound { id: id.to_string() });
            }

            debug!("Deleted document: {}", id);
            Ok(())
        })
    }

    // Chunk operations

    async fn insert_chunks(&self, chunks: &[Chunk]) -> Result<()> {
        let chunks: Vec<Chunk> = chunks.to_vec();
        self.with_conn(|conn| {
            let tx = conn
                .unchecked_transaction()
                .map_err(|e| RagError::database(e.to_string()))?;

            {
                let mut stmt = tx
                    .prepare(
                        r#"
                        INSERT INTO chunks (id, doc_id, chunk_index, content, token_count,
                                           start_line, end_line, content_hash, hlc)
                        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                        "#,
                    )
                    .map_err(|e| RagError::database(e.to_string()))?;

                for chunk in &chunks {
                    let content_hash = chunk.content_hash.map(|h| h.to_vec());
                    stmt.execute(params![
                        chunk.id.to_string(),
                        chunk.doc_id.to_string(),
                        chunk.chunk_index,
                        chunk.content,
                        chunk.token_count,
                        chunk.start_line,
                        chunk.end_line,
                        content_hash,
                        chunk.hlc.to_bytes().as_slice(),
                    ])
                    .map_err(|e| RagError::database(format!("Failed to insert chunk: {}", e)))?;
                }
            }

            tx.commit()
                .map_err(|e| RagError::database(e.to_string()))?;

            debug!("Inserted {} chunks", chunks.len());
            Ok(())
        })
    }

    async fn get_chunks_for_document(&self, doc_id: Ulid) -> Result<Vec<Chunk>> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    r#"
                    SELECT id, doc_id, chunk_index, content, token_count,
                           start_line, end_line, content_hash, hlc
                    FROM chunks
                    WHERE doc_id = ?1
                    ORDER BY chunk_index
                    "#,
                )
                .map_err(|e| RagError::database(e.to_string()))?;

            let chunks = stmt
                .query_map(params![doc_id.to_string()], |row| Self::row_to_chunk(row))
                .map_err(|e| RagError::database(e.to_string()))?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| RagError::database(e.to_string()))?;

            Ok(chunks)
        })
    }

    async fn get_chunk(&self, id: Ulid) -> Result<Option<Chunk>> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    r#"
                    SELECT id, doc_id, chunk_index, content, token_count,
                           start_line, end_line, content_hash, hlc
                    FROM chunks WHERE id = ?1
                    "#,
                )
                .map_err(|e| RagError::database(e.to_string()))?;

            let result = stmt
                .query_row(params![id.to_string()], |row| Self::row_to_chunk(row))
                .optional()
                .map_err(|e| RagError::database(e.to_string()))?;

            Ok(result)
        })
    }

    async fn delete_chunks_for_document(&self, doc_id: Ulid) -> Result<()> {
        let vec_enabled = self.vec_enabled;
        self.with_conn(|conn| {
            // Delete embeddings first
            if vec_enabled {
                conn.execute(
                    "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?1)",
                    params![doc_id.to_string()],
                )
                .map_err(|e| RagError::database(e.to_string()))?;
            }

            conn.execute(
                "DELETE FROM chunks WHERE doc_id = ?1",
                params![doc_id.to_string()],
            )
            .map_err(|e| RagError::database(e.to_string()))?;

            Ok(())
        })
    }

    // Embedding operations

    async fn insert_embeddings(&self, chunk_ids: &[Ulid], embeddings: &[Vec<f32>]) -> Result<()> {
        if !self.vec_enabled {
            return Err(RagError::database("sqlite-vec extension not loaded"));
        }

        if chunk_ids.len() != embeddings.len() {
            return Err(RagError::invalid_argument(
                "chunk_ids and embeddings must have same length",
            ));
        }

        let chunk_ids: Vec<Ulid> = chunk_ids.to_vec();
        let embeddings: Vec<Vec<f32>> = embeddings.to_vec();

        self.with_conn(|conn| {
            let tx = conn
                .unchecked_transaction()
                .map_err(|e| RagError::database(e.to_string()))?;

            {
                let mut stmt = tx
                    .prepare("INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)")
                    .map_err(|e| RagError::database(e.to_string()))?;

                for (chunk_id, embedding) in chunk_ids.iter().zip(embeddings.iter()) {
                    let embedding_bytes = Self::vec_to_bytes(embedding);
                    stmt.execute(params![chunk_id.to_string(), embedding_bytes])
                        .map_err(|e| RagError::database(format!("Failed to insert embedding: {}", e)))?;
                }
            }

            tx.commit()
                .map_err(|e| RagError::database(e.to_string()))?;

            debug!("Inserted {} embeddings", chunk_ids.len());
            Ok(())
        })
    }

    // Search operations

    async fn vector_search(
        &self,
        embedding: &[f32],
        k: u32,
        collection: Option<&str>,
    ) -> Result<Vec<(Ulid, f32)>> {
        if !self.vec_enabled {
            return Err(RagError::database("sqlite-vec extension not loaded"));
        }

        let embedding_bytes = Self::vec_to_bytes(embedding);
        let collection = collection.map(String::from);

        self.with_conn(move |conn| {
            if let Some(coll) = &collection {
                let mut stmt = conn
                    .prepare(
                        r#"
                        SELECT v.chunk_id, v.distance
                        FROM vec_chunks v
                        JOIN chunks c ON c.id = v.chunk_id
                        JOIN documents d ON d.id = c.doc_id
                        WHERE d.collection = ?2
                        AND v.embedding MATCH ?1
                        ORDER BY v.distance
                        LIMIT ?3
                        "#,
                    )
                    .map_err(|e| RagError::database(e.to_string()))?;

                let rows = stmt
                    .query_map(params![embedding_bytes, coll, k], |row| {
                        let id_str: String = row.get(0)?;
                        let distance: f64 = row.get(1)?;
                        let similarity = 1.0 - distance as f32;
                        Ok((
                            Ulid::from_string(&id_str).unwrap_or_else(|_| Ulid::nil()),
                            similarity,
                        ))
                    })
                    .map_err(|e| RagError::database(e.to_string()))?;

                let results: Vec<_> = rows
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| RagError::database(e.to_string()))?;

                Ok(results)
            } else {
                let mut stmt = conn
                    .prepare(
                        r#"
                        SELECT chunk_id, distance
                        FROM vec_chunks
                        WHERE embedding MATCH ?1
                        ORDER BY distance
                        LIMIT ?2
                        "#,
                    )
                    .map_err(|e| RagError::database(e.to_string()))?;

                let rows = stmt
                    .query_map(params![embedding_bytes, k], |row| {
                        let id_str: String = row.get(0)?;
                        let distance: f64 = row.get(1)?;
                        let similarity = 1.0 - distance as f32;
                        Ok((
                            Ulid::from_string(&id_str).unwrap_or_else(|_| Ulid::nil()),
                            similarity,
                        ))
                    })
                    .map_err(|e| RagError::database(e.to_string()))?;

                let results: Vec<_> = rows
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| RagError::database(e.to_string()))?;

                Ok(results)
            }
        })
    }

    async fn keyword_search(
        &self,
        query: &str,
        k: u32,
        collection: Option<&str>,
    ) -> Result<Vec<(Ulid, f32)>> {
        // Escape FTS5 special characters
        let escaped_query = Self::escape_fts5_query(query);
        let collection = collection.map(String::from);

        self.with_conn(move |conn| {
            if let Some(coll) = &collection {
                let mut stmt = conn
                    .prepare(
                        r#"
                        SELECT c.id, bm25(chunks_fts) as score
                        FROM chunks_fts f
                        JOIN chunks c ON c.rowid = f.rowid
                        JOIN documents d ON d.id = c.doc_id
                        WHERE chunks_fts MATCH ?1
                        AND d.collection = ?2
                        ORDER BY score
                        LIMIT ?3
                        "#,
                    )
                    .map_err(|e| RagError::database(e.to_string()))?;

                let rows = stmt
                    .query_map(params![escaped_query, coll, k], |row| {
                        let id_str: String = row.get(0)?;
                        let score: f64 = row.get(1)?;
                        let similarity = (-score) as f32;
                        Ok((
                            Ulid::from_string(&id_str).unwrap_or_else(|_| Ulid::nil()),
                            similarity,
                        ))
                    })
                    .map_err(|e| RagError::database(e.to_string()))?;

                let results: Vec<_> = rows
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| RagError::database(e.to_string()))?;

                Ok(results)
            } else {
                let mut stmt = conn
                    .prepare(
                        r#"
                        SELECT c.id, bm25(chunks_fts) as score
                        FROM chunks_fts f
                        JOIN chunks c ON c.rowid = f.rowid
                        WHERE chunks_fts MATCH ?1
                        ORDER BY score
                        LIMIT ?2
                        "#,
                    )
                    .map_err(|e| RagError::database(e.to_string()))?;

                let rows = stmt
                    .query_map(params![escaped_query, k], |row| {
                        let id_str: String = row.get(0)?;
                        let score: f64 = row.get(1)?;
                        let similarity = (-score) as f32;
                        Ok((
                            Ulid::from_string(&id_str).unwrap_or_else(|_| Ulid::nil()),
                            similarity,
                        ))
                    })
                    .map_err(|e| RagError::database(e.to_string()))?;

                let results: Vec<_> = rows
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| RagError::database(e.to_string()))?;

                Ok(results)
            }
        })
    }

    // Stats

    async fn get_stats(&self, collection: Option<&str>) -> Result<Stats> {
        let collection = collection.map(String::from);
        let vec_enabled = self.vec_enabled;

        self.with_conn(move |conn| {
            let collections: u64 = conn
                .query_row("SELECT COUNT(*) FROM collections", [], |row| row.get(0))
                .map_err(|e| RagError::database(e.to_string()))?;

            let (documents, chunks): (u64, u64) = if let Some(ref coll) = collection {
                let docs: u64 = conn
                    .query_row(
                        "SELECT COUNT(*) FROM documents WHERE collection = ?1",
                        params![coll],
                        |row| row.get(0),
                    )
                    .map_err(|e| RagError::database(e.to_string()))?;

                let chunks: u64 = conn
                    .query_row(
                        r#"
                        SELECT COUNT(*) FROM chunks c
                        JOIN documents d ON d.id = c.doc_id
                        WHERE d.collection = ?1
                        "#,
                        params![coll],
                        |row| row.get(0),
                    )
                    .map_err(|e| RagError::database(e.to_string()))?;

                (docs, chunks)
            } else {
                let docs: u64 = conn
                    .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
                    .map_err(|e| RagError::database(e.to_string()))?;

                let chunks: u64 = conn
                    .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
                    .map_err(|e| RagError::database(e.to_string()))?;

                (docs, chunks)
            };

            let embeddings: u64 = if vec_enabled {
                conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |row| row.get(0))
                    .unwrap_or(0)
            } else {
                0
            };

            // Get page count and page size to estimate storage
            let page_count: u64 = conn
                .query_row("PRAGMA page_count", [], |row| row.get(0))
                .unwrap_or(0);
            let page_size: u64 = conn
                .query_row("PRAGMA page_size", [], |row| row.get(0))
                .unwrap_or(4096);

            Ok(Stats {
                collections,
                documents,
                chunks,
                embeddings,
                storage_bytes: page_count * page_size,
                filter: collection,
            })
        })
    }

    // Sync operations

    async fn get_watermark(&self) -> Result<HybridLogicalClock> {
        self.with_conn(|conn| {
            // Get max HLC from all tables
            let result: Option<Vec<u8>> = conn
                .query_row(
                    r#"
                    SELECT MAX(hlc) FROM (
                        SELECT hlc FROM collections
                        UNION ALL
                        SELECT hlc FROM documents
                        UNION ALL
                        SELECT hlc FROM chunks
                    )
                    "#,
                    [],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| RagError::database(e.to_string()))?
                .flatten();

            match result {
                Some(bytes) => Ok(HybridLogicalClock::from_bytes(&bytes)
                    .unwrap_or_else(HybridLogicalClock::zero)),
                None => Ok(HybridLogicalClock::zero()),
            }
        })
    }

    async fn get_changes_since(&self, _hlc: &HybridLogicalClock) -> Result<Vec<SyncChange>> {
        // TODO: Implement full sync change retrieval
        // This would query all tables for rows with HLC > given HLC
        Ok(Vec::new())
    }

    async fn apply_changes(&self, _changes: &[SyncChange]) -> Result<()> {
        // TODO: Implement applying sync changes
        // This would insert/update rows, handling conflicts via LWW
        Ok(())
    }
}

// Helper methods
impl SqliteStore {
    /// Convert a row to a Document.
    fn row_to_document(row: &rusqlite::Row<'_>) -> rusqlite::Result<Document> {
        let id_str: String = row.get(0)?;
        let content_hash: Option<Vec<u8>> = row.get(3)?;
        let content_type_str: String = row.get(5)?;
        let metadata_str: String = row.get(6)?;
        let hlc_bytes: Vec<u8> = row.get(9)?;

        Ok(Document {
            id: Ulid::from_string(&id_str).unwrap_or_else(|_| Ulid::nil()),
            collection: row.get(1)?,
            source_uri: row.get(2)?,
            content_hash: content_hash.and_then(|v| v.try_into().ok()),
            raw_content: row.get(4)?,
            content_type: ContentType::from_path(&content_type_str),
            metadata: serde_json::from_str(&metadata_str).unwrap_or_default(),
            created_at: row.get::<_, i64>(7)? as u64,
            updated_at: row.get::<_, i64>(8)? as u64,
            hlc: HybridLogicalClock::from_bytes(&hlc_bytes)
                .unwrap_or_else(HybridLogicalClock::zero),
        })
    }

    /// Convert a row to a Chunk.
    fn row_to_chunk(row: &rusqlite::Row<'_>) -> rusqlite::Result<Chunk> {
        let id_str: String = row.get(0)?;
        let doc_id_str: String = row.get(1)?;
        let content_hash: Option<Vec<u8>> = row.get(7)?;
        let hlc_bytes: Vec<u8> = row.get(8)?;

        Ok(Chunk {
            id: Ulid::from_string(&id_str).unwrap_or_else(|_| Ulid::nil()),
            doc_id: Ulid::from_string(&doc_id_str).unwrap_or_else(|_| Ulid::nil()),
            chunk_index: row.get(2)?,
            content: row.get(3)?,
            token_count: row.get(4)?,
            start_line: row.get(5)?,
            end_line: row.get(6)?,
            content_hash: content_hash.and_then(|v| v.try_into().ok()),
            hlc: HybridLogicalClock::from_bytes(&hlc_bytes)
                .unwrap_or_else(HybridLogicalClock::zero),
        })
    }

    /// Convert f32 vector to bytes (little-endian).
    fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Escape FTS5 query special characters.
    fn escape_fts5_query(query: &str) -> String {
        // Simple escaping: wrap each term in quotes if it contains special chars
        query
            .split_whitespace()
            .map(|term| {
                if term.contains(|c: char| "+-*()\"".contains(c)) {
                    format!("\"{}\"", term.replace('"', "\"\""))
                } else {
                    term.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_open_memory() {
        let store = SqliteStore::open_memory(1).unwrap();
        assert!(store.list_collections().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_collection_crud() {
        let store = SqliteStore::open_memory(1).unwrap();

        // Create
        let coll = Collection::new("test", Some("Test collection"));
        store.create_collection(coll).await.unwrap();

        // Read
        let retrieved = store.get_collection("test").await.unwrap().unwrap();
        assert_eq!(retrieved.name, "test");
        assert_eq!(retrieved.description, Some("Test collection".to_string()));

        // List
        let all = store.list_collections().await.unwrap();
        assert_eq!(all.len(), 1);

        // Delete
        store.delete_collection("test").await.unwrap();
        assert!(store.get_collection("test").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_document_crud() {
        let store = SqliteStore::open_memory(1).unwrap();

        // Create collection first
        store
            .create_collection(Collection::new("test", None))
            .await
            .unwrap();

        // Create document
        let doc = Document::new("test", "file://test.rs", "fn main() {}", ContentType::Rust);
        let doc_id = doc.id;
        store.insert_document(doc).await.unwrap();

        // Read
        let retrieved = store.get_document(doc_id).await.unwrap().unwrap();
        assert_eq!(retrieved.source_uri, "file://test.rs");

        // Read by URI
        let by_uri = store
            .get_document_by_uri("file://test.rs")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(by_uri.id, doc_id);

        // Delete
        store.delete_document(doc_id).await.unwrap();
        assert!(store.get_document(doc_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_chunks() {
        let store = SqliteStore::open_memory(1).unwrap();

        store
            .create_collection(Collection::new("test", None))
            .await
            .unwrap();

        let doc = Document::new("test", "file://test.rs", "fn main() {}", ContentType::Rust);
        let doc_id = doc.id;
        store.insert_document(doc).await.unwrap();

        // Insert chunks
        let chunks = vec![
            Chunk::new(doc_id, 0, "fn main() {", 5, 1, 1),
            Chunk::new(doc_id, 1, "    println!(\"Hello\");", 8, 2, 2),
            Chunk::new(doc_id, 2, "}", 1, 3, 3),
        ];
        store.insert_chunks(&chunks).await.unwrap();

        // Retrieve
        let retrieved = store.get_chunks_for_document(doc_id).await.unwrap();
        assert_eq!(retrieved.len(), 3);
        assert_eq!(retrieved[0].chunk_index, 0);
        assert_eq!(retrieved[1].chunk_index, 1);
    }

    #[tokio::test]
    async fn test_stats() {
        let store = SqliteStore::open_memory(1).unwrap();

        store
            .create_collection(Collection::new("test", None))
            .await
            .unwrap();

        let doc = Document::new("test", "file://test.rs", "fn main() {}", ContentType::Rust);
        let doc_id = doc.id;
        store.insert_document(doc).await.unwrap();

        let chunks = vec![Chunk::new(doc_id, 0, "fn main() {}", 5, 1, 1)];
        store.insert_chunks(&chunks).await.unwrap();

        let stats = store.get_stats(None).await.unwrap();
        assert_eq!(stats.collections, 1);
        assert_eq!(stats.documents, 1);
        assert_eq!(stats.chunks, 1);
    }

    #[tokio::test]
    async fn test_keyword_search() {
        let store = SqliteStore::open_memory(1).unwrap();

        store
            .create_collection(Collection::new("test", None))
            .await
            .unwrap();

        let doc = Document::new("test", "file://test.rs", "fn main() {}", ContentType::Rust);
        let doc_id = doc.id;
        store.insert_document(doc).await.unwrap();

        let chunks = vec![
            Chunk::new(doc_id, 0, "fn main() { println!(\"Hello World\"); }", 10, 1, 1),
            Chunk::new(doc_id, 1, "fn helper() { return 42; }", 8, 2, 2),
        ];
        store.insert_chunks(&chunks).await.unwrap();

        // Search
        let results = store.keyword_search("Hello World", 10, None).await.unwrap();
        assert!(!results.is_empty());
    }
}
