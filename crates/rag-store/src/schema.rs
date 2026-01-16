//! Database schema definitions.

/// Main schema SQL for initializing the database.
pub const SCHEMA: &str = r#"
-- Collections table
CREATE TABLE IF NOT EXISTS collections (
    name TEXT PRIMARY KEY,
    description TEXT,
    created_at INTEGER NOT NULL,
    hlc BLOB NOT NULL
);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    source_uri TEXT NOT NULL,
    content_hash BLOB,
    raw_content TEXT,
    content_type TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    hlc BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
CREATE INDEX IF NOT EXISTS idx_documents_source_uri ON documents(source_uri);
CREATE INDEX IF NOT EXISTS idx_documents_hlc ON documents(hlc);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content_hash BLOB,
    hlc BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hlc ON chunks(hlc);

-- FTS5 virtual table for keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=rowid
);

-- Triggers to keep FTS5 in sync with chunks table
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

-- Sync metadata table for tracking replication state
CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL
);
"#;

/// Schema for sqlite-vec virtual table.
/// This must be created separately after loading the extension.
pub const VEC_SCHEMA: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding float[768] distance_metric=cosine
);
"#;

/// Schema version for migrations.
pub const SCHEMA_VERSION: u32 = 1;
