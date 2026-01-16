//! Configuration types for the RAG system.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for the RAG system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Database configuration.
    #[serde(default)]
    pub database: DatabaseConfig,

    /// Embedding configuration.
    #[serde(default)]
    pub embedding: EmbeddingConfig,

    /// Chunking configuration.
    #[serde(default)]
    pub chunking: ChunkingConfig,

    /// Search configuration.
    #[serde(default)]
    pub search: SearchConfig,

    /// Sync configuration.
    #[serde(default)]
    pub sync: SyncConfig,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            database: DatabaseConfig::default(),
            embedding: EmbeddingConfig::default(),
            chunking: ChunkingConfig::default(),
            search: SearchConfig::default(),
            sync: SyncConfig::default(),
        }
    }
}

/// Database configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Path to SQLite database file.
    pub path: PathBuf,

    /// Node ID for distributed sync (0-65535).
    pub node_id: u16,

    /// Enable WAL mode (recommended).
    #[serde(default = "default_true")]
    pub wal_mode: bool,

    /// SQLite cache size in KB (negative = KB, positive = pages).
    #[serde(default = "default_cache_size")]
    pub cache_size: i32,

    /// Busy timeout in milliseconds.
    #[serde(default = "default_busy_timeout")]
    pub busy_timeout_ms: u32,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: default_database_path(),
            node_id: 1,
            wal_mode: true,
            cache_size: -64000, // 64MB
            busy_timeout_ms: 30000,
        }
    }
}

/// Embedding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Path to ONNX model directory.
    pub model_path: PathBuf,

    /// Batch size for embedding.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Use GPU if available.
    #[serde(default)]
    pub use_gpu: bool,

    /// Number of threads for CPU inference.
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_path: default_model_path(),
            batch_size: 32,
            use_gpu: false,
            num_threads: 4,
        }
    }
}

/// Chunking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Maximum tokens per chunk.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Minimum tokens per chunk.
    #[serde(default = "default_min_tokens")]
    pub min_tokens: usize,

    /// Token overlap for sliding window.
    #[serde(default)]
    pub overlap_tokens: usize,

    /// Use AST-aware chunking for code.
    #[serde(default = "default_true")]
    pub ast_aware: bool,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            min_tokens: 50,
            overlap_tokens: 0,
            ast_aware: true,
        }
    }
}

/// Search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Default number of results.
    #[serde(default = "default_top_k")]
    pub default_top_k: usize,

    /// Maximum number of results.
    #[serde(default = "default_max_top_k")]
    pub max_top_k: usize,

    /// Use hybrid search by default.
    #[serde(default = "default_true")]
    pub hybrid: bool,

    /// Hybrid search alpha (0 = keyword only, 1 = vector only).
    #[serde(default = "default_hybrid_alpha")]
    pub hybrid_alpha: f32,

    /// RRF constant k.
    #[serde(default = "default_rrf_k")]
    pub rrf_k: u32,

    /// Expand context to adjacent chunks.
    #[serde(default = "default_true")]
    pub expand_context: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_top_k: 10,
            max_top_k: 100,
            hybrid: true,
            hybrid_alpha: 0.5,
            rrf_k: 60,
            expand_context: true,
        }
    }
}

/// Sync configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Enable sync.
    #[serde(default)]
    pub enabled: bool,

    /// Sync interval in seconds.
    #[serde(default = "default_sync_interval")]
    pub interval_secs: u64,

    /// Peer endpoints.
    #[serde(default)]
    pub peers: Vec<PeerConfig>,

    /// HTTP bind address for sync server.
    #[serde(default = "default_bind_address")]
    pub bind_address: String,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_secs: 60,
            peers: Vec::new(),
            bind_address: "0.0.0.0:8765".to_string(),
        }
    }
}

/// Peer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerConfig {
    /// Peer identifier.
    pub id: String,

    /// Peer endpoint URL.
    pub endpoint: String,
}

// Default value functions

fn default_true() -> bool {
    true
}

fn default_cache_size() -> i32 {
    -64000
}

fn default_busy_timeout() -> u32 {
    30000
}

fn default_batch_size() -> usize {
    32
}

fn default_num_threads() -> usize {
    4
}

fn default_max_tokens() -> usize {
    512
}

fn default_min_tokens() -> usize {
    50
}

fn default_top_k() -> usize {
    10
}

fn default_max_top_k() -> usize {
    100
}

fn default_hybrid_alpha() -> f32 {
    0.5
}

fn default_rrf_k() -> u32 {
    60
}

fn default_sync_interval() -> u64 {
    60
}

fn default_bind_address() -> String {
    "0.0.0.0:8765".to_string()
}

fn default_database_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("rag-mcp")
        .join("rag.db")
}

fn default_model_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("rag-mcp")
        .join("models")
        .join("nomic-embed-text-v1.5")
}

impl RagConfig {
    /// Load configuration from file.
    pub fn load(path: &std::path::Path) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content).map_err(|e| {
            crate::error::RagError::Config {
                message: format!("Failed to parse config: {}", e),
            }
        })?;
        Ok(config)
    }

    /// Load configuration from default paths.
    pub fn load_default() -> crate::error::Result<Self> {
        // Try user config first
        if let Some(config_dir) = dirs::config_dir() {
            let user_config = config_dir.join("rag-mcp").join("config.toml");
            if user_config.exists() {
                return Self::load(&user_config);
            }
        }

        // Try local config
        let local_config = PathBuf::from("rag-mcp.toml");
        if local_config.exists() {
            return Self::load(&local_config);
        }

        // Return defaults
        Ok(Self::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RagConfig::default();
        assert_eq!(config.search.default_top_k, 10);
        assert!(config.search.hybrid);
        assert_eq!(config.chunking.max_tokens, 512);
    }

    #[test]
    fn test_database_config_default() {
        let config = DatabaseConfig::default();
        assert!(config.wal_mode);
        assert_eq!(config.node_id, 1);
    }
}
