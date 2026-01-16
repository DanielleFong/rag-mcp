//! Error types for the RAG system.

use thiserror::Error;

/// Result type alias using RagError.
pub type Result<T> = std::result::Result<T, RagError>;

/// Errors that can occur in the RAG system.
#[derive(Error, Debug)]
pub enum RagError {
    /// Document not found.
    #[error("Document not found: {id}")]
    DocumentNotFound { id: String },

    /// Collection not found.
    #[error("Collection not found: {name}")]
    CollectionNotFound { name: String },

    /// Collection already exists.
    #[error("Collection already exists: {name}")]
    CollectionExists { name: String },

    /// Invalid argument provided.
    #[error("Invalid argument: {message}")]
    InvalidArgument { message: String },

    /// Invalid URI format.
    #[error("Invalid URI: {uri} - {reason}")]
    InvalidUri { uri: String, reason: String },

    /// Failed to load content from source.
    #[error("Failed to load content from {uri}: {reason}")]
    LoadFailed { uri: String, reason: String },

    /// Text exceeds maximum token limit.
    #[error("Text too long: {tokens} tokens exceeds maximum of {max_tokens}")]
    TextTooLong { tokens: usize, max_tokens: usize },

    /// Database error.
    #[error("Database error: {message}")]
    Database { message: String },

    /// Embedding model error.
    #[error("Embedding error: {message}")]
    Embedding { message: String },

    /// Chunking error.
    #[error("Chunking error: {message}")]
    Chunking { message: String },

    /// Sync error.
    #[error("Sync error: {message}")]
    Sync { message: String },

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Configuration error.
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// Internal error (unexpected).
    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl RagError {
    /// Create an invalid argument error.
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument {
            message: message.into(),
        }
    }

    /// Create a database error.
    pub fn database(message: impl Into<String>) -> Self {
        Self::Database {
            message: message.into(),
        }
    }

    /// Create an embedding error.
    pub fn embedding(message: impl Into<String>) -> Self {
        Self::Embedding {
            message: message.into(),
        }
    }

    /// Create a chunking error.
    pub fn chunking(message: impl Into<String>) -> Self {
        Self::Chunking {
            message: message.into(),
        }
    }

    /// Create a sync error.
    pub fn sync(message: impl Into<String>) -> Self {
        Self::Sync {
            message: message.into(),
        }
    }

    /// Create an internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Get the error code for MCP responses.
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::DocumentNotFound { .. } => "DOCUMENT_NOT_FOUND",
            Self::CollectionNotFound { .. } => "COLLECTION_NOT_FOUND",
            Self::CollectionExists { .. } => "COLLECTION_EXISTS",
            Self::InvalidArgument { .. } => "INVALID_ARGUMENT",
            Self::InvalidUri { .. } => "INVALID_URI",
            Self::LoadFailed { .. } => "LOAD_FAILED",
            Self::TextTooLong { .. } => "TEXT_TOO_LONG",
            Self::Database { .. } => "DATABASE_ERROR",
            Self::Embedding { .. } => "EMBEDDING_ERROR",
            Self::Chunking { .. } => "CHUNKING_ERROR",
            Self::Sync { .. } => "SYNC_ERROR",
            Self::Io(_) => "IO_ERROR",
            Self::Serialization(_) => "SERIALIZATION_ERROR",
            Self::Config { .. } => "CONFIG_ERROR",
            Self::Internal { .. } => "INTERNAL_ERROR",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RagError::DocumentNotFound {
            id: "abc123".to_string(),
        };
        assert!(err.to_string().contains("abc123"));
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(
            RagError::DocumentNotFound {
                id: "x".to_string()
            }
            .error_code(),
            "DOCUMENT_NOT_FOUND"
        );
        assert_eq!(
            RagError::database("test").error_code(),
            "DATABASE_ERROR"
        );
    }
}
