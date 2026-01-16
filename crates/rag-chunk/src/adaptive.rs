//! Adaptive chunker that selects the best strategy based on content type.

use std::sync::Arc;

use rag_core::{ChunkConfig, ChunkData, Chunker, ContentType, Result};

use crate::RecursiveChunker;

/// Adaptive chunker that dispatches to specialized chunkers based on content type.
///
/// Falls back to RecursiveChunker for unsupported types.
pub struct AdaptiveChunker {
    /// Fallback recursive chunker.
    recursive: RecursiveChunker,

    /// Optional custom token counter.
    token_counter: Option<Arc<dyn Fn(&str) -> usize + Send + Sync>>,
}

impl AdaptiveChunker {
    /// Create a new adaptive chunker.
    pub fn new() -> Self {
        Self {
            recursive: RecursiveChunker::new(),
            token_counter: None,
        }
    }

    /// Create an adaptive chunker with a custom token counter.
    pub fn with_token_counter<F>(counter: F) -> Self
    where
        F: Fn(&str) -> usize + Send + Sync + 'static,
    {
        let counter = Arc::new(counter);
        let counter_clone = counter.clone();

        Self {
            recursive: RecursiveChunker::with_token_counter(move |s| counter_clone(s)),
            token_counter: Some(counter),
        }
    }
}

impl Default for AdaptiveChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker for AdaptiveChunker {
    fn chunk(
        &self,
        content: &str,
        content_type: ContentType,
        config: &ChunkConfig,
    ) -> Result<Vec<ChunkData>> {
        // For now, use recursive chunker for all types
        // In the future, we can add specialized chunkers:
        // - AstChunker for code (tree-sitter)
        // - SemanticChunker for markdown (pulldown-cmark)
        self.recursive.chunk(content, content_type, config)
    }

    fn supported_types(&self) -> Vec<ContentType> {
        self.recursive.supported_types()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_chunk_text() {
        let chunker = AdaptiveChunker::new();
        let config = ChunkConfig {
            max_tokens: 100,
            min_tokens: 1,
            overlap_tokens: 0,
        };

        let text = "Hello world. This is a test.";
        let chunks = chunker.chunk(text, ContentType::PlainText, &config).unwrap();

        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_adaptive_chunk_rust() {
        let chunker = AdaptiveChunker::new();
        let config = ChunkConfig {
            max_tokens: 50,
            min_tokens: 1,
            overlap_tokens: 0,
        };

        let code = r#"
fn main() {
    println!("Hello");
}

fn helper() {
    println!("Helper");
}
"#;
        let chunks = chunker.chunk(code, ContentType::Rust, &config).unwrap();

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_adaptive_with_token_counter() {
        // Custom counter: 1 token per word
        let chunker = AdaptiveChunker::with_token_counter(|s| s.split_whitespace().count());

        let config = ChunkConfig {
            max_tokens: 5,
            min_tokens: 1,
            overlap_tokens: 0,
        };

        let text = "one two three four five six seven eight nine ten";
        let chunks = chunker.chunk(text, ContentType::PlainText, &config).unwrap();

        assert!(chunks.len() >= 2);
    }
}
