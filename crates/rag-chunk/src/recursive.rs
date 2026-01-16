//! Recursive text chunker.
//!
//! Splits text by trying progressively smaller separators until chunks
//! fit within the token limit.

use rag_core::{ChunkConfig, ChunkData, Chunker, ContentType, Result};

/// Recursive chunker that splits text by multiple separators.
///
/// Tries each separator in order until chunks are small enough:
/// 1. Double newline (paragraph breaks)
/// 2. Single newline
/// 3. Sentence boundaries (. ! ?)
/// 4. Word boundaries (space)
/// 5. Character (last resort)
pub struct RecursiveChunker {
    /// Function to count tokens in text.
    /// Uses simple word count approximation if None.
    token_counter: Option<Box<dyn Fn(&str) -> usize + Send + Sync>>,
}

impl RecursiveChunker {
    /// Create a new recursive chunker with default word-based token estimation.
    pub fn new() -> Self {
        Self { token_counter: None }
    }

    /// Create a chunker with a custom token counter.
    pub fn with_token_counter<F>(counter: F) -> Self
    where
        F: Fn(&str) -> usize + Send + Sync + 'static,
    {
        Self {
            token_counter: Some(Box::new(counter)),
        }
    }

    /// Count tokens in text.
    fn count_tokens(&self, text: &str) -> usize {
        match &self.token_counter {
            Some(counter) => counter(text),
            // Simple approximation: ~4 chars per token on average
            None => (text.len() / 4).max(1),
        }
    }

    /// Get separators for the given content type.
    fn separators(&self, content_type: &ContentType) -> Vec<&'static str> {
        match content_type {
            ContentType::Markdown => vec!["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
            ContentType::Rust
            | ContentType::Python
            | ContentType::TypeScript
            | ContentType::JavaScript
            | ContentType::Go
            | ContentType::Java
            | ContentType::Cpp
            | ContentType::C => vec!["\n\n", "\nfn ", "\ndef ", "\nfunc ", "\nclass ", "\nimpl ", "\n", " "],
            ContentType::Json | ContentType::Yaml | ContentType::Toml => {
                vec!["\n\n", "\n", ", ", " "]
            }
            _ => vec!["\n\n", "\n", ". ", " "],
        }
    }

    /// Split text by a separator.
    fn split_by_separator<'a>(&self, text: &'a str, separator: &str) -> Vec<&'a str> {
        if separator.is_empty() {
            // Character-level split as last resort
            return text
                .char_indices()
                .map(|(i, c)| &text[i..i + c.len_utf8()])
                .collect();
        }

        text.split(separator)
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Recursively chunk text.
    fn chunk_recursive(
        &self,
        text: &str,
        separators: &[&str],
        config: &ChunkConfig,
        start_line: u32,
    ) -> Vec<ChunkData> {
        let tokens = self.count_tokens(text);

        // If text fits in one chunk, return it
        if tokens <= config.max_tokens {
            return vec![ChunkData {
                content: text.to_string(),
                token_count: tokens,
                start_line,
                end_line: start_line + text.lines().count().saturating_sub(1) as u32,
            }];
        }

        // Try each separator
        for (sep_idx, separator) in separators.iter().enumerate() {
            let parts = self.split_by_separator(text, separator);

            if parts.len() <= 1 {
                continue;
            }

            let mut chunks = Vec::new();
            let mut current_chunk = String::new();
            let mut chunk_start_line = start_line;
            let mut current_line = start_line;

            for part in parts {
                let part_with_sep = if current_chunk.is_empty() {
                    part.to_string()
                } else {
                    format!("{}{}{}", current_chunk, separator, part)
                };

                let combined_tokens = self.count_tokens(&part_with_sep);

                if combined_tokens <= config.max_tokens {
                    // Add to current chunk
                    if current_chunk.is_empty() {
                        current_chunk = part.to_string();
                    } else {
                        current_chunk = part_with_sep;
                    }
                } else if current_chunk.is_empty() {
                    // Part itself is too big, recurse with next separator
                    let remaining_seps = &separators[sep_idx + 1..];
                    if remaining_seps.is_empty() {
                        // No more separators, just split by max chars
                        chunks.extend(self.split_by_size(part, config, current_line));
                    } else {
                        chunks.extend(self.chunk_recursive(part, remaining_seps, config, current_line));
                    }
                    current_line += part.lines().count() as u32;
                } else {
                    // Save current chunk and start new one
                    let tokens = self.count_tokens(&current_chunk);
                    if tokens >= config.min_tokens {
                        chunks.push(ChunkData {
                            content: current_chunk.clone(),
                            token_count: tokens,
                            start_line: chunk_start_line,
                            end_line: current_line.saturating_sub(1),
                        });
                    }

                    // Check if part itself fits
                    let part_tokens = self.count_tokens(part);
                    if part_tokens <= config.max_tokens {
                        current_chunk = part.to_string();
                        chunk_start_line = current_line;
                    } else {
                        // Part too big, recurse
                        let remaining_seps = &separators[sep_idx + 1..];
                        if remaining_seps.is_empty() {
                            chunks.extend(self.split_by_size(part, config, current_line));
                        } else {
                            chunks.extend(self.chunk_recursive(part, remaining_seps, config, current_line));
                        }
                        current_chunk = String::new();
                        chunk_start_line = current_line + part.lines().count() as u32;
                    }
                    current_line += part.lines().count() as u32;
                }
            }

            // Don't forget the last chunk
            if !current_chunk.is_empty() {
                let tokens = self.count_tokens(&current_chunk);
                if tokens >= config.min_tokens {
                    chunks.push(ChunkData {
                        content: current_chunk,
                        token_count: tokens,
                        start_line: chunk_start_line,
                        end_line: chunk_start_line + current_line.saturating_sub(chunk_start_line),
                    });
                }
            }

            if !chunks.is_empty() {
                return chunks;
            }
        }

        // Fallback: split by size
        self.split_by_size(text, config, start_line)
    }

    /// Split text by size (last resort).
    fn split_by_size(&self, text: &str, config: &ChunkConfig, start_line: u32) -> Vec<ChunkData> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let target_chars = config.max_tokens * 4; // Approximate chars per chunk
        let mut start = 0;
        let mut current_line = start_line;

        while start < chars.len() {
            let end = (start + target_chars).min(chars.len());

            // Try to break at word boundary
            let mut actual_end = end;
            if end < chars.len() {
                // Look for space backwards
                for i in (start..end).rev() {
                    if chars[i] == ' ' || chars[i] == '\n' {
                        actual_end = i + 1;
                        break;
                    }
                }
            }

            let chunk_text: String = chars[start..actual_end].iter().collect();
            let tokens = self.count_tokens(&chunk_text);
            let lines_in_chunk = chunk_text.lines().count() as u32;

            if tokens > 0 {
                chunks.push(ChunkData {
                    content: chunk_text,
                    token_count: tokens,
                    start_line: current_line,
                    end_line: current_line + lines_in_chunk.saturating_sub(1),
                });
            }

            current_line += lines_in_chunk;
            start = actual_end;
        }

        chunks
    }
}

impl Default for RecursiveChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker for RecursiveChunker {
    fn chunk(
        &self,
        content: &str,
        content_type: ContentType,
        config: &ChunkConfig,
    ) -> Result<Vec<ChunkData>> {
        if content.is_empty() {
            return Ok(Vec::new());
        }

        let separators = self.separators(&content_type);
        let chunks = self.chunk_recursive(content, &separators, config, 1);

        // Filter out chunks that are too small
        let chunks: Vec<_> = chunks
            .into_iter()
            .filter(|c| c.token_count >= config.min_tokens)
            .collect();

        Ok(chunks)
    }

    fn supported_types(&self) -> Vec<ContentType> {
        // Supports all types as a fallback
        vec![
            ContentType::PlainText,
            ContentType::Markdown,
            ContentType::Rust,
            ContentType::Python,
            ContentType::TypeScript,
            ContentType::JavaScript,
            ContentType::Go,
            ContentType::Java,
            ContentType::Cpp,
            ContentType::C,
            ContentType::Ruby,
            ContentType::Json,
            ContentType::Yaml,
            ContentType::Toml,
            ContentType::Html,
            ContentType::Unknown,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_chunk() {
        let chunker = RecursiveChunker::new();
        let config = ChunkConfig {
            max_tokens: 100,
            min_tokens: 1,
            overlap_tokens: 0,
        };

        let text = "Hello world. This is a test.";
        let chunks = chunker.chunk(text, ContentType::PlainText, &config).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, text);
    }

    #[test]
    fn test_paragraph_split() {
        // Use a custom token counter to make splitting more predictable
        let chunker = RecursiveChunker::with_token_counter(|s| s.split_whitespace().count());
        let config = ChunkConfig {
            max_tokens: 5, // Very low to force splits
            min_tokens: 1,
            overlap_tokens: 0,
        };

        let text = "First paragraph with several words here.\n\nSecond paragraph also with words.\n\nThird paragraph too.";
        let chunks = chunker.chunk(text, ContentType::PlainText, &config).unwrap();

        assert!(chunks.len() >= 2, "Expected at least 2 chunks, got {}", chunks.len());
    }

    #[test]
    fn test_line_numbers() {
        let chunker = RecursiveChunker::new();
        let config = ChunkConfig {
            max_tokens: 20,
            min_tokens: 1,
            overlap_tokens: 0,
        };

        let text = "Line 1\nLine 2\nLine 3\n\nLine 5\nLine 6";
        let chunks = chunker.chunk(text, ContentType::PlainText, &config).unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].start_line, 1);
    }

    #[test]
    fn test_empty_content() {
        let chunker = RecursiveChunker::new();
        let config = ChunkConfig::default();

        let chunks = chunker.chunk("", ContentType::PlainText, &config).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_supported_types() {
        let chunker = RecursiveChunker::new();
        let supported = chunker.supported_types();
        assert!(supported.contains(&ContentType::PlainText));
        assert!(supported.contains(&ContentType::Rust));
    }
}
