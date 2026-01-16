# rag-chunk

Content-aware chunking strategies for different document types.

## Overview

| Attribute | Value |
|-----------|-------|
| Purpose | Split documents into embeddable chunks |
| Dependencies | tree-sitter, pulldown-cmark, tokenizers, rag-core |
| Est. Lines | ~1,500 |
| Confidence | HIGH |

This crate implements the `Chunker` trait from rag-core with multiple strategies.

---

## Module Structure

```
rag-chunk/
├── src/
│   ├── lib.rs              # Public exports
│   ├── chunker.rs          # AdaptiveChunker (main entry point)
│   ├── strategies/
│   │   ├── mod.rs
│   │   ├── ast.rs          # AST-aware chunking for code
│   │   ├── semantic.rs     # Heading-aware for markdown
│   │   ├── window.rs       # Sliding window for chat/logs
│   │   ├── record.rs       # Record-based for JSON/YAML
│   │   └── recursive.rs    # Recursive split fallback
│   ├── languages/
│   │   ├── mod.rs          # Language detection
│   │   ├── rust.rs         # Rust-specific AST queries
│   │   ├── python.rs       # Python-specific AST queries
│   │   └── ...             # Other languages
│   └── utils.rs            # Shared utilities
└── Cargo.toml
```

---

## AdaptiveChunker

The main entry point that selects the appropriate strategy based on content type.

```rust
use rag_core::{Chunk, ChunkOutput, ChunkMetadata, ChunkingSettings, ContentType, Result};

/// Adaptive chunker that selects strategy based on content type.
pub struct AdaptiveChunker {
    /// Token counter for size estimation.
    token_counter: TokenCounter,

    /// AST chunker for code.
    ast_chunker: AstChunker,

    /// Semantic chunker for markdown.
    semantic_chunker: SemanticChunker,

    /// Window chunker for chat.
    window_chunker: WindowChunker,

    /// Record chunker for structured data.
    record_chunker: RecordChunker,

    /// Recursive chunker for fallback.
    recursive_chunker: RecursiveChunker,
}

impl AdaptiveChunker {
    pub fn new(tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        let token_counter = TokenCounter::new(tokenizer_path)?;

        Ok(Self {
            token_counter,
            ast_chunker: AstChunker::new(),
            semantic_chunker: SemanticChunker::new(),
            window_chunker: WindowChunker::new(),
            record_chunker: RecordChunker::new(),
            recursive_chunker: RecursiveChunker::new(),
        })
    }
}

impl Chunker for AdaptiveChunker {
    fn chunk(
        &self,
        content: &str,
        content_type: ContentType,
        settings: &ChunkingSettings,
    ) -> Result<Vec<ChunkOutput>> {
        // Select strategy based on content type
        let chunks = match content_type {
            // Code - use AST-aware chunking
            ContentType::Rust
            | ContentType::Python
            | ContentType::TypeScript
            | ContentType::JavaScript
            | ContentType::Go
            | ContentType::Java
            | ContentType::Cpp
            | ContentType::C
            | ContentType::Ruby
            | ContentType::Php => {
                self.ast_chunker.chunk(content, content_type, settings, &self.token_counter)?
            }

            // Documentation - use semantic chunking
            ContentType::Markdown
            | ContentType::RestructuredText
            | ContentType::AsciiDoc
            | ContentType::Html => {
                self.semantic_chunker.chunk(content, content_type, settings, &self.token_counter)?
            }

            // Chat/logs - use sliding window
            ContentType::ChatLog => {
                self.window_chunker.chunk(content, settings, &self.token_counter)?
            }

            // Structured data - use record-based
            ContentType::Json
            | ContentType::Yaml
            | ContentType::Toml
            | ContentType::Xml => {
                self.record_chunker.chunk(content, content_type, settings, &self.token_counter)?
            }

            // Everything else - use recursive
            _ => {
                self.recursive_chunker.chunk(content, settings, &self.token_counter)?
            }
        };

        if chunks.is_empty() {
            return Err(RagError::EmptyChunks);
        }

        Ok(chunks)
    }

    fn supported_types(&self) -> &[ContentType] {
        &[
            ContentType::Rust,
            ContentType::Python,
            ContentType::TypeScript,
            ContentType::JavaScript,
            ContentType::Go,
            ContentType::Java,
            ContentType::Cpp,
            ContentType::C,
            ContentType::Markdown,
            ContentType::Html,
            ContentType::PlainText,
            ContentType::Json,
            ContentType::Yaml,
            ContentType::ChatLog,
        ]
    }
}
```

---

## AST-Aware Chunking (Code)

```rust
use tree_sitter::{Parser, Query, QueryCursor, Language};

/// AST-aware chunker for programming languages.
pub struct AstChunker {
    /// Parser instance (reused).
    parser: Parser,

    /// Language configurations.
    languages: HashMap<ContentType, LanguageConfig>,
}

struct LanguageConfig {
    /// tree-sitter Language.
    language: Language,

    /// Query for semantic boundaries.
    boundary_query: Query,

    /// Node types that form boundaries.
    boundary_types: Vec<&'static str>,
}

impl AstChunker {
    pub fn new() -> Self {
        let mut languages = HashMap::new();

        // Rust configuration
        languages.insert(ContentType::Rust, LanguageConfig {
            language: tree_sitter_rust::language(),
            boundary_query: Query::new(
                tree_sitter_rust::language(),
                "(function_item) @func
                 (impl_item) @impl
                 (struct_item) @struct
                 (enum_item) @enum
                 (mod_item) @mod
                 (trait_item) @trait"
            ).unwrap(),
            boundary_types: vec![
                "function_item", "impl_item", "struct_item",
                "enum_item", "mod_item", "trait_item"
            ],
        });

        // Python configuration
        languages.insert(ContentType::Python, LanguageConfig {
            language: tree_sitter_python::language(),
            boundary_query: Query::new(
                tree_sitter_python::language(),
                "(function_definition) @func
                 (class_definition) @class
                 (decorated_definition) @decorated"
            ).unwrap(),
            boundary_types: vec![
                "function_definition", "class_definition", "decorated_definition"
            ],
        });

        // TypeScript/JavaScript configuration
        let ts_lang = tree_sitter_typescript::language_typescript();
        languages.insert(ContentType::TypeScript, LanguageConfig {
            language: ts_lang,
            boundary_query: Query::new(
                ts_lang,
                "(function_declaration) @func
                 (class_declaration) @class
                 (method_definition) @method
                 (arrow_function) @arrow
                 (export_statement) @export"
            ).unwrap(),
            boundary_types: vec![
                "function_declaration", "class_declaration",
                "method_definition", "arrow_function", "export_statement"
            ],
        });

        // Add more languages...

        Self {
            parser: Parser::new(),
            languages,
        }
    }

    pub fn chunk(
        &self,
        content: &str,
        content_type: ContentType,
        settings: &ChunkingSettings,
        token_counter: &TokenCounter,
    ) -> Result<Vec<ChunkOutput>> {
        let config = self.languages.get(&content_type)
            .ok_or(RagError::UnsupportedContentType(content_type))?;

        // Parse the content
        self.parser.set_language(config.language)?;
        let tree = self.parser.parse(content, None)
            .ok_or(RagError::ParseError {
                content_type: format!("{:?}", content_type),
                reason: "Failed to parse".to_string(),
            })?;

        // Find semantic boundaries using query
        let mut cursor = QueryCursor::new();
        let matches = cursor.matches(&config.boundary_query, tree.root_node(), content.as_bytes());

        let mut chunks = Vec::new();
        let mut last_end = 0;

        for match_ in matches {
            for capture in match_.captures {
                let node = capture.node;
                let node_text = &content[node.byte_range()];
                let token_count = token_counter.count(node_text)?;

                // Handle gap before this node
                if node.start_byte() > last_end {
                    let gap_text = &content[last_end..node.start_byte()].trim();
                    if !gap_text.is_empty() {
                        let gap_tokens = token_counter.count(gap_text)?;
                        if gap_tokens >= settings.min_tokens {
                            chunks.push(self.create_chunk(
                                gap_text,
                                gap_tokens,
                                last_end,
                                node.start_byte(),
                                content,
                                None,
                                None,
                            ));
                        }
                    }
                }

                if token_count <= settings.max_tokens {
                    // Node fits in one chunk
                    chunks.push(self.create_chunk(
                        node_text,
                        token_count,
                        node.start_byte(),
                        node.end_byte(),
                        content,
                        Some(node.kind()),
                        self.extract_name(node, content),
                    ));
                } else {
                    // Node too large - split recursively
                    let sub_chunks = self.split_large_node(
                        node,
                        content,
                        settings,
                        token_counter,
                    )?;
                    chunks.extend(sub_chunks);
                }

                last_end = node.end_byte();
            }
        }

        // Handle trailing content
        if last_end < content.len() {
            let trailing = &content[last_end..].trim();
            if !trailing.is_empty() {
                let tokens = token_counter.count(trailing)?;
                if tokens >= settings.min_tokens {
                    chunks.push(self.create_chunk(
                        trailing,
                        tokens,
                        last_end,
                        content.len(),
                        content,
                        None,
                        None,
                    ));
                }
            }
        }

        // Merge small chunks
        chunks = self.merge_small_chunks(chunks, settings, token_counter)?;

        Ok(chunks)
    }

    fn split_large_node(
        &self,
        node: tree_sitter::Node,
        content: &str,
        settings: &ChunkingSettings,
        token_counter: &TokenCounter,
    ) -> Result<Vec<ChunkOutput>> {
        // Try splitting at child boundaries first
        let mut chunks = Vec::new();
        let mut current_text = String::new();
        let mut current_start = node.start_byte();

        for child in node.children(&mut node.walk()) {
            let child_text = &content[child.byte_range()];
            let combined = format!("{}{}", current_text, child_text);
            let combined_tokens = token_counter.count(&combined)?;

            if combined_tokens > settings.max_tokens && !current_text.is_empty() {
                // Flush current chunk
                let tokens = token_counter.count(&current_text)?;
                chunks.push(ChunkOutput {
                    content: current_text.clone(),
                    token_count: tokens as u32,
                    char_offset_start: current_start as u64,
                    char_offset_end: child.start_byte() as u64,
                    metadata: ChunkMetadata {
                        chunking_strategy: "ast_split".to_string(),
                        ..Default::default()
                    },
                });
                current_text = child_text.to_string();
                current_start = child.start_byte();
            } else {
                current_text = combined;
            }
        }

        // Flush remaining
        if !current_text.is_empty() {
            let tokens = token_counter.count(&current_text)?;
            chunks.push(ChunkOutput {
                content: current_text,
                token_count: tokens as u32,
                char_offset_start: current_start as u64,
                char_offset_end: node.end_byte() as u64,
                metadata: ChunkMetadata {
                    chunking_strategy: "ast_split".to_string(),
                    ..Default::default()
                },
            });
        }

        Ok(chunks)
    }

    fn extract_name(&self, node: tree_sitter::Node, content: &str) -> Option<String> {
        // Language-specific name extraction
        for child in node.children(&mut node.walk()) {
            match child.kind() {
                "identifier" | "name" | "property_identifier" => {
                    return Some(content[child.byte_range()].to_string());
                }
                _ => {}
            }
        }
        None
    }

    fn create_chunk(
        &self,
        text: &str,
        token_count: usize,
        start: usize,
        end: usize,
        full_content: &str,
        node_type: Option<&str>,
        node_name: Option<String>,
    ) -> ChunkOutput {
        let line_start = full_content[..start].matches('\n').count() as u32 + 1;
        let line_end = full_content[..end].matches('\n').count() as u32 + 1;

        ChunkOutput {
            content: text.to_string(),
            token_count: token_count as u32,
            char_offset_start: start as u64,
            char_offset_end: end as u64,
            metadata: ChunkMetadata {
                line_start,
                line_end,
                syntax_node_type: node_type.map(String::from),
                syntax_node_name: node_name,
                chunking_strategy: "ast".to_string(),
                ..Default::default()
            },
        }
    }

    fn merge_small_chunks(
        &self,
        chunks: Vec<ChunkOutput>,
        settings: &ChunkingSettings,
        token_counter: &TokenCounter,
    ) -> Result<Vec<ChunkOutput>> {
        let mut merged = Vec::new();
        let mut pending: Option<ChunkOutput> = None;

        for chunk in chunks {
            if let Some(mut p) = pending.take() {
                let combined_tokens = p.token_count + chunk.token_count;

                if p.token_count < settings.min_tokens as u32
                    && combined_tokens <= settings.max_tokens as u32
                {
                    // Merge
                    p.content.push_str("\n\n");
                    p.content.push_str(&chunk.content);
                    p.token_count = combined_tokens;
                    p.char_offset_end = chunk.char_offset_end;
                    p.metadata.line_end = chunk.metadata.line_end;
                    pending = Some(p);
                } else {
                    merged.push(p);
                    pending = Some(chunk);
                }
            } else {
                pending = Some(chunk);
            }
        }

        if let Some(p) = pending {
            merged.push(p);
        }

        Ok(merged)
    }
}
```

---

## Semantic Chunking (Markdown)

```rust
use pulldown_cmark::{Event, Parser, Tag, HeadingLevel};

/// Semantic chunker for markdown and documentation.
pub struct SemanticChunker;

impl SemanticChunker {
    pub fn new() -> Self {
        Self
    }

    pub fn chunk(
        &self,
        content: &str,
        content_type: ContentType,
        settings: &ChunkingSettings,
        token_counter: &TokenCounter,
    ) -> Result<Vec<ChunkOutput>> {
        match content_type {
            ContentType::Markdown => self.chunk_markdown(content, settings, token_counter),
            ContentType::Html => self.chunk_html(content, settings, token_counter),
            _ => Err(RagError::UnsupportedContentType(content_type)),
        }
    }

    fn chunk_markdown(
        &self,
        content: &str,
        settings: &ChunkingSettings,
        token_counter: &TokenCounter,
    ) -> Result<Vec<ChunkOutput>> {
        let parser = Parser::new(content);
        let mut chunks = Vec::new();

        let mut current_text = String::new();
        let mut current_start = 0;
        let mut heading_stack: Vec<String> = Vec::new();
        let mut in_heading = false;
        let mut heading_text = String::new();

        for (event, range) in parser.into_offset_iter() {
            match event {
                Event::Start(Tag::Heading(level, _, _)) => {
                    // Flush current chunk before new section
                    if !current_text.trim().is_empty() {
                        let tokens = token_counter.count(&current_text)?;
                        if tokens >= settings.min_tokens {
                            chunks.push(ChunkOutput {
                                content: current_text.trim().to_string(),
                                token_count: tokens as u32,
                                char_offset_start: current_start as u64,
                                char_offset_end: range.start as u64,
                                metadata: ChunkMetadata {
                                    heading_hierarchy: Some(heading_stack.clone()),
                                    chunking_strategy: "semantic".to_string(),
                                    ..Default::default()
                                },
                            });
                        }
                    }

                    // Update heading stack
                    let level_idx = match level {
                        HeadingLevel::H1 => 0,
                        HeadingLevel::H2 => 1,
                        HeadingLevel::H3 => 2,
                        HeadingLevel::H4 => 3,
                        HeadingLevel::H5 => 4,
                        HeadingLevel::H6 => 5,
                    };
                    heading_stack.truncate(level_idx);

                    in_heading = true;
                    heading_text.clear();
                    current_text.clear();
                    current_start = range.start;
                }

                Event::End(Tag::Heading(..)) => {
                    heading_stack.push(heading_text.trim().to_string());
                    current_text.push_str("# ");
                    current_text.push_str(&heading_text);
                    current_text.push_str("\n\n");
                    in_heading = false;
                }

                Event::Text(text) => {
                    if in_heading {
                        heading_text.push_str(&text);
                    } else {
                        current_text.push_str(&text);
                    }
                }

                Event::Code(code) => {
                    if !in_heading {
                        current_text.push('`');
                        current_text.push_str(&code);
                        current_text.push('`');
                    }
                }

                Event::SoftBreak | Event::HardBreak => {
                    if !in_heading {
                        current_text.push('\n');
                    }
                }

                Event::Start(Tag::CodeBlock(_)) => {
                    current_text.push_str("\n```\n");
                }

                Event::End(Tag::CodeBlock(_)) => {
                    current_text.push_str("\n```\n");
                }

                Event::Start(Tag::Paragraph) => {}
                Event::End(Tag::Paragraph) => {
                    current_text.push_str("\n\n");

                    // Check if we need to split
                    let tokens = token_counter.count(&current_text)?;
                    if tokens > settings.max_tokens {
                        // Split at paragraph boundary
                        chunks.push(ChunkOutput {
                            content: current_text.trim().to_string(),
                            token_count: tokens as u32,
                            char_offset_start: current_start as u64,
                            char_offset_end: range.end as u64,
                            metadata: ChunkMetadata {
                                heading_hierarchy: Some(heading_stack.clone()),
                                chunking_strategy: "semantic".to_string(),
                                ..Default::default()
                            },
                        });
                        current_text.clear();
                        current_start = range.end;
                    }
                }

                _ => {}
            }
        }

        // Flush remaining content
        if !current_text.trim().is_empty() {
            let tokens = token_counter.count(&current_text)?;
            if tokens >= settings.min_tokens {
                chunks.push(ChunkOutput {
                    content: current_text.trim().to_string(),
                    token_count: tokens as u32,
                    char_offset_start: current_start as u64,
                    char_offset_end: content.len() as u64,
                    metadata: ChunkMetadata {
                        heading_hierarchy: Some(heading_stack),
                        chunking_strategy: "semantic".to_string(),
                        ..Default::default()
                    },
                });
            }
        }

        Ok(chunks)
    }

    fn chunk_html(
        &self,
        content: &str,
        settings: &ChunkingSettings,
        token_counter: &TokenCounter,
    ) -> Result<Vec<ChunkOutput>> {
        // Convert HTML to text, then use recursive chunker
        let text = html2text::from_read(content.as_bytes(), 80);
        RecursiveChunker::new().chunk(&text, settings, token_counter)
    }
}
```

---

## Sliding Window Chunking (Chat)

```rust
/// Sliding window chunker for chat logs and conversations.
pub struct WindowChunker;

impl WindowChunker {
    pub fn new() -> Self {
        Self
    }

    pub fn chunk(
        &self,
        content: &str,
        settings: &ChunkingSettings,
        token_counter: &TokenCounter,
    ) -> Result<Vec<ChunkOutput>> {
        let window_tokens = settings.max_tokens.min(128);  // Default 128 for chat
        let overlap_tokens = settings.overlap_tokens.max(window_tokens / 2);  // 50% overlap

        // Tokenize to get token boundaries
        let tokens: Vec<&str> = content.split_whitespace().collect();
        let stride = window_tokens.saturating_sub(overlap_tokens).max(1);

        let mut chunks = Vec::new();
        let mut char_pos = 0;
        let mut token_pos = 0;

        while token_pos < tokens.len() {
            let end_pos = (token_pos + window_tokens).min(tokens.len());
            let window_tokens_slice = &tokens[token_pos..end_pos];
            let window_text = window_tokens_slice.join(" ");

            // Find actual character positions
            let chunk_start = char_pos;
            let chunk_end = if end_pos < tokens.len() {
                content.find(tokens[end_pos]).unwrap_or(content.len())
            } else {
                content.len()
            };

            let actual_tokens = token_counter.count(&window_text)?;

            chunks.push(ChunkOutput {
                content: window_text,
                token_count: actual_tokens as u32,
                char_offset_start: chunk_start as u64,
                char_offset_end: chunk_end as u64,
                metadata: ChunkMetadata {
                    chunking_strategy: "window".to_string(),
                    overlaps_previous: token_pos > 0,
                    overlaps_next: end_pos < tokens.len(),
                    ..Default::default()
                },
            });

            // Advance by stride
            for _ in 0..stride {
                if token_pos < tokens.len() {
                    char_pos = content[char_pos..].find(tokens[token_pos])
                        .map(|i| char_pos + i + tokens[token_pos].len())
                        .unwrap_or(char_pos);
                    token_pos += 1;
                }
            }
        }

        Ok(chunks)
    }
}
```

---

## Recursive Chunking (Fallback)

```rust
/// Recursive text splitter for general content.
pub struct RecursiveChunker {
    /// Separators in order of preference.
    separators: Vec<&'static str>,
}

impl RecursiveChunker {
    pub fn new() -> Self {
        Self {
            separators: vec![
                "\n\n\n",      // Section breaks
                "\n\n",        // Paragraphs
                "\n",          // Lines
                ". ",          // Sentences
                "! ",
                "? ",
                "; ",          // Clauses
                ", ",
                " ",           // Words
                "",            // Characters
            ],
        }
    }

    pub fn chunk(
        &self,
        content: &str,
        settings: &ChunkingSettings,
        token_counter: &TokenCounter,
    ) -> Result<Vec<ChunkOutput>> {
        self.split_recursive(content, 0, settings, token_counter)
    }

    fn split_recursive(
        &self,
        text: &str,
        sep_idx: usize,
        settings: &ChunkingSettings,
        token_counter: &TokenCounter,
    ) -> Result<Vec<ChunkOutput>> {
        let tokens = token_counter.count(text)?;

        // Base case: text fits
        if tokens <= settings.max_tokens {
            return Ok(vec![ChunkOutput {
                content: text.to_string(),
                token_count: tokens as u32,
                char_offset_start: 0,
                char_offset_end: text.len() as u64,
                metadata: ChunkMetadata {
                    chunking_strategy: "recursive".to_string(),
                    ..Default::default()
                },
            }]);
        }

        // Try current separator
        if sep_idx >= self.separators.len() {
            // No more separators - force split
            return self.force_split(text, settings, token_counter);
        }

        let separator = self.separators[sep_idx];
        let parts: Vec<&str> = if separator.is_empty() {
            text.chars().map(|c| {
                let start = text.find(c).unwrap();
                &text[start..start + c.len_utf8()]
            }).collect()
        } else {
            text.split(separator).collect()
        };

        if parts.len() == 1 {
            // Separator not found - try next
            return self.split_recursive(text, sep_idx + 1, settings, token_counter);
        }

        // Merge parts into chunks that fit
        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut current_start = 0;

        for (i, part) in parts.iter().enumerate() {
            let candidate = if current.is_empty() {
                part.to_string()
            } else {
                format!("{}{}{}", current, separator, part)
            };

            let candidate_tokens = token_counter.count(&candidate)?;

            if candidate_tokens > settings.max_tokens {
                // Current chunk is full
                if !current.is_empty() {
                    let current_tokens = token_counter.count(&current)?;
                    chunks.push(ChunkOutput {
                        content: current.clone(),
                        token_count: current_tokens as u32,
                        char_offset_start: current_start as u64,
                        char_offset_end: (current_start + current.len()) as u64,
                        metadata: ChunkMetadata {
                            chunking_strategy: "recursive".to_string(),
                            ..Default::default()
                        },
                    });
                }

                // Start new chunk with this part
                // If part itself is too large, recurse
                let part_tokens = token_counter.count(part)?;
                if part_tokens > settings.max_tokens {
                    let sub_chunks = self.split_recursive(part, sep_idx + 1, settings, token_counter)?;
                    chunks.extend(sub_chunks);
                    current.clear();
                } else {
                    current = part.to_string();
                    current_start = current_start + current.len() + separator.len();
                }
            } else {
                current = candidate;
            }
        }

        // Flush remaining
        if !current.is_empty() {
            let current_tokens = token_counter.count(&current)?;
            chunks.push(ChunkOutput {
                content: current.clone(),
                token_count: current_tokens as u32,
                char_offset_start: current_start as u64,
                char_offset_end: (current_start + current.len()) as u64,
                metadata: ChunkMetadata {
                    chunking_strategy: "recursive".to_string(),
                    ..Default::default()
                },
            });
        }

        Ok(chunks)
    }

    fn force_split(
        &self,
        text: &str,
        settings: &ChunkingSettings,
        token_counter: &TokenCounter,
    ) -> Result<Vec<ChunkOutput>> {
        // Last resort: split by character count estimate
        let chars_per_token = 4;  // Rough estimate
        let chunk_chars = settings.max_tokens * chars_per_token;

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = (start + chunk_chars).min(text.len());

            // Adjust to char boundary
            let end = text.floor_char_boundary(end);

            let chunk_text = &text[start..end];
            let tokens = token_counter.count(chunk_text)?;

            chunks.push(ChunkOutput {
                content: chunk_text.to_string(),
                token_count: tokens as u32,
                char_offset_start: start as u64,
                char_offset_end: end as u64,
                metadata: ChunkMetadata {
                    chunking_strategy: "force_split".to_string(),
                    ..Default::default()
                },
            });

            start = end;
        }

        Ok(chunks)
    }
}
```

---

## Confidence Assessment

| Component | Confidence | Notes |
|-----------|------------|-------|
| tree-sitter integration | HIGH | Mature library |
| AST chunking | HIGH | Well-understood technique |
| Markdown parsing | HIGH | pulldown-cmark is solid |
| Window chunking | HIGH | Simple algorithm |
| Recursive chunking | HIGH | Standard technique |
| Token counting | HIGH | Uses same tokenizer as embedder |
| Language support | MEDIUM | May need tuning per language |

---

## Cargo.toml

```toml
[package]
name = "rag-chunk"
version = "0.1.0"
edition = "2021"
description = "Content-aware chunking for the RAG system"
license = "MIT"

[dependencies]
rag-core = { path = "../rag-core" }
tree-sitter = "0.22"
tree-sitter-rust = "0.21"
tree-sitter-python = "0.21"
tree-sitter-typescript = "0.21"
tree-sitter-javascript = "0.21"
tree-sitter-go = "0.21"
tree-sitter-java = "0.21"
tree-sitter-c = "0.21"
tree-sitter-cpp = "0.21"
pulldown-cmark = "0.10"
html2text = "0.12"
tokenizers = "0.15"
tracing = "0.1"

[dev-dependencies]
tokio = { version = "1.0", features = ["macros", "rt-multi-thread"] }
```
