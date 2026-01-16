# FEATURES.md - Extended Feature Roadmap

This document outlines both core features and suggested enhancements for the RAG system. Features are organized by priority tier and include confidence assessments.

## Table of Contents

1. [Core Features (MVP)](#core-features-mvp)
2. [Enhanced Retrieval](#enhanced-retrieval)
3. [Advanced Chunking](#advanced-chunking)
4. [Intelligent Ingestion](#intelligent-ingestion)
5. [Query Intelligence](#query-intelligence)
6. [Observability & Debugging](#observability--debugging)
7. [Multi-Modal Support](#multi-modal-support)
8. [Integration Features](#integration-features)
9. [Performance Optimizations](#performance-optimizations)
10. [Security & Access Control](#security--access-control)
11. [Developer Experience](#developer-experience)
12. [AI-Assisted Features](#ai-assisted-features)

---

## Core Features (MVP)

These are essential for a functioning system.

### 1.1 Basic Ingestion Pipeline

| Feature | Description | Confidence |
|---------|-------------|------------|
| File ingestion | Load files from local filesystem | HIGH |
| URL ingestion | Fetch and process web pages | HIGH |
| Raw text ingestion | Accept inline text content | HIGH |
| Content type detection | Auto-detect based on extension + magic bytes | HIGH |
| Deduplication | Skip unchanged documents by content hash | HIGH |

### 1.2 Core Search

| Feature | Description | Confidence |
|---------|-------------|------------|
| Vector search | Semantic similarity via embeddings | HIGH |
| Keyword search | BM25 via FTS5 | HIGH |
| Hybrid search | RRF fusion of vector + keyword | HIGH |
| Collection filtering | Scope search to specific collections | HIGH |
| Result pagination | Offset-based pagination | HIGH |

### 1.3 Basic MCP Interface

| Feature | Description | Confidence |
|---------|-------------|------------|
| search tool | Query the knowledge base | HIGH |
| ingest tool | Add documents | HIGH |
| delete tool | Remove documents | HIGH |
| list_collections tool | Enumerate collections | HIGH |
| list_documents tool | Enumerate documents | HIGH |

### 1.4 Persistence

| Feature | Description | Confidence |
|---------|-------------|------------|
| SQLite storage | Core relational data | HIGH |
| sqlite-vec vectors | Embedding storage + search | HIGH |
| FTS5 index | Full-text search index | HIGH |
| WAL mode | Concurrent read access | HIGH |

---

## Enhanced Retrieval

Features to improve search quality and relevance.

### 2.1 Query Expansion

**Description**: Automatically expand queries with synonyms, related terms, or reformulations.

```rust
struct QueryExpander {
    // Synonym dictionary (WordNet-based or custom)
    synonyms: HashMap<String, Vec<String>>,
    // Embedding-based expansion
    embedder: Arc<dyn Embedder>,
}

impl QueryExpander {
    fn expand(&self, query: &str) -> Vec<String> {
        let mut expansions = vec![query.to_string()];

        // Synonym expansion
        for word in query.split_whitespace() {
            if let Some(syns) = self.synonyms.get(word) {
                for syn in syns.iter().take(2) {
                    expansions.push(query.replace(word, syn));
                }
            }
        }

        // Embedding-based: find similar queries from history
        // ...

        expansions
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Improves recall for imprecise queries |
| Complexity | MEDIUM - Requires synonym source |
| Confidence | MEDIUM |

### 2.2 Reranking

**Description**: Apply a cross-encoder model to rerank initial results for better precision.

```rust
struct CrossEncoderReranker {
    model: ort::Session,  // e.g., ms-marco-MiniLM
}

impl CrossEncoderReranker {
    fn rerank(&self, query: &str, results: Vec<SearchResult>) -> Vec<SearchResult> {
        // Create query-document pairs
        let pairs: Vec<(String, String)> = results.iter()
            .map(|r| (query.to_string(), r.chunk.content.clone()))
            .collect();

        // Score each pair
        let scores = self.model.run(pairs)?;

        // Sort by reranked score
        let mut reranked: Vec<_> = results.into_iter()
            .zip(scores)
            .collect();
        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        reranked.into_iter().map(|(r, _)| r).collect()
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Significant precision improvement |
| Complexity | MEDIUM - Additional model to load |
| Confidence | HIGH |
| Latency Impact | +50-100ms |

### 2.3 Contextual Retrieval (Anthropic-style)

**Description**: Prepend each chunk with LLM-generated context that situates it within the document.

```
Original chunk:
"The company's revenue grew 15% YoY."

With context:
"This chunk is from Acme Corp's Q3 2024 earnings report, specifically the
financial highlights section. The company's revenue grew 15% YoY."
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Reduces "lost in the middle" problem |
| Complexity | HIGH - Requires LLM call per chunk at ingest |
| Confidence | MEDIUM |
| Cost | Significant LLM API cost at ingest time |

### 2.4 Hypothetical Document Embeddings (HyDE)

**Description**: Generate a hypothetical answer to the query, embed that, and use it for retrieval.

```rust
async fn hyde_search(&self, query: &str) -> Result<Vec<SearchResult>> {
    // Generate hypothetical answer
    let hypothetical = self.llm.generate(&format!(
        "Write a short passage that would answer this question: {}", query
    )).await?;

    // Embed the hypothetical answer
    let embedding = self.embedder.embed_query(&hypothetical)?;

    // Search with hypothetical embedding
    self.store.vector_search(&embedding, self.config.vector_k, None).await
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Helps with abstract queries |
| Complexity | MEDIUM - Requires LLM integration |
| Confidence | MEDIUM |
| Latency Impact | +500-2000ms (LLM call) |

### 2.5 Multi-Vector Retrieval

**Description**: Embed documents at multiple granularities (sentence, paragraph, document) and search all levels.

```rust
struct MultiVectorStore {
    sentence_vectors: VecTable,
    paragraph_vectors: VecTable,
    document_vectors: VecTable,
}

impl MultiVectorStore {
    async fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        // Search all levels in parallel
        let (sentences, paragraphs, documents) = tokio::join!(
            self.sentence_vectors.search(query, k),
            self.paragraph_vectors.search(query, k),
            self.document_vectors.search(query, k),
        );

        // Merge and deduplicate
        self.merge_results(sentences, paragraphs, documents, k)
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Better for varied query types |
| Complexity | HIGH - 3x embedding storage |
| Confidence | MEDIUM |

### 2.6 Filtered Vector Search

**Description**: Apply metadata filters before or during vector search.

```sql
-- Pre-filter approach (sqlite-vec limitation workaround)
WITH filtered_chunks AS (
    SELECT c.id, c.content
    FROM chunks c
    JOIN documents d ON c.doc_id = d.id
    WHERE d.collection = ?
    AND json_extract(d.metadata, '$.language') = 'rust'
)
SELECT fc.id, vc.distance
FROM vec_chunks vc
JOIN filtered_chunks fc ON vc.chunk_id = fc.id
WHERE vc.embedding MATCH ?
ORDER BY vc.distance
LIMIT ?;
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Essential for multi-tenant use |
| Complexity | MEDIUM - SQL query complexity |
| Confidence | HIGH |

---

## Advanced Chunking

Features for smarter content segmentation.

### 3.1 Late Chunking

**Description**: Embed the full document first, then chunk while preserving full-document context in embeddings.

```rust
fn late_chunk(content: &str, embedder: &dyn Embedder) -> Vec<ChunkWithEmbedding> {
    // Embed full document (up to model max tokens)
    let full_embedding = embedder.embed_document(content)?;

    // Generate token-level embeddings
    let token_embeddings = embedder.embed_tokens(content)?;

    // Chunk by semantic boundaries
    let chunks = chunk_by_semantics(content);

    // Aggregate token embeddings for each chunk
    chunks.iter().map(|chunk| {
        let chunk_embedding = aggregate_token_embeddings(
            &token_embeddings,
            chunk.token_start,
            chunk.token_end,
        );
        ChunkWithEmbedding { chunk, embedding: chunk_embedding }
    }).collect()
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Preserves document context |
| Complexity | HIGH - Requires token-level embeddings |
| Confidence | LOW - Requires specific model support |

### 3.2 Agentic Chunking

**Description**: Use an LLM to determine optimal chunk boundaries based on semantic coherence.

```rust
async fn agentic_chunk(content: &str, llm: &dyn LLM) -> Vec<Chunk> {
    let prompt = format!(
        "Split this document into coherent chunks. Each chunk should be \
         self-contained and cover a single topic. Output chunk boundaries \
         as line numbers.\n\nDocument:\n{}", content
    );

    let response = llm.generate(&prompt).await?;
    parse_chunk_boundaries(&response, content)
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Better boundaries for complex docs |
| Complexity | MEDIUM - LLM integration |
| Confidence | MEDIUM |
| Cost | High LLM cost at ingest |

### 3.3 Proposition-Based Chunking

**Description**: Break documents into atomic propositions/facts rather than text spans.

```
Original:
"John, a 35-year-old software engineer from Seattle, works at Microsoft
where he leads the Azure team."

Propositions:
1. "John is 35 years old."
2. "John is a software engineer."
3. "John is from Seattle."
4. "John works at Microsoft."
5. "John leads the Azure team at Microsoft."
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Very precise retrieval |
| Complexity | HIGH - Requires NLP or LLM |
| Confidence | MEDIUM |

### 3.4 Hierarchical Chunking

**Description**: Create chunks at multiple levels (document → section → paragraph → sentence) with parent-child relationships.

```rust
struct HierarchicalChunk {
    id: Ulid,
    level: ChunkLevel,  // Document, Section, Paragraph, Sentence
    content: String,
    parent_id: Option<Ulid>,
    children_ids: Vec<Ulid>,
    embedding: Vec<f32>,
}

// At query time, retrieve leaf chunks but return with ancestor context
fn retrieve_with_context(chunk_id: Ulid) -> ChunkWithContext {
    let chunk = store.get_chunk(chunk_id);
    let ancestors = store.get_ancestors(chunk_id);

    ChunkWithContext {
        chunk,
        document_context: ancestors.document,
        section_context: ancestors.section,
        paragraph_context: ancestors.paragraph,
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Flexible retrieval granularity |
| Complexity | HIGH - Complex data model |
| Confidence | MEDIUM |

### 3.5 Code-Aware Chunking Enhancements

**Description**: Enhanced AST-based chunking with cross-reference awareness.

```rust
struct EnhancedCodeChunker {
    parser: tree_sitter::Parser,
    reference_tracker: ReferenceTracker,
}

impl EnhancedCodeChunker {
    fn chunk(&self, code: &str) -> Vec<CodeChunk> {
        let tree = self.parser.parse(code, None)?;

        // Extract functions, classes, etc.
        let nodes = extract_semantic_nodes(&tree);

        // Build reference graph
        let references = self.reference_tracker.analyze(&tree);

        // Create chunks with reference metadata
        nodes.iter().map(|node| {
            CodeChunk {
                content: node.text(),
                node_type: node.kind(),
                name: extract_name(node),
                // Include what this chunk references
                references: references.outgoing(node.id()),
                // Include what references this chunk
                referenced_by: references.incoming(node.id()),
                // Include imports needed to understand this chunk
                required_imports: references.required_imports(node.id()),
            }
        }).collect()
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Much better code understanding |
| Complexity | HIGH - Requires symbol resolution |
| Confidence | MEDIUM |

---

## Intelligent Ingestion

Features for smarter document processing.

### 4.1 Incremental Ingestion

**Description**: Only re-process changed portions of documents.

Already specified in DESIGN.md. **Confidence: HIGH**

### 4.2 File Watching

**Description**: Automatically detect and ingest file changes.

```rust
struct FileWatcher {
    watcher: notify::RecommendedWatcher,
    debouncer: Debouncer,
    ingester: Arc<Ingester>,
}

impl FileWatcher {
    async fn watch(&self, path: &Path, collection: &str) -> Result<()> {
        self.watcher.watch(path, RecursiveMode::Recursive)?;

        loop {
            let events = self.debouncer.next_batch().await;

            for event in events {
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) => {
                        self.ingester.ingest(&event.path, collection).await?;
                    }
                    EventKind::Remove(_) => {
                        self.ingester.delete_by_path(&event.path).await?;
                    }
                    _ => {}
                }
            }
        }
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Essential for developer workflow |
| Complexity | LOW - notify crate handles it |
| Confidence | HIGH |

### 4.3 Git Integration

**Description**: Index git repositories with commit-aware versioning.

```rust
struct GitIngester {
    store: Arc<Store>,
    chunker: Arc<Chunker>,
    embedder: Arc<Embedder>,
}

impl GitIngester {
    async fn ingest_repo(&self, repo_path: &Path, collection: &str) -> Result<()> {
        let repo = git2::Repository::open(repo_path)?;

        // Get current HEAD
        let head = repo.head()?.peel_to_commit()?;

        // Walk tree and ingest files
        let tree = head.tree()?;
        self.walk_tree(&repo, &tree, Path::new(""), collection).await
    }

    async fn ingest_diff(&self, old_commit: &str, new_commit: &str) -> Result<()> {
        // Only process changed files
        let diff = repo.diff_tree_to_tree(old_tree, new_tree, None)?;

        for delta in diff.deltas() {
            match delta.status() {
                Delta::Added | Delta::Modified => {
                    self.ingest_file(&delta.new_file().path()).await?;
                }
                Delta::Deleted => {
                    self.delete_file(&delta.old_file().path()).await?;
                }
                _ => {}
            }
        }
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Natural fit for code repos |
| Complexity | MEDIUM - git2 crate handles heavy lifting |
| Confidence | HIGH |

### 4.4 Structured Data Extraction

**Description**: Extract and index structured data from semi-structured documents.

```rust
struct StructuredExtractor {
    // Named entity recognition
    ner_model: Option<NerModel>,
    // Regex patterns for common entities
    patterns: Vec<(String, Regex)>,  // (entity_type, pattern)
}

impl StructuredExtractor {
    fn extract(&self, content: &str) -> ExtractedEntities {
        let mut entities = Vec::new();

        // Pattern-based extraction
        for (entity_type, pattern) in &self.patterns {
            for m in pattern.find_iter(content) {
                entities.push(Entity {
                    entity_type: entity_type.clone(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // NER-based extraction
        if let Some(ner) = &self.ner_model {
            entities.extend(ner.predict(content));
        }

        ExtractedEntities { entities }
    }
}
```

Extractable entities:
- Dates and times
- URLs and file paths
- Code references (function names, classes)
- Version numbers
- People and organizations
- Technical terms

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Enables structured queries |
| Complexity | MEDIUM - Regex is easy, NER adds complexity |
| Confidence | MEDIUM |

### 4.5 Document Summarization

**Description**: Generate and store summaries for documents to enable summary-level search.

```rust
struct DocumentSummarizer {
    llm: Arc<dyn LLM>,
    max_summary_tokens: usize,
}

impl DocumentSummarizer {
    async fn summarize(&self, content: &str) -> Result<Summary> {
        let prompt = format!(
            "Summarize this document in 2-3 sentences, focusing on the main \
             topics and key information:\n\n{}", content
        );

        let summary_text = self.llm.generate(&prompt).await?;

        Ok(Summary {
            text: summary_text,
            generated_at: Utc::now(),
        })
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Useful for long documents |
| Complexity | LOW - Simple LLM call |
| Confidence | HIGH |
| Cost | LLM API cost per document |

### 4.6 Link Extraction and Graph Building

**Description**: Extract links between documents and build a knowledge graph.

```rust
struct LinkExtractor {
    // Patterns for different link types
    markdown_link: Regex,      // [text](url)
    wiki_link: Regex,          // [[page]]
    code_import: Regex,        // import/require statements
    file_reference: Regex,     // file paths
}

struct KnowledgeGraph {
    // Document → linked documents
    edges: HashMap<Ulid, Vec<Link>>,
}

impl KnowledgeGraph {
    fn pagerank(&self) -> HashMap<Ulid, f32> {
        // Compute PageRank scores for documents
        // High-scoring documents are "more important"
    }

    fn related_documents(&self, doc_id: Ulid, depth: usize) -> Vec<Ulid> {
        // BFS traversal to find related documents
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Useful for navigation and ranking |
| Complexity | MEDIUM - Graph algorithms are standard |
| Confidence | HIGH |

---

## Query Intelligence

Features for smarter query processing.

### 5.1 Query Classification

**Description**: Classify queries to route to optimal retrieval strategy.

```rust
enum QueryType {
    Factual,        // "What is X?" → prioritize precision
    Exploratory,    // "How does X work?" → prioritize recall
    Navigational,   // "Find the file X" → use keyword + metadata
    Comparative,    // "Difference between X and Y" → retrieve both
    Procedural,     // "How to do X" → prioritize step-by-step content
    Debugging,      // "Why does X fail?" → retrieve error-related content
}

struct QueryClassifier {
    model: ort::Session,  // Fine-tuned classifier
}

impl QueryClassifier {
    fn classify(&self, query: &str) -> QueryType {
        // Run classification model
        let logits = self.model.run(query)?;
        QueryType::from_logits(logits)
    }
}

// Use classification to adjust search parameters
fn search_with_classification(query: &str) -> SearchConfig {
    match classifier.classify(query) {
        QueryType::Factual => SearchConfig {
            vector_weight: 0.7,
            keyword_weight: 0.3,
            top_k: 5,
        },
        QueryType::Exploratory => SearchConfig {
            vector_weight: 0.5,
            keyword_weight: 0.5,
            top_k: 15,
        },
        // ...
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Improves results for varied queries |
| Complexity | MEDIUM - Needs training data |
| Confidence | MEDIUM |

### 5.2 Query History and Analytics

**Description**: Track queries and analyze patterns to improve system.

```rust
struct QueryAnalytics {
    store: Arc<AnalyticsStore>,
}

impl QueryAnalytics {
    fn record(&self, query: QueryRecord) {
        self.store.insert(query);
    }

    fn top_queries(&self, period: Duration) -> Vec<(String, u64)> {
        // Most frequent queries
    }

    fn failed_queries(&self) -> Vec<QueryRecord> {
        // Queries with no results or low scores
    }

    fn query_suggestions(&self, partial: &str) -> Vec<String> {
        // Autocomplete from history
    }

    fn coverage_analysis(&self) -> CoverageReport {
        // Which topics are well-covered vs gaps
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Essential for system improvement |
| Complexity | LOW - Basic analytics |
| Confidence | HIGH |

### 5.3 Conversational Context

**Description**: Maintain conversation context for follow-up queries.

```rust
struct ConversationContext {
    history: Vec<(String, Vec<Ulid>)>,  // (query, result_ids)
    entities_mentioned: HashSet<String>,
    current_topic: Option<String>,
}

impl ConversationContext {
    fn expand_query(&self, query: &str) -> String {
        // Expand pronouns and references using context
        // "What about the second one?" → "What about [chunk from previous results]?"

        // If query is short and references previous results
        if query.len() < 50 && self.contains_reference(query) {
            // Inject context from previous results
            format!("{} (in context of: {})", query, self.current_topic.unwrap_or_default())
        } else {
            query.to_string()
        }
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Essential for chat interfaces |
| Complexity | MEDIUM - Reference resolution is tricky |
| Confidence | MEDIUM |

### 5.4 Semantic Caching

**Description**: Cache semantically similar queries to avoid redundant computation.

```rust
struct SemanticCache {
    cache: HashMap<Vec<u8>, CachedResult>,  // quantized embedding → result
    threshold: f32,  // similarity threshold for cache hit
}

impl SemanticCache {
    fn get(&self, query_embedding: &[f32]) -> Option<&CachedResult> {
        let quantized = quantize(query_embedding);

        // Check exact match first
        if let Some(result) = self.cache.get(&quantized) {
            return Some(result);
        }

        // Check for similar queries
        for (cached_emb, result) in &self.cache {
            let sim = cosine_similarity(query_embedding, &dequantize(cached_emb));
            if sim > self.threshold {
                return Some(result);
            }
        }

        None
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Reduces latency for repeated queries |
| Complexity | LOW - Standard caching |
| Confidence | HIGH |

---

## Observability & Debugging

Features for understanding system behavior.

### 6.1 Query Tracing

**Description**: Detailed traces of query execution for debugging.

```rust
struct QueryTrace {
    query_id: Ulid,
    query: String,
    stages: Vec<TraceStage>,
    total_duration: Duration,
}

struct TraceStage {
    name: String,
    duration: Duration,
    details: Value,  // Stage-specific data
}

// Example trace:
// QueryTrace {
//     query: "error handling",
//     stages: [
//         TraceStage { name: "embed_query", duration: 15ms, details: { tokens: 3 } },
//         TraceStage { name: "vector_search", duration: 23ms, details: { candidates: 50 } },
//         TraceStage { name: "keyword_search", duration: 12ms, details: { candidates: 50 } },
//         TraceStage { name: "rrf_fusion", duration: 1ms, details: { merged: 67 } },
//         TraceStage { name: "fetch_chunks", duration: 8ms, details: { fetched: 10 } },
//         TraceStage { name: "context_expand", duration: 5ms, details: { expanded: 14 } },
//     ],
//     total_duration: 64ms,
// }
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Essential for debugging |
| Complexity | LOW - Just instrumentation |
| Confidence | HIGH |

### 6.2 Retrieval Quality Metrics

**Description**: Compute and track retrieval quality metrics.

```rust
struct RetrievalMetrics {
    // Requires ground truth labels
    precision_at_k: HashMap<usize, f32>,
    recall_at_k: HashMap<usize, f32>,
    ndcg: f32,
    mrr: f32,  // Mean Reciprocal Rank

    // Doesn't require labels (proxy metrics)
    avg_similarity: f32,
    result_diversity: f32,  // Inverse of avg pairwise similarity
    coverage: f32,  // Fraction of collection represented in top results
}

impl RetrievalMetrics {
    fn evaluate(results: &[SearchResult], relevance_labels: &[f32]) -> Self {
        // Compute standard IR metrics
    }

    fn evaluate_unsupervised(results: &[SearchResult]) -> Self {
        // Compute proxy metrics that don't need labels
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Needed for evaluation |
| Complexity | LOW - Standard formulas |
| Confidence | HIGH |

### 6.3 Embedding Visualization

**Description**: Visualize embedding space for debugging and exploration.

Already specified in TUI doc. **Confidence: MEDIUM**

### 6.4 Chunk Quality Analysis

**Description**: Analyze and report on chunk quality issues.

```rust
struct ChunkAnalyzer {
    min_tokens: usize,
    max_tokens: usize,
}

impl ChunkAnalyzer {
    fn analyze(&self, chunks: &[Chunk]) -> ChunkAnalysis {
        let mut issues = Vec::new();

        for chunk in chunks {
            // Too short (may lack context)
            if chunk.token_count < self.min_tokens {
                issues.push(Issue::TooShort(chunk.id));
            }

            // Too long (may exceed model context)
            if chunk.token_count > self.max_tokens {
                issues.push(Issue::TooLong(chunk.id));
            }

            // Truncated mid-sentence
            if !chunk.content.ends_with(['.', '!', '?', '}', ']']) {
                issues.push(Issue::TruncatedContent(chunk.id));
            }

            // High overlap with adjacent chunks
            // ... compare with neighbors
        }

        ChunkAnalysis {
            total_chunks: chunks.len(),
            avg_tokens: chunks.iter().map(|c| c.token_count).sum::<u32>() / chunks.len() as u32,
            issues,
            token_distribution: compute_histogram(chunks),
        }
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Helps optimize chunking |
| Complexity | LOW - Simple analysis |
| Confidence | HIGH |

---

## Multi-Modal Support

Features for handling non-text content.

### 7.1 Image Support

**Description**: Index images with vision embeddings.

```rust
struct ImageEmbedder {
    model: ort::Session,  // CLIP or SigLIP
    dimension: usize,
}

impl ImageEmbedder {
    fn embed_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        // Preprocess image
        let tensor = preprocess_image(image)?;

        // Run through vision encoder
        let embedding = self.model.run(tensor)?;

        Ok(embedding)
    }
}

// Search images with text query
fn search_images(query: &str) -> Vec<ImageResult> {
    // Use CLIP text encoder for query
    let query_embedding = clip_text_encoder.embed(query)?;

    // Search image embeddings
    image_store.vector_search(&query_embedding, k)
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Useful for docs with images |
| Complexity | MEDIUM - Requires CLIP model |
| Confidence | MEDIUM |

### 7.2 PDF Processing

**Description**: Extract and index content from PDFs.

```rust
struct PdfProcessor {
    text_extractor: PdfTextExtractor,  // pdfium or poppler
    ocr_engine: Option<OcrEngine>,     // tesseract for scanned PDFs
    image_extractor: PdfImageExtractor,
}

impl PdfProcessor {
    fn process(&self, pdf_bytes: &[u8]) -> ProcessedPdf {
        let doc = self.text_extractor.open(pdf_bytes)?;

        let mut pages = Vec::new();
        for page in doc.pages() {
            let text = page.extract_text();

            // If text is empty, try OCR
            let text = if text.is_empty() && self.ocr_engine.is_some() {
                let image = page.render_to_image()?;
                self.ocr_engine.as_ref().unwrap().recognize(&image)?
            } else {
                text
            };

            // Extract images for multi-modal indexing
            let images = page.extract_images();

            pages.push(PageContent { text, images });
        }

        ProcessedPdf { pages }
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - PDFs are common |
| Complexity | HIGH - PDF parsing is complex |
| Confidence | MEDIUM |

### 7.3 Table Extraction

**Description**: Extract and index structured tables from documents.

```rust
struct TableExtractor {
    // Detect tables in markdown
    markdown_table: Regex,
    // Detect tables in HTML
    html_table_parser: HtmlTableParser,
}

impl TableExtractor {
    fn extract(&self, content: &str) -> Vec<Table> {
        // ... detect and parse tables

        // For each table, create multiple representations:
        // 1. Raw table (for exact matching)
        // 2. Row-by-row chunks (for specific data)
        // 3. Natural language summary (for semantic search)
    }
}

struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    natural_language_summary: String,  // "This table shows X with columns A, B, C"
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Tables contain dense info |
| Complexity | MEDIUM - Table detection is tricky |
| Confidence | MEDIUM |

---

## Integration Features

Features for integrating with external systems.

### 8.1 IDE Integration

**Description**: VSCode/JetBrains plugins for in-editor search.

```typescript
// VSCode extension pseudo-code
class RagCodeLensProvider implements vscode.CodeLensProvider {
    async provideCodeLenses(document: vscode.TextDocument): Promise<vscode.CodeLens[]> {
        const symbols = await vscode.commands.executeCommand(
            'vscode.executeDocumentSymbolProvider', document.uri
        );

        return symbols.map(symbol => new vscode.CodeLens(
            symbol.range,
            {
                title: "Search related code",
                command: "rag.searchRelated",
                arguments: [symbol.name]
            }
        ));
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Developer productivity |
| Complexity | MEDIUM - Platform-specific |
| Confidence | HIGH |

### 8.2 Webhook Notifications

**Description**: Send webhooks on events (ingest complete, sync, errors).

```rust
struct WebhookManager {
    endpoints: Vec<WebhookEndpoint>,
    client: reqwest::Client,
}

impl WebhookManager {
    async fn notify(&self, event: Event) {
        for endpoint in &self.endpoints {
            if endpoint.subscribed_to(&event) {
                let payload = event.to_json();
                self.client.post(&endpoint.url)
                    .json(&payload)
                    .send()
                    .await
                    .ok();  // Best-effort delivery
            }
        }
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Useful for automation |
| Complexity | LOW - Simple HTTP calls |
| Confidence | HIGH |

### 8.3 REST API

**Description**: HTTP API in addition to MCP for broader integration.

```yaml
openapi: 3.0.0
info:
  title: RAG API
  version: 1.0.0

paths:
  /v1/search:
    post:
      summary: Search the knowledge base
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                query: { type: string }
                collection: { type: string }
                top_k: { type: integer }
      responses:
        '200':
          description: Search results
          content:
            application/json:
              schema:
                type: object
                properties:
                  results: { type: array }

  /v1/documents:
    post:
      summary: Ingest a document
    get:
      summary: List documents

  /v1/documents/{id}:
    get:
      summary: Get document details
    delete:
      summary: Delete document

  /v1/collections:
    get:
      summary: List collections
    post:
      summary: Create collection
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Universal integration |
| Complexity | LOW - Standard REST |
| Confidence | HIGH |

### 8.4 Export/Import

**Description**: Export and import knowledge bases for backup/migration.

```rust
struct Exporter {
    store: Arc<Store>,
}

impl Exporter {
    async fn export(&self, path: &Path, format: ExportFormat) -> Result<()> {
        match format {
            ExportFormat::SqliteDump => {
                // Raw SQLite dump (fastest, largest)
                self.store.backup(path)?;
            }
            ExportFormat::Json => {
                // JSON export (portable, human-readable)
                let data = ExportData {
                    collections: self.store.list_collections().await?,
                    documents: self.store.list_all_documents().await?,
                    // Note: embeddings are large, optionally exclude
                };
                std::fs::write(path, serde_json::to_string(&data)?)?;
            }
            ExportFormat::Parquet => {
                // Parquet export (efficient, analytics-friendly)
                // ... write to parquet using arrow-rs
            }
        }
        Ok(())
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Essential for production |
| Complexity | LOW - Serialization |
| Confidence | HIGH |

---

## Performance Optimizations

Features for improved speed and efficiency.

### 9.1 Embedding Quantization

**Description**: Reduce embedding storage and improve search speed with quantization.

```rust
enum QuantizationType {
    Float32,     // 4 bytes/dim - baseline
    Float16,     // 2 bytes/dim - 50% reduction, minimal quality loss
    Int8,        // 1 byte/dim - 75% reduction, small quality loss
    Binary,      // 1 bit/dim - 96% reduction, larger quality loss
}

struct QuantizedStore {
    quantization: QuantizationType,
}

impl QuantizedStore {
    fn store(&self, embedding: Vec<f32>) -> Vec<u8> {
        match self.quantization {
            QuantizationType::Int8 => {
                // Scale to [-128, 127] range
                embedding.iter()
                    .map(|v| (v * 127.0).clamp(-128.0, 127.0) as i8)
                    .collect()
            }
            QuantizationType::Binary => {
                // Sign bit only
                embedding.chunks(8)
                    .map(|chunk| {
                        chunk.iter().enumerate()
                            .fold(0u8, |acc, (i, v)| acc | ((*v > 0.0) as u8) << i)
                    })
                    .collect()
            }
            // ...
        }
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Major storage/speed improvement |
| Complexity | LOW - Simple math |
| Confidence | HIGH |

### 9.2 Index Sharding

**Description**: Split index across multiple SQLite files for parallelism.

```rust
struct ShardedStore {
    shards: Vec<Arc<Store>>,
    shard_count: usize,
}

impl ShardedStore {
    fn shard_for(&self, id: &Ulid) -> &Arc<Store> {
        let hash = id.as_bytes()[0] as usize;  // Simple hash
        &self.shards[hash % self.shard_count]
    }

    async fn vector_search(&self, query: &[f32], k: usize) -> Vec<(Ulid, f32)> {
        // Search all shards in parallel
        let futures: Vec<_> = self.shards.iter()
            .map(|shard| shard.vector_search(query, k))
            .collect();

        let results = futures::future::join_all(futures).await;

        // Merge and take top-k
        let mut all: Vec<_> = results.into_iter().flatten().collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all.truncate(k);
        all
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Needed for very large indices |
| Complexity | MEDIUM - Adds operational complexity |
| Confidence | MEDIUM |

### 9.3 Async Ingestion Pipeline

**Description**: Fully async ingestion with backpressure.

```rust
struct AsyncIngester {
    load_buffer: mpsc::Sender<LoadJob>,
    chunk_buffer: mpsc::Sender<ChunkJob>,
    embed_buffer: mpsc::Sender<EmbedJob>,
    store_buffer: mpsc::Sender<StoreJob>,
}

impl AsyncIngester {
    async fn spawn_pipeline(&self) {
        // Stage 1: Loading (parallelism = 4)
        for _ in 0..4 {
            tokio::spawn(async move {
                while let Some(job) = load_rx.recv().await {
                    let content = load_file(&job.path).await?;
                    chunk_tx.send(ChunkJob { content, doc: job.doc }).await?;
                }
            });
        }

        // Stage 2: Chunking (parallelism = 2)
        // ...

        // Stage 3: Embedding (parallelism = 1, batched)
        // ...

        // Stage 4: Storing (parallelism = 1)
        // ...
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Much better throughput |
| Complexity | MEDIUM - Standard async patterns |
| Confidence | HIGH |

### 9.4 Lazy Loading

**Description**: Load chunk content on demand rather than at search time.

```rust
struct LazyChunk {
    id: Ulid,
    doc_id: Ulid,
    chunk_index: u32,
    token_count: u32,
    // Content is loaded on demand
    content: OnceCell<String>,
}

impl LazyChunk {
    async fn content(&self, store: &Store) -> Result<&str> {
        self.content.get_or_try_init(|| async {
            store.get_chunk_content(self.id).await
        }).await.map(|s| s.as_str())
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Reduces memory for large result sets |
| Complexity | LOW - Standard pattern |
| Confidence | HIGH |

---

## Security & Access Control

Features for securing the system.

### 10.1 Collection-Level Access Control

**Description**: Restrict access to collections by user/role.

```rust
struct AccessControl {
    rules: Vec<AccessRule>,
}

struct AccessRule {
    principal: Principal,    // User or Role
    collection: String,      // Collection name or "*"
    permissions: Permissions,
}

bitflags! {
    struct Permissions: u32 {
        const READ = 0b001;
        const WRITE = 0b010;
        const DELETE = 0b100;
        const ALL = Self::READ.bits | Self::WRITE.bits | Self::DELETE.bits;
    }
}

impl AccessControl {
    fn check(&self, principal: &Principal, collection: &str, required: Permissions) -> bool {
        self.rules.iter()
            .filter(|r| r.principal == *principal || r.principal.is_role_of(principal))
            .filter(|r| r.collection == "*" || r.collection == collection)
            .any(|r| r.permissions.contains(required))
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Needed for multi-user |
| Complexity | MEDIUM - Standard RBAC |
| Confidence | HIGH |

### 10.2 Audit Logging

**Description**: Log all access for compliance and debugging.

```rust
struct AuditLog {
    writer: Arc<dyn AuditWriter>,
}

struct AuditEntry {
    timestamp: DateTime<Utc>,
    principal: String,
    action: String,
    resource: String,
    outcome: Outcome,
    metadata: Value,
}

impl AuditLog {
    fn log(&self, entry: AuditEntry) {
        self.writer.write(entry);
    }
}

// Integration with store operations
impl Store {
    async fn search(&self, query: &str, ctx: &RequestContext) -> Result<Vec<SearchResult>> {
        let result = self.inner_search(query).await;

        self.audit.log(AuditEntry {
            principal: ctx.principal.clone(),
            action: "search".to_string(),
            resource: format!("query:{}", query),
            outcome: if result.is_ok() { Outcome::Success } else { Outcome::Failure },
            ..Default::default()
        });

        result
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Compliance requirement |
| Complexity | LOW - Simple logging |
| Confidence | HIGH |

### 10.3 Encryption at Rest

**Description**: Encrypt stored content and embeddings.

```rust
struct EncryptedStore {
    inner: Arc<Store>,
    cipher: Aes256Gcm,
    key: Key,
}

impl EncryptedStore {
    fn encrypt(&self, data: &[u8]) -> Vec<u8> {
        let nonce = generate_nonce();
        let ciphertext = self.cipher.encrypt(&nonce, data)?;
        [nonce.as_slice(), ciphertext.as_slice()].concat()
    }

    fn decrypt(&self, data: &[u8]) -> Vec<u8> {
        let (nonce, ciphertext) = data.split_at(12);
        self.cipher.decrypt(nonce.into(), ciphertext)?
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Depends on deployment |
| Complexity | LOW - Standard encryption |
| Confidence | HIGH |

---

## Developer Experience

Features for easier development and debugging.

### 11.1 REPL Mode

**Description**: Interactive shell for exploring and debugging.

```
$ rag-cli repl
rag> .collections
code (1,234 docs)
docs (456 docs)

rag> .search "error handling" --collection code --limit 3
[1] src/query.rs:142-168 (0.847)
    async fn handle_query_error(&self, err: QueryError) -> Result<()> {
    ...

rag> .inspect 01HRE4KXQN...
Chunk ID: 01HRE4KXQN...
Document: src/query.rs
Tokens: 312
Embedding: [0.023, -0.156, ...]

rag> .explain "error handling"
Query Execution Trace:
  embed_query: 15ms
  vector_search: 23ms (50 candidates)
  keyword_search: 12ms (50 candidates)
  rrf_fusion: 1ms (67 unique)
  fetch_chunks: 8ms (10 results)
Total: 59ms

rag> .similar 01HRE4KXQN...
Most similar chunks to "handle_query_error":
  [0.92] src/query.rs:170-195 (execute_search)
  [0.87] src/error.rs:45-78 (QueryError::from)
  [0.84] src/store.rs:234-267 (handle_store_error)
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Essential for debugging |
| Complexity | MEDIUM - Needs good UX |
| Confidence | HIGH |

### 11.2 Test Fixtures

**Description**: Built-in test data and fixture generation.

```rust
struct TestFixtures;

impl TestFixtures {
    fn sample_documents() -> Vec<Document> {
        vec![
            Document::code("fn main() { println!(\"Hello\"); }", "rust"),
            Document::markdown("# Title\n\nSome content here."),
            Document::chat("User: Hello\nAssistant: Hi there!"),
        ]
    }

    fn random_documents(count: usize, seed: u64) -> Vec<Document> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..count).map(|_| Document::random(&mut rng)).collect()
    }

    fn wikipedia_sample() -> Vec<Document> {
        // Small curated set of Wikipedia articles for testing
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Speeds up development |
| Complexity | LOW - Data generation |
| Confidence | HIGH |

### 11.3 Benchmarking Suite

**Description**: Built-in benchmarks for performance testing.

```rust
struct Benchmark {
    name: String,
    queries: Vec<String>,
    expected_results: HashMap<String, Vec<Ulid>>,  // For accuracy
}

impl Benchmark {
    fn run(&self, engine: &QueryEngine) -> BenchmarkResult {
        let mut latencies = Vec::new();
        let mut accuracies = Vec::new();

        for query in &self.queries {
            let start = Instant::now();
            let results = engine.search(query).await?;
            latencies.push(start.elapsed());

            if let Some(expected) = self.expected_results.get(query) {
                let accuracy = compute_accuracy(&results, expected);
                accuracies.push(accuracy);
            }
        }

        BenchmarkResult {
            name: self.name.clone(),
            p50_latency: percentile(&latencies, 50),
            p99_latency: percentile(&latencies, 99),
            avg_accuracy: accuracies.iter().sum::<f32>() / accuracies.len() as f32,
            throughput: self.queries.len() as f32 / latencies.iter().sum::<Duration>().as_secs_f32(),
        }
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Essential for optimization |
| Complexity | LOW - Standard benchmarking |
| Confidence | HIGH |

---

## AI-Assisted Features

Features that leverage LLMs for improved functionality.

### 12.1 Smart Ingestion Recommendations

**Description**: Suggest which files/directories to ingest based on project structure.

```rust
async fn recommend_ingestion(project_root: &Path, llm: &dyn LLM) -> Vec<Recommendation> {
    let file_tree = scan_directory(project_root)?;

    let prompt = format!(
        "Given this project structure, recommend which files and directories \
         should be indexed in a RAG system for code search. Consider importance, \
         relevance, and exclude generated/vendored files.\n\nStructure:\n{}",
        file_tree
    );

    let response = llm.generate(&prompt).await?;
    parse_recommendations(&response)
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Nice to have |
| Complexity | LOW - Simple LLM call |
| Confidence | HIGH |

### 12.2 Answer Generation

**Description**: Generate answers from retrieved context (full RAG loop).

```rust
struct AnswerGenerator {
    retriever: Arc<QueryEngine>,
    llm: Arc<dyn LLM>,
}

impl AnswerGenerator {
    async fn answer(&self, question: &str) -> Answer {
        // Retrieve relevant context
        let results = self.retriever.search(question).await?;

        // Format context for LLM
        let context = results.iter()
            .map(|r| format!("Source: {}\n{}", r.source, r.content))
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        // Generate answer
        let prompt = format!(
            "Answer this question using only the provided context. \
             If the context doesn't contain the answer, say so.\n\n\
             Context:\n{}\n\nQuestion: {}",
            context, question
        );

        let answer_text = self.llm.generate(&prompt).await?;

        Answer {
            text: answer_text,
            sources: results.iter().map(|r| r.source.clone()).collect(),
        }
    }
}
```

| Aspect | Assessment |
|--------|------------|
| Value | HIGH - Complete RAG experience |
| Complexity | LOW - Standard RAG pattern |
| Confidence | HIGH |

### 12.3 Query Suggestions

**Description**: Suggest related or follow-up queries.

```rust
async fn suggest_queries(query: &str, results: &[SearchResult], llm: &dyn LLM) -> Vec<String> {
    let context = results.iter()
        .take(3)
        .map(|r| r.content.chars().take(200).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "Given this search query and results, suggest 3 related follow-up queries \
         that would help the user explore this topic further.\n\n\
         Query: {}\n\nResults:\n{}",
        query, context
    );

    let response = llm.generate(&prompt).await?;
    parse_query_suggestions(&response)
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Improves exploration |
| Complexity | LOW - Simple LLM call |
| Confidence | HIGH |

### 12.4 Automatic Tagging

**Description**: Automatically generate tags/categories for documents.

```rust
async fn auto_tag(content: &str, llm: &dyn LLM) -> Vec<String> {
    let prompt = format!(
        "Generate 3-5 short tags/categories for this content. \
         Output as comma-separated values.\n\nContent:\n{}",
        content.chars().take(2000).collect::<String>()
    );

    let response = llm.generate(&prompt).await?;
    response.split(',').map(|s| s.trim().to_string()).collect()
}
```

| Aspect | Assessment |
|--------|------------|
| Value | MEDIUM - Improves organization |
| Complexity | LOW - Simple LLM call |
| Confidence | HIGH |

---

## Feature Priority Matrix

| Feature | Value | Complexity | Priority |
|---------|-------|------------|----------|
| **Core MVP** | | | |
| Basic search | HIGH | LOW | P0 |
| Basic ingest | HIGH | LOW | P0 |
| MCP interface | HIGH | LOW | P0 |
| TUI | HIGH | MEDIUM | P0 |
| **High Value** | | | |
| Reranking | HIGH | MEDIUM | P1 |
| File watching | HIGH | LOW | P1 |
| Query tracing | HIGH | LOW | P1 |
| REPL mode | HIGH | MEDIUM | P1 |
| REST API | HIGH | LOW | P1 |
| **Medium Value** | | | |
| Query expansion | HIGH | MEDIUM | P2 |
| Git integration | HIGH | MEDIUM | P2 |
| PDF processing | HIGH | HIGH | P2 |
| Answer generation | HIGH | LOW | P2 |
| Access control | HIGH | MEDIUM | P2 |
| **Nice to Have** | | | |
| HyDE | MEDIUM | MEDIUM | P3 |
| Image support | MEDIUM | MEDIUM | P3 |
| Embedding visualization | MEDIUM | MEDIUM | P3 |
| Semantic caching | MEDIUM | LOW | P3 |
| Auto-tagging | MEDIUM | LOW | P3 |

---

## Confidence Summary

| Category | Avg Confidence | Notes |
|----------|----------------|-------|
| Core retrieval | HIGH | Well-understood techniques |
| Chunking | MEDIUM-HIGH | AST is proven, LLM-based is newer |
| Performance | MEDIUM | Estimates need validation |
| Multi-modal | MEDIUM | CLIP is proven, integration complexity |
| AI-assisted | HIGH | Standard LLM patterns |
| Security | HIGH | Standard patterns |
