# rag-query

Hybrid search engine with vector and keyword retrieval, RRF fusion, and context assembly.

## Overview

| Attribute | Value |
|-----------|-------|
| Purpose | Execute and rank search queries |
| Dependencies | rag-core, rag-store, rag-embed |
| Est. Lines | ~1,000 |
| Confidence | HIGH |

This crate orchestrates the search pipeline.

---

## Module Structure

```
rag-query/
├── src/
│   ├── lib.rs           # Public exports
│   ├── engine.rs        # QueryEngine implementation
│   ├── fusion.rs        # RRF and other fusion strategies
│   ├── context.rs       # Context expansion and assembly
│   ├── trace.rs         # Query tracing
│   └── config.rs        # Search configuration
└── Cargo.toml
```

---

## QueryEngine

```rust
use rag_core::{Chunk, Result, Store, Embedder};
use std::sync::Arc;
use std::collections::HashMap;

/// Main query engine for hybrid search.
pub struct QueryEngine {
    store: Arc<dyn Store>,
    embedder: Arc<dyn Embedder>,
    config: QueryConfig,
}

/// Configuration for query execution.
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Number of candidates from vector search.
    pub vector_k: usize,

    /// Number of candidates from keyword search.
    pub keyword_k: usize,

    /// RRF constant (typically 60).
    pub rrf_k: f32,

    /// Number of results to return.
    pub final_k: usize,

    /// Weight for vector search in hybrid (0.0 to 1.0).
    /// 1.0 = vector only, 0.0 = keyword only.
    pub hybrid_alpha: f32,

    /// Expand results with adjacent chunks.
    pub expand_context: bool,

    /// Maximum tokens in final context.
    pub max_context_tokens: usize,

    /// Enable query tracing.
    pub enable_tracing: bool,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            vector_k: 50,
            keyword_k: 50,
            rrf_k: 60.0,
            final_k: 10,
            hybrid_alpha: 0.5,
            expand_context: true,
            max_context_tokens: 4000,
            enable_tracing: false,
        }
    }
}

impl QueryEngine {
    pub fn new(
        store: Arc<dyn Store>,
        embedder: Arc<dyn Embedder>,
        config: QueryConfig,
    ) -> Self {
        Self { store, embedder, config }
    }

    /// Execute a search query.
    pub async fn search(
        &self,
        query: &str,
        collection: Option<&str>,
    ) -> Result<SearchResults> {
        self.search_with_config(query, collection, &self.config).await
    }

    /// Execute with custom config.
    pub async fn search_with_config(
        &self,
        query: &str,
        collection: Option<&str>,
        config: &QueryConfig,
    ) -> Result<SearchResults> {
        let mut trace = if config.enable_tracing {
            Some(QueryTrace::new(query))
        } else {
            None
        };

        // 1. Embed query
        let embed_start = std::time::Instant::now();
        let query_embedding = self.embedder.embed_query(query)?;
        if let Some(t) = &mut trace {
            t.add_stage("embed_query", embed_start.elapsed(), json!({
                "dimension": query_embedding.len(),
            }));
        }

        // 2. Parallel search
        let search_start = std::time::Instant::now();
        let (vector_results, keyword_results) = tokio::join!(
            self.store.vector_search(&query_embedding, config.vector_k, collection),
            self.store.keyword_search(query, config.keyword_k, collection),
        );

        let vector_results = vector_results?;
        let keyword_results = keyword_results?;

        if let Some(t) = &mut trace {
            t.add_stage("parallel_search", search_start.elapsed(), json!({
                "vector_candidates": vector_results.len(),
                "keyword_candidates": keyword_results.len(),
            }));
        }

        // 3. RRF fusion
        let fusion_start = std::time::Instant::now();
        let fused = self.reciprocal_rank_fusion(
            &vector_results,
            &keyword_results,
            config,
        );
        if let Some(t) = &mut trace {
            t.add_stage("rrf_fusion", fusion_start.elapsed(), json!({
                "unique_results": fused.len(),
            }));
        }

        // 4. Fetch chunk content
        let fetch_start = std::time::Instant::now();
        let mut results = Vec::new();
        for (chunk_id, score) in fused.into_iter().take(config.final_k) {
            if let Some(chunk) = self.store.get_chunk(chunk_id).await? {
                results.push(SearchResult {
                    chunk,
                    score,
                    is_context: false,
                });
            }
        }
        if let Some(t) = &mut trace {
            t.add_stage("fetch_chunks", fetch_start.elapsed(), json!({
                "fetched": results.len(),
            }));
        }

        // 5. Expand context
        if config.expand_context {
            let expand_start = std::time::Instant::now();
            results = self.expand_with_adjacent(results).await?;
            if let Some(t) = &mut trace {
                t.add_stage("expand_context", expand_start.elapsed(), json!({
                    "total_after_expand": results.len(),
                }));
            }
        }

        // 6. Deduplicate overlapping chunks
        let dedup_start = std::time::Instant::now();
        results = self.deduplicate_overlaps(results);
        if let Some(t) = &mut trace {
            t.add_stage("deduplicate", dedup_start.elapsed(), json!({
                "after_dedup": results.len(),
            }));
        }

        // 7. Truncate to token limit
        let truncate_start = std::time::Instant::now();
        results = self.truncate_to_limit(results, config.max_context_tokens);
        if let Some(t) = &mut trace {
            t.add_stage("truncate", truncate_start.elapsed(), json!({
                "final_count": results.len(),
            }));
        }

        Ok(SearchResults {
            query: query.to_string(),
            results,
            trace,
        })
    }
}

/// A single search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The chunk content and metadata.
    pub chunk: Chunk,

    /// Relevance score (higher is better).
    pub score: f32,

    /// True if this is expanded context (not a direct match).
    pub is_context: bool,
}

/// Collection of search results with metadata.
#[derive(Debug)]
pub struct SearchResults {
    /// Original query string.
    pub query: String,

    /// Ordered list of results (best first).
    pub results: Vec<SearchResult>,

    /// Execution trace (if enabled).
    pub trace: Option<QueryTrace>,
}

impl SearchResults {
    /// Get total token count across all results.
    pub fn total_tokens(&self) -> u32 {
        self.results.iter().map(|r| r.chunk.token_count).sum()
    }

    /// Format results as context string for LLM.
    pub fn as_context(&self) -> String {
        self.results.iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "[{}] Source: {}\n{}\n",
                    i + 1,
                    r.chunk.doc_id,
                    r.chunk.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n---\n\n")
    }
}
```

---

## Reciprocal Rank Fusion

```rust
impl QueryEngine {
    /// Combine vector and keyword results using RRF.
    ///
    /// RRF score = Σ (1 / (k + rank))
    ///
    /// Where k is typically 60, and rank starts at 1.
    fn reciprocal_rank_fusion(
        &self,
        vector: &[(Ulid, f32)],
        keyword: &[(Ulid, f32)],
        config: &QueryConfig,
    ) -> Vec<(Ulid, f32)> {
        let mut scores: HashMap<Ulid, f32> = HashMap::new();
        let k = config.rrf_k;
        let alpha = config.hybrid_alpha;

        // Vector scores (weighted by alpha)
        for (rank, (id, _distance)) in vector.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank as f32) + 1.0);
            *scores.entry(*id).or_default() += rrf_score * alpha;
        }

        // Keyword scores (weighted by 1-alpha)
        for (rank, (id, _bm25)) in keyword.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank as f32) + 1.0);
            *scores.entry(*id).or_default() += rrf_score * (1.0 - alpha);
        }

        // Sort by combined score (descending)
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
    }
}
```

### Alternative Fusion Strategies

```rust
/// Different fusion strategies.
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion.
    RRF { k: f32 },

    /// Weighted score combination.
    WeightedSum { vector_weight: f32, keyword_weight: f32 },

    /// Take top from each, interleave.
    Interleave,

    /// Vector only.
    VectorOnly,

    /// Keyword only.
    KeywordOnly,
}

impl QueryEngine {
    fn fuse(
        &self,
        vector: &[(Ulid, f32)],
        keyword: &[(Ulid, f32)],
        strategy: &FusionStrategy,
    ) -> Vec<(Ulid, f32)> {
        match strategy {
            FusionStrategy::RRF { k } => {
                self.rrf_fusion(vector, keyword, *k)
            }

            FusionStrategy::WeightedSum { vector_weight, keyword_weight } => {
                // Normalize scores to [0, 1] range first
                let v_normalized = normalize_scores(vector);
                let k_normalized = normalize_scores(keyword);

                let mut scores: HashMap<Ulid, f32> = HashMap::new();

                for (id, score) in v_normalized {
                    *scores.entry(id).or_default() += score * vector_weight;
                }
                for (id, score) in k_normalized {
                    *scores.entry(id).or_default() += score * keyword_weight;
                }

                let mut results: Vec<_> = scores.into_iter().collect();
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                results
            }

            FusionStrategy::Interleave => {
                let mut results = Vec::new();
                let max_len = vector.len().max(keyword.len());

                for i in 0..max_len {
                    if i < vector.len() {
                        results.push(vector[i]);
                    }
                    if i < keyword.len() {
                        results.push(keyword[i]);
                    }
                }

                // Deduplicate, keeping first occurrence
                let mut seen = HashSet::new();
                results.retain(|(id, _)| seen.insert(*id));

                results
            }

            FusionStrategy::VectorOnly => vector.to_vec(),
            FusionStrategy::KeywordOnly => keyword.to_vec(),
        }
    }
}

fn normalize_scores(results: &[(Ulid, f32)]) -> Vec<(Ulid, f32)> {
    if results.is_empty() {
        return Vec::new();
    }

    let min = results.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
    let max = results.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    if range == 0.0 {
        return results.iter().map(|(id, _)| (*id, 1.0)).collect();
    }

    results.iter()
        .map(|(id, score)| (*id, (score - min) / range))
        .collect()
}
```

---

## Context Expansion

```rust
impl QueryEngine {
    /// Expand results with adjacent chunks for better context.
    async fn expand_with_adjacent(
        &self,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>> {
        let mut expanded = Vec::new();

        for result in results {
            let doc_id = result.chunk.doc_id;
            let chunk_index = result.chunk.chunk_index;

            // Fetch previous chunk
            if chunk_index > 0 {
                if let Some(prev) = self.store.get_chunk_by_index(doc_id, chunk_index - 1).await? {
                    expanded.push(SearchResult {
                        chunk: prev,
                        score: result.score * 0.5,  // Reduced score for context
                        is_context: true,
                    });
                }
            }

            // Add the original result
            expanded.push(result.clone());

            // Fetch next chunk
            if let Some(next) = self.store.get_chunk_by_index(doc_id, chunk_index + 1).await? {
                expanded.push(SearchResult {
                    chunk: next,
                    score: result.score * 0.5,
                    is_context: true,
                });
            }
        }

        // Sort by (doc_id, chunk_index) to maintain document order
        expanded.sort_by(|a, b| {
            (a.chunk.doc_id, a.chunk.chunk_index)
                .cmp(&(b.chunk.doc_id, b.chunk.chunk_index))
        });

        Ok(expanded)
    }

    /// Remove chunks that overlap significantly.
    fn deduplicate_overlaps(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        let mut seen: HashMap<(Ulid, u32), bool> = HashMap::new();
        let mut deduped = Vec::new();

        for result in results {
            let key = (result.chunk.doc_id, result.chunk.chunk_index);
            if seen.insert(key, true).is_none() {
                deduped.push(result);
            }
        }

        deduped
    }

    /// Truncate results to fit within token limit.
    fn truncate_to_limit(
        &self,
        results: Vec<SearchResult>,
        max_tokens: usize,
    ) -> Vec<SearchResult> {
        let mut total_tokens = 0u32;
        let mut truncated = Vec::new();

        for result in results {
            if total_tokens + result.chunk.token_count <= max_tokens as u32 {
                total_tokens += result.chunk.token_count;
                truncated.push(result);
            } else {
                // Stop adding when limit reached
                break;
            }
        }

        truncated
    }
}
```

---

## Query Tracing

```rust
use serde::{Serialize, Deserialize};
use std::time::Duration;

/// Trace of query execution for debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTrace {
    pub query_id: Ulid,
    pub query: String,
    pub stages: Vec<TraceStage>,
    pub total_duration: Duration,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStage {
    pub name: String,
    pub duration: Duration,
    pub details: serde_json::Value,
}

impl QueryTrace {
    pub fn new(query: &str) -> Self {
        Self {
            query_id: Ulid::new(),
            query: query.to_string(),
            stages: Vec::new(),
            total_duration: Duration::ZERO,
            started_at: chrono::Utc::now(),
        }
    }

    pub fn add_stage(&mut self, name: &str, duration: Duration, details: serde_json::Value) {
        self.stages.push(TraceStage {
            name: name.to_string(),
            duration,
            details,
        });
        self.total_duration += duration;
    }

    pub fn summary(&self) -> String {
        let mut lines = vec![
            format!("Query: {}", self.query),
            format!("Total: {:?}", self.total_duration),
            "Stages:".to_string(),
        ];

        for stage in &self.stages {
            lines.push(format!(
                "  {}: {:?}",
                stage.name,
                stage.duration
            ));
        }

        lines.join("\n")
    }
}
```

---

## Specialized Query Types

```rust
impl QueryEngine {
    /// Vector-only search (semantic similarity).
    pub async fn vector_search(
        &self,
        query: &str,
        collection: Option<&str>,
        k: usize,
    ) -> Result<SearchResults> {
        let config = QueryConfig {
            vector_k: k,
            keyword_k: 0,
            final_k: k,
            hybrid_alpha: 1.0,  // Vector only
            expand_context: false,
            ..Default::default()
        };
        self.search_with_config(query, collection, &config).await
    }

    /// Keyword-only search (BM25).
    pub async fn keyword_search(
        &self,
        query: &str,
        collection: Option<&str>,
        k: usize,
    ) -> Result<SearchResults> {
        let config = QueryConfig {
            vector_k: 0,
            keyword_k: k,
            final_k: k,
            hybrid_alpha: 0.0,  // Keyword only
            expand_context: false,
            ..Default::default()
        };
        self.search_with_config(query, collection, &config).await
    }

    /// Find similar chunks to a given chunk.
    pub async fn find_similar(
        &self,
        chunk_id: Ulid,
        k: usize,
    ) -> Result<SearchResults> {
        // Get the chunk's embedding from the store
        let chunk = self.store.get_chunk(chunk_id).await?
            .ok_or(RagError::ChunkNotFound(chunk_id))?;

        // Embed the chunk content and search
        let embedding = self.embedder.embed(&[&chunk.content])?;
        let results = self.store.vector_search(&embedding[0], k + 1, None).await?;

        // Filter out the source chunk and fetch content
        let mut search_results = Vec::new();
        for (id, distance) in results {
            if id != chunk_id {
                if let Some(c) = self.store.get_chunk(id).await? {
                    search_results.push(SearchResult {
                        chunk: c,
                        score: 1.0 - distance,  // Convert distance to similarity
                        is_context: false,
                    });
                }
            }
        }

        Ok(SearchResults {
            query: format!("similar to chunk {}", chunk_id),
            results: search_results,
            trace: None,
        })
    }
}
```

---

## Performance Considerations

### Query Latency Breakdown (Typical)

| Stage | Typical Duration | Notes |
|-------|-----------------|-------|
| Embed query | 10-20ms | Single text, fast |
| Vector search | 20-50ms | Depends on index size |
| Keyword search | 10-30ms | FTS5 is fast |
| RRF fusion | <1ms | In-memory hash |
| Fetch chunks | 10-30ms | ~10 DB reads |
| Context expand | 5-15ms | ~20 additional reads |
| **Total** | **60-150ms** | Well under 1s target |

### Caching Opportunities

```rust
/// Optional query cache.
pub struct QueryCache {
    cache: moka::sync::Cache<String, SearchResults>,
    ttl: Duration,
}

impl QueryCache {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            cache: moka::sync::Cache::builder()
                .max_capacity(capacity as u64)
                .time_to_live(ttl)
                .build(),
            ttl,
        }
    }

    pub fn get(&self, query: &str, collection: Option<&str>) -> Option<SearchResults> {
        let key = format!("{}:{}", query, collection.unwrap_or("*"));
        self.cache.get(&key)
    }

    pub fn put(&self, query: &str, collection: Option<&str>, results: SearchResults) {
        let key = format!("{}:{}", query, collection.unwrap_or("*"));
        self.cache.insert(key, results);
    }
}
```

---

## Confidence Assessment

| Component | Confidence | Notes |
|-----------|------------|-------|
| RRF implementation | HIGH | Standard algorithm |
| Parallel search | HIGH | Simple tokio::join! |
| Context expansion | HIGH | Simple DB reads |
| Deduplication | HIGH | Hash-based |
| Tracing | HIGH | Standard instrumentation |
| Cache integration | HIGH | moka is mature |
| Performance estimates | MEDIUM | Depends on data size |

---

## Cargo.toml

```toml
[package]
name = "rag-query"
version = "0.1.0"
edition = "2021"
description = "Query engine for the RAG system"
license = "MIT"

[dependencies]
rag-core = { path = "../rag-core" }
tokio = { version = "1.0", features = ["sync", "macros"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
tracing = "0.1"
moka = { version = "0.12", features = ["sync"], optional = true }

[features]
default = []
cache = ["moka"]

[dev-dependencies]
tokio = { version = "1.0", features = ["macros", "rt-multi-thread"] }
```
