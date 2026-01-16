//! Query engine for hybrid search.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, info};
use ulid::Ulid;

use rag_core::{
    Embedder, Result, SearchResult, SearchResults, Store,
};

use crate::fusion::reciprocal_rank_fusion;

/// Configuration for search queries.
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Maximum number of results to return.
    pub top_k: u32,

    /// Weight for vector search (0.0 to 1.0).
    pub vector_weight: f32,

    /// Weight for keyword search (0.0 to 1.0).
    pub keyword_weight: f32,

    /// Whether to expand context by including adjacent chunks.
    pub expand_context: bool,

    /// Number of adjacent chunks to include when expanding.
    pub context_chunks: u32,

    /// Collection to search (None for all collections).
    pub collection: Option<String>,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            vector_weight: 0.7,
            keyword_weight: 0.3,
            expand_context: true,
            context_chunks: 1,
            collection: None,
        }
    }
}

/// Hybrid search query engine.
///
/// Performs vector similarity search and keyword search, then fuses
/// the results using Reciprocal Rank Fusion (RRF).
pub struct QueryEngine<S, E> {
    /// Storage backend.
    store: Arc<S>,

    /// Embedding model.
    embedder: Arc<E>,
}

impl<S, E> QueryEngine<S, E>
where
    S: Store + Send + Sync,
    E: Embedder + Send + Sync,
{
    /// Create a new query engine.
    pub fn new(store: Arc<S>, embedder: Arc<E>) -> Self {
        Self { store, embedder }
    }

    /// Perform a hybrid search.
    pub async fn search(&self, query: &str, config: QueryConfig) -> Result<SearchResults> {
        let start = Instant::now();

        info!("Searching for: {:?}", query);

        // Embed the query
        let query_embedding = self.embedder.embed_query(query).await?;

        // Determine how many results to fetch (fetch more for fusion)
        let fetch_k = (config.top_k * 2).max(20);

        // Perform searches in parallel
        let (vector_results, keyword_results): (
            Result<Vec<(Ulid, f32)>>,
            Result<Vec<(Ulid, f32)>>,
        ) = tokio::join!(
            self.vector_search(&query_embedding, fetch_k, &config.collection),
            self.keyword_search(query, fetch_k, &config.collection)
        );

        let vector_results = vector_results?;
        let keyword_results = keyword_results?;

        debug!(
            "Vector search returned {} results, keyword search returned {} results",
            vector_results.len(),
            keyword_results.len()
        );

        // Fuse results using RRF
        let fused = reciprocal_rank_fusion(
            vec![vector_results, keyword_results],
            config.top_k as usize,
        );

        debug!("Fused to {} results", fused.len());

        // Build search results
        let mut results = Vec::with_capacity(fused.len());
        let mut seen_chunks: HashSet<Ulid> = HashSet::new();

        for (rank, (chunk_id, score)) in fused.iter().enumerate() {
            if seen_chunks.contains(chunk_id) {
                continue;
            }
            seen_chunks.insert(*chunk_id);

            // Fetch the chunk
            let chunk = match self.store.get_chunk(*chunk_id).await? {
                Some(c) => c,
                None => continue,
            };

            // Fetch the document for metadata
            let doc = match self.store.get_document(chunk.doc_id).await? {
                Some(d) => d,
                None => continue,
            };

            results.push(SearchResult {
                rank: rank as u32 + 1,
                score: *score,
                chunk,
                source_uri: doc.source_uri,
                collection: doc.collection,
            });
        }

        // Optionally expand context
        if config.expand_context && config.context_chunks > 0 {
            results = self
                .expand_context(results, config.context_chunks, &mut seen_chunks)
                .await?;
        }

        let latency_ms = start.elapsed().as_millis() as u64;

        info!(
            "Search completed in {}ms, returned {} results",
            latency_ms,
            results.len()
        );

        Ok(SearchResults {
            query: query.to_string(),
            total_results: results.len(),
            latency_ms,
            results,
        })
    }

    /// Perform vector similarity search.
    async fn vector_search(
        &self,
        embedding: &[f32],
        k: u32,
        collection: &Option<String>,
    ) -> Result<Vec<(Ulid, f32)>> {
        self.store
            .vector_search(embedding, k, collection.as_deref())
            .await
    }

    /// Perform keyword search.
    async fn keyword_search(
        &self,
        query: &str,
        k: u32,
        collection: &Option<String>,
    ) -> Result<Vec<(Ulid, f32)>> {
        self.store
            .keyword_search(query, k, collection.as_deref())
            .await
    }

    /// Expand results with adjacent chunks for more context.
    async fn expand_context(
        &self,
        results: Vec<SearchResult>,
        context_chunks: u32,
        seen: &mut HashSet<Ulid>,
    ) -> Result<Vec<SearchResult>> {
        let mut expanded = Vec::with_capacity(results.len() * 2);

        for result in results {
            let doc_chunks = self.store.get_chunks_for_document(result.chunk.doc_id).await?;

            // Find the index of the current chunk
            let current_idx = doc_chunks
                .iter()
                .position(|c| c.id == result.chunk.id)
                .unwrap_or(0);

            // Add previous chunks
            for i in 1..=context_chunks as usize {
                if current_idx >= i {
                    let prev_chunk = &doc_chunks[current_idx - i];
                    if !seen.contains(&prev_chunk.id) {
                        seen.insert(prev_chunk.id);
                        // Add with lower score to indicate context
                        expanded.push(SearchResult {
                            rank: 0, // Will be recalculated
                            score: result.score * 0.5,
                            chunk: prev_chunk.clone(),
                            source_uri: result.source_uri.clone(),
                            collection: result.collection.clone(),
                        });
                    }
                }
            }

            // Add the main chunk
            expanded.push(result.clone());

            // Add following chunks
            for i in 1..=context_chunks as usize {
                if current_idx + i < doc_chunks.len() {
                    let next_chunk = &doc_chunks[current_idx + i];
                    if !seen.contains(&next_chunk.id) {
                        seen.insert(next_chunk.id);
                        expanded.push(SearchResult {
                            rank: 0,
                            score: result.score * 0.5,
                            chunk: next_chunk.clone(),
                            source_uri: result.source_uri.clone(),
                            collection: result.collection.clone(),
                        });
                    }
                }
            }
        }

        // Sort by score and reassign ranks
        expanded.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, result) in expanded.iter_mut().enumerate() {
            result.rank = i as u32 + 1;
        }

        Ok(expanded)
    }

    /// Simple search without embedding (keyword only).
    pub async fn keyword_only_search(
        &self,
        query: &str,
        top_k: u32,
        collection: Option<&str>,
    ) -> Result<SearchResults> {
        let start = Instant::now();

        let results = self
            .store
            .keyword_search(query, top_k, collection)
            .await?;

        let mut search_results = Vec::with_capacity(results.len());

        for (rank, (chunk_id, score)) in results.iter().enumerate() {
            let chunk = match self.store.get_chunk(*chunk_id).await? {
                Some(c) => c,
                None => continue,
            };

            let doc = match self.store.get_document(chunk.doc_id).await? {
                Some(d) => d,
                None => continue,
            };

            search_results.push(SearchResult {
                rank: rank as u32 + 1,
                score: *score,
                chunk,
                source_uri: doc.source_uri,
                collection: doc.collection,
            });
        }

        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(SearchResults {
            query: query.to_string(),
            total_results: search_results.len(),
            latency_ms,
            results: search_results,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_config_default() {
        let config = QueryConfig::default();
        assert_eq!(config.top_k, 10);
        assert!(config.vector_weight > 0.0);
        assert!(config.keyword_weight > 0.0);
    }
}
