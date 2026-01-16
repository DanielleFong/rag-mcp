//! Reciprocal Rank Fusion (RRF) for combining search results.

use std::collections::HashMap;
use ulid::Ulid;

/// RRF constant (commonly 60).
/// Higher values give more weight to lower-ranked results.
const RRF_K: f32 = 60.0;

/// Fuse multiple result lists using Reciprocal Rank Fusion.
///
/// RRF score = Σ (1 / (k + rank_i)) for each result list
///
/// # Arguments
/// * `results` - Vector of result lists, each containing (id, original_score) pairs
/// * `k` - Maximum number of results to return
///
/// # Returns
/// Vector of (id, fused_score) pairs, sorted by fused score descending
pub fn reciprocal_rank_fusion(
    results: Vec<Vec<(Ulid, f32)>>,
    k: usize,
) -> Vec<(Ulid, f32)> {
    let mut scores: HashMap<Ulid, f32> = HashMap::new();

    // Calculate RRF scores
    for result_list in results {
        for (rank, (id, _original_score)) in result_list.into_iter().enumerate() {
            let rrf_score = 1.0 / (RRF_K + rank as f32 + 1.0);
            *scores.entry(id).or_default() += rrf_score;
        }
    }

    // Sort by score descending
    let mut fused: Vec<_> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top k
    fused.truncate(k);

    fused
}

/// Combine results using weighted fusion.
///
/// # Arguments
/// * `results` - Vector of (results, weight) pairs
/// * `k` - Maximum number of results to return
pub fn weighted_fusion(
    results: Vec<(Vec<(Ulid, f32)>, f32)>,
    k: usize,
) -> Vec<(Ulid, f32)> {
    let mut scores: HashMap<Ulid, f32> = HashMap::new();

    for (result_list, weight) in results {
        for (id, score) in result_list {
            *scores.entry(id).or_default() += score * weight;
        }
    }

    let mut fused: Vec<_> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(k);

    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ulid(s: &str) -> Ulid {
        // Create deterministic ULIDs from strings for testing
        let hash = s.bytes().fold(0u128, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u128));
        Ulid::from(hash)
    }

    #[test]
    fn test_rrf_single_list() {
        let results = vec![vec![
            (ulid("a"), 0.9),
            (ulid("b"), 0.8),
            (ulid("c"), 0.7),
        ]];

        let fused = reciprocal_rank_fusion(results, 10);

        assert_eq!(fused.len(), 3);
        // First result should have highest score
        assert_eq!(fused[0].0, ulid("a"));
    }

    #[test]
    fn test_rrf_multiple_lists() {
        let results = vec![
            vec![
                (ulid("a"), 0.9),
                (ulid("b"), 0.8),
                (ulid("c"), 0.7),
            ],
            vec![
                (ulid("b"), 0.95), // b is first in this list
                (ulid("a"), 0.85),
                (ulid("d"), 0.75),
            ],
        ];

        let fused = reciprocal_rank_fusion(results, 10);

        // b should likely be first because it appears high in both lists
        // a should be second
        assert!(fused.len() >= 3);
    }

    #[test]
    fn test_rrf_truncation() {
        let results = vec![vec![
            (ulid("a"), 0.9),
            (ulid("b"), 0.8),
            (ulid("c"), 0.7),
            (ulid("d"), 0.6),
            (ulid("e"), 0.5),
        ]];

        let fused = reciprocal_rank_fusion(results, 3);

        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn test_weighted_fusion() {
        let results = vec![
            (vec![(ulid("a"), 1.0), (ulid("b"), 0.5)], 0.7), // vector: weight 0.7
            (vec![(ulid("b"), 1.0), (ulid("c"), 0.5)], 0.3), // keyword: weight 0.3
        ];

        let fused = weighted_fusion(results, 10);

        assert!(!fused.is_empty());
    }
}
