//! Jaccard-based virtual affinity edge generation.

use crate::graph::{EdgeInput, EdgeKind};

/// Configuration for virtual affinity edges.
#[derive(Debug, Clone)]
pub struct AffinityConfig {
    /// Maximum weight for virtual edges. Default: 0.4.
    pub max_weight: f64,
    /// Minimum Jaccard similarity to create edge. Default: 0.3.
    pub min_jaccard: f64,
}

impl Default for AffinityConfig {
    fn default() -> Self {
        AffinityConfig {
            max_weight: 0.4,
            min_jaccard: 0.3,
        }
    }
}

/// Compute Jaccard similarity between two tag sets.
///
/// J(A, B) = |A ∩ B| / |A ∪ B|
pub fn jaccard_similarity(a: &[u32], b: &[u32]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }

    // Tags are represented as sorted tag IDs for O(n+m) intersection.
    let mut i = 0;
    let mut j = 0;
    let mut intersection = 0u32;
    let mut union_size = 0u32;

    // Merge-style intersection on sorted slices.
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            intersection += 1;
            union_size += 1;
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            union_size += 1;
            i += 1;
        } else {
            union_size += 1;
            j += 1;
        }
    }
    union_size += (a.len() - i) as u32 + (b.len() - j) as u32;

    if union_size == 0 {
        return 0.0;
    }

    intersection as f64 / union_size as f64
}

/// Generate virtual affinity edges for all node pairs exceeding minJaccard.
///
/// `node_tags` maps node index to sorted tag IDs.
/// Returns edges to add to the graph (or overlay during propagation).
pub fn compute_affinity_edges(
    node_tags: &[Vec<u32>],
    config: &AffinityConfig,
) -> Vec<EdgeInput> {
    let n = node_tags.len();
    let mut edges = Vec::new();

    for i in 0..n {
        if node_tags[i].is_empty() {
            continue;
        }
        for j in (i + 1)..n {
            if node_tags[j].is_empty() {
                continue;
            }

            let j_sim = jaccard_similarity(&node_tags[i], &node_tags[j]);
            if j_sim >= config.min_jaccard {
                let weight = j_sim * config.max_weight;
                edges.push(EdgeInput {
                    source: i as u32,
                    target: j as u32,
                    weight,
                    kind: EdgeKind::FeatureAffinity,
                });
            }
        }
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_identical() {
        assert!((jaccard_similarity(&[1, 2, 3], &[1, 2, 3]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jaccard_disjoint() {
        assert!((jaccard_similarity(&[1, 2], &[3, 4])).abs() < 1e-10);
    }

    #[test]
    fn jaccard_partial() {
        // {1,2,3} ∩ {2,3,4} = {2,3}, union = {1,2,3,4}
        let j = jaccard_similarity(&[1, 2, 3], &[2, 3, 4]);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn jaccard_empty() {
        assert_eq!(jaccard_similarity(&[], &[]), 0.0);
        assert_eq!(jaccard_similarity(&[1], &[]), 0.0);
    }

    #[test]
    fn affinity_edges_above_threshold() {
        let tags = vec![
            vec![1, 2, 3],    // node 0
            vec![2, 3, 4],    // node 1 — J(0,1)=0.5 >= 0.3 ✓
            vec![10, 11, 12], // node 2 — J(0,2)=0.0 < 0.3 ✗
        ];
        let cfg = AffinityConfig::default();
        let edges = compute_affinity_edges(&tags, &cfg);

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, 0);
        assert_eq!(edges[0].target, 1);
        assert!((edges[0].weight - 0.5 * 0.4).abs() < 1e-10);
    }
}
