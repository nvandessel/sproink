//! Tag-based affinity edge generation using Jaccard similarity.
//!
//! Given per-node tag sets, generates [`FeatureAffinity`](crate::EdgeKind::FeatureAffinity)
//! edges between nodes whose tags overlap above a configurable threshold.

use typed_builder::TypedBuilder;

use crate::graph::{EdgeInput, EdgeKind};
use crate::types::{EdgeWeight, NodeId, TagId};

/// Configuration for affinity edge generation.
#[derive(Debug, Clone, TypedBuilder)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AffinityConfig {
    /// Maximum edge weight assigned to a perfect Jaccard match. Default: `0.4`.
    #[builder(default = 0.4)]
    pub max_weight: f64,
    /// Minimum Jaccard similarity to create an edge. Default: `0.3`.
    #[builder(default = 0.3)]
    pub min_jaccard: f64,
}

impl Default for AffinityConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

/// Trait for generating affinity edges from node tag sets.
pub trait AffinityGenerator: Send + Sync {
    /// Generates affinity edges for all node pairs exceeding the similarity threshold.
    fn generate(&self, node_tags: &[Vec<TagId>], config: &AffinityConfig) -> Vec<EdgeInput>;
}

/// Jaccard-based affinity generator using two-pointer merge on sorted tag slices.
pub struct JaccardAffinity;

/// Computes Jaccard similarity between two sorted tag slices.
///
/// Returns `0.0` if either slice is empty.
#[must_use]
pub fn jaccard_similarity(a: &[TagId], b: &[TagId]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let mut i = 0;
    let mut j = 0;
    let mut intersection = 0usize;
    let mut union = 0usize;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                intersection += 1;
                union += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                union += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                union += 1;
                j += 1;
            }
        }
    }

    union += (a.len() - i) + (b.len() - j);

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

impl AffinityGenerator for JaccardAffinity {
    fn generate(&self, node_tags: &[Vec<TagId>], config: &AffinityConfig) -> Vec<EdgeInput> {
        let mut edges = Vec::new();
        let n = node_tags.len();

        for i in 0..n {
            if node_tags[i].is_empty() {
                continue;
            }
            for j in (i + 1)..n {
                if node_tags[j].is_empty() {
                    continue;
                }
                let sim = jaccard_similarity(&node_tags[i], &node_tags[j]);
                if sim >= config.min_jaccard {
                    edges.push(EdgeInput {
                        source: NodeId::new(i as u32),
                        target: NodeId::new(j as u32),
                        weight: EdgeWeight::new_unchecked((sim * config.max_weight).min(1.0)),
                        kind: EdgeKind::FeatureAffinity,
                        last_activated: None,
                    });
                }
            }
        }

        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(id: u32) -> TagId {
        TagId::new(id)
    }

    #[test]
    fn jaccard_identical_sets() {
        let a = vec![tag(1), tag(2), tag(3)];
        assert!((jaccard_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a = vec![tag(1), tag(2)];
        let b = vec![tag(3), tag(4)];
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a = vec![tag(1), tag(2), tag(3)];
        let b = vec![tag(2), tag(3), tag(4)];
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn jaccard_both_empty() {
        let a: Vec<TagId> = vec![];
        assert_eq!(jaccard_similarity(&a, &a), 0.0);
    }

    #[test]
    fn jaccard_one_empty() {
        let a = vec![tag(1)];
        let b: Vec<TagId> = vec![];
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn jaccard_is_symmetric() {
        let a = vec![tag(1), tag(2), tag(3)];
        let b = vec![tag(2), tag(4)];
        assert_eq!(jaccard_similarity(&a, &b), jaccard_similarity(&b, &a));
    }

    #[test]
    fn generates_edges_above_threshold() {
        let node_tags = vec![
            vec![tag(1), tag(2), tag(3)],
            vec![tag(2), tag(3), tag(4)],
            vec![tag(10), tag(11)],
        ];
        let config = AffinityConfig {
            max_weight: 0.4,
            min_jaccard: 0.3,
        };
        let generator = JaccardAffinity;
        let edges = generator.generate(&node_tags, &config);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, NodeId::new(0));
        assert_eq!(edges[0].target, NodeId::new(1));
        assert_eq!(edges[0].kind, EdgeKind::FeatureAffinity);
        assert!((edges[0].weight.get() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn skips_empty_tag_sets() {
        let node_tags = vec![vec![tag(1), tag(2)], vec![], vec![tag(1), tag(2)]];
        let config = AffinityConfig {
            max_weight: 0.4,
            min_jaccard: 0.0,
        };
        let generator = JaccardAffinity;
        let edges = generator.generate(&node_tags, &config);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, NodeId::new(0));
        assert_eq!(edges[0].target, NodeId::new(2));
    }

    #[test]
    fn edges_have_canonical_source_less_than_target() {
        let node_tags = vec![vec![tag(1)], vec![tag(1)]];
        let config = AffinityConfig {
            max_weight: 0.4,
            min_jaccard: 0.0,
        };
        let generator = JaccardAffinity;
        let edges = generator.generate(&node_tags, &config);
        for edge in &edges {
            assert!(edge.source.get() < edge.target.get());
        }
    }
}
