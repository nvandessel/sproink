//! Tag-based affinity edge generation using Jaccard similarity.
//!
//! Given per-node tag sets, generates [`FeatureAffinity`](crate::EdgeKind::FeatureAffinity)
//! edges between nodes whose tags overlap above a configurable threshold.

use typed_builder::TypedBuilder;

use crate::error::SproinkError;
use crate::graph::{EdgeInput, EdgeKind};
use crate::types::{EdgeWeight, NodeId, TagId};

/// Configuration for affinity edge generation.
///
/// # Invariants
///
/// `max_weight` **must** lie in `[0.0, 1.0]`. The typed-builder cannot enforce
/// this at compile time, so [`JaccardAffinity::generate`] clamps the final edge
/// weight into `[0.0, 1.0]` defensively. Passing a value outside `[0.0, 1.0]`
/// is considered a programming error and produces silently clamped weights
/// rather than the naive linear scaling.
#[derive(Debug, Clone, TypedBuilder)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AffinityConfig {
    /// Maximum edge weight assigned to a perfect Jaccard match. Default: `0.4`.
    ///
    /// Must be in `[0.0, 1.0]`. See the type-level docs for details.
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

impl AffinityConfig {
    /// Validates that the config has sane values.
    pub fn validate(&self) -> Result<(), SproinkError> {
        if !self.max_weight.is_finite() || self.max_weight < 0.0 || self.max_weight > 1.0 {
            return Err(SproinkError::InvalidValue {
                field: "affinity_max_weight",
                value: self.max_weight,
            });
        }
        if !self.min_jaccard.is_finite() || self.min_jaccard < 0.0 {
            return Err(SproinkError::InvalidValue {
                field: "affinity_min_jaccard",
                value: self.min_jaccard,
            });
        }
        Ok(())
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
///
/// # Contract
///
/// The input slices **MUST** be sorted in ascending order and contain unique
/// elements (no duplicates). Violating this contract produces incorrect
/// results. In debug builds, violations are caught by `debug_assert!`; in
/// release builds, the caller is trusted for performance.
#[must_use]
pub fn jaccard_similarity(a: &[TagId], b: &[TagId]) -> f64 {
    debug_assert!(
        a.windows(2).all(|w| w[0] < w[1]),
        "jaccard_similarity: slice `a` must be sorted ascending with no duplicates"
    );
    debug_assert!(
        b.windows(2).all(|w| w[0] < w[1]),
        "jaccard_similarity: slice `b` must be sorted ascending with no duplicates"
    );

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
        if config.max_weight.is_nan() {
            return Vec::new();
        }
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
                        weight: EdgeWeight::new_unchecked(
                            (sim * config.max_weight).clamp(0.0, 1.0),
                        ),
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
    fn negative_max_weight_is_clamped_to_zero() {
        // Regression: I38 — negative max_weight previously bypassed the
        // EdgeWeight invariant via new_unchecked.
        let node_tags = vec![vec![tag(1), tag(2)], vec![tag(1), tag(2)]];
        let config = AffinityConfig {
            max_weight: -5.0,
            min_jaccard: 0.0,
        };
        let generator = JaccardAffinity;
        let edges = generator.generate(&node_tags, &config);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].weight.get(), 0.0);
    }

    #[test]
    fn excessive_max_weight_is_clamped_to_one() {
        let node_tags = vec![vec![tag(1)], vec![tag(1)]];
        let config = AffinityConfig {
            max_weight: 50.0,
            min_jaccard: 0.0,
        };
        let generator = JaccardAffinity;
        let edges = generator.generate(&node_tags, &config);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].weight.get(), 1.0);
    }

    #[test]
    #[should_panic(expected = "must be sorted")]
    fn jaccard_debug_asserts_unsorted_input() {
        let a = vec![tag(3), tag(1), tag(2)];
        let b = vec![tag(1), tag(2)];
        let _ = jaccard_similarity(&a, &b);
    }

    #[test]
    #[should_panic(expected = "must be sorted")]
    fn jaccard_debug_asserts_duplicate_input() {
        // Duplicates violate the unique-elements contract and are rejected by
        // the same strictly-ascending assertion.
        let a = vec![tag(1), tag(1), tag(2)];
        let b = vec![tag(1), tag(2)];
        let _ = jaccard_similarity(&a, &b);
    }

    #[test]
    fn validate_rejects_nan_max_weight() {
        let cfg = AffinityConfig {
            max_weight: f64::NAN,
            min_jaccard: 0.3,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_max_weight() {
        let cfg = AffinityConfig {
            max_weight: -1.0,
            min_jaccard: 0.3,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_nan_min_jaccard() {
        let cfg = AffinityConfig {
            max_weight: 0.4,
            min_jaccard: f64::NAN,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_min_jaccard() {
        let cfg = AffinityConfig {
            max_weight: 0.4,
            min_jaccard: -0.1,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_defaults() {
        assert!(AffinityConfig::default().validate().is_ok());
    }

    #[test]
    fn nan_max_weight_generates_no_edges() {
        let node_tags = vec![vec![tag(1), tag(2)], vec![tag(1), tag(2)]];
        let config = AffinityConfig {
            max_weight: f64::NAN,
            min_jaccard: 0.0,
        };
        let generator = JaccardAffinity;
        let edges = generator.generate(&node_tags, &config);
        assert!(edges.is_empty());
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
