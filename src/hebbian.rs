//! Hebbian/Oja learning rule for weight updates.
//!
//! After propagation, co-activated node pairs can have their edge weights
//! updated using Oja's rule, which combines Hebbian strengthening with
//! a normalization term for stability.

use std::collections::HashSet;
use typed_builder::TypedBuilder;

use crate::types::{Activation, ActivationResult, EdgeWeight, NodeId};

/// Configuration for Hebbian/Oja learning.
#[derive(Debug, Clone, TypedBuilder)]
pub struct HebbianConfig {
    /// Learning rate (η). Default: `0.05`.
    #[builder(default = 0.05)]
    pub learning_rate: f64,
    /// Minimum allowed edge weight after update. Default: `0.01`.
    #[builder(default = 0.01)]
    pub min_weight: f64,
    /// Maximum allowed edge weight after update. Default: `0.95`.
    #[builder(default = 0.95)]
    pub max_weight: f64,
    /// Minimum activation to be considered co-active. Default: `0.15`.
    #[builder(default = 0.15)]
    pub activation_threshold: f64,
}

impl Default for HebbianConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

/// A pair of co-activated nodes extracted after propagation.
///
/// Pairs are canonically ordered: `node_a < node_b`.
#[derive(Debug, Clone, PartialEq)]
pub struct CoActivationPair {
    /// The lower-ID node.
    pub node_a: NodeId,
    /// The higher-ID node.
    pub node_b: NodeId,
    /// Activation of `node_a`.
    pub activation_a: Activation,
    /// Activation of `node_b`.
    pub activation_b: Activation,
}

/// Trait for Hebbian weight-update strategies.
pub trait Learner: Send + Sync {
    /// Computes a new edge weight given the current weight and both endpoint activations.
    fn update_weight(
        &self,
        current_weight: EdgeWeight,
        activation_a: Activation,
        activation_b: Activation,
        config: &HebbianConfig,
    ) -> EdgeWeight;
}

/// Oja's rule learner: `ΔW = η · (a_i · a_j − a_j² · W)`.
pub struct OjaLearner;

impl Learner for OjaLearner {
    fn update_weight(
        &self,
        current_weight: EdgeWeight,
        activation_a: Activation,
        activation_b: Activation,
        config: &HebbianConfig,
    ) -> EdgeWeight {
        let w = current_weight.get();
        let a_i = activation_a.get();
        let a_j = activation_b.get();
        let eta = config.learning_rate;

        let dw = eta * (a_i * a_j - a_j * a_j * w);
        let new_w = w + dw;

        // NaN/Inf safety: fall back to min_weight
        let new_w = if new_w.is_finite() {
            new_w.clamp(config.min_weight, config.max_weight)
        } else {
            config.min_weight
        };

        EdgeWeight::new_unchecked(new_w)
    }
}

/// Extracts co-activation pairs from propagation results.
///
/// Filters to nodes above `config.activation_threshold`, then returns all
/// pairs excluding seed-seed combinations. Pairs are canonically ordered.
#[must_use]
pub fn extract_co_activation_pairs(
    results: &[ActivationResult],
    seed_nodes: &HashSet<NodeId>,
    config: &HebbianConfig,
) -> Vec<CoActivationPair> {
    // Filter nodes above threshold
    let active: Vec<&ActivationResult> = results
        .iter()
        .filter(|r| r.activation.get() >= config.activation_threshold)
        .collect();

    let mut pairs = Vec::with_capacity(active.len() * active.len().saturating_sub(1) / 2);
    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            let (a, b) = if active[i].node < active[j].node {
                (active[i], active[j])
            } else {
                (active[j], active[i])
            };

            // Exclude seed-seed pairs
            if seed_nodes.contains(&a.node) && seed_nodes.contains(&b.node) {
                continue;
            }

            pairs.push(CoActivationPair {
                node_a: a.node,
                node_b: b.node,
                activation_a: a.activation,
                activation_b: b.activation,
            });
        }
    }

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> HebbianConfig {
        HebbianConfig::builder().build()
    }

    fn act(v: f64) -> Activation {
        Activation::new(v).unwrap()
    }
    fn wt(v: f64) -> EdgeWeight {
        EdgeWeight::new(v).unwrap()
    }

    #[test]
    fn oja_strengthens_coactivation() {
        let learner = OjaLearner;
        let cfg = default_config();
        let old_w = wt(0.3);
        let new_w = learner.update_weight(old_w, act(0.8), act(0.7), &cfg);
        assert!(new_w.get() > old_w.get());
    }

    #[test]
    fn oja_stabilizes_at_high_weight() {
        // Equilibrium: W = A_i / A_j. When A_i=0.5, A_j=0.8, equilibrium W=0.625
        // At W=0.9 (above equilibrium), forgetting dominates → weight decreases
        let learner = OjaLearner;
        let cfg = default_config();
        let old_w = wt(0.9);
        let new_w = learner.update_weight(old_w, act(0.5), act(0.8), &cfg);
        assert!(new_w.get() < old_w.get());
    }

    #[test]
    fn oja_clamps_to_bounds() {
        let learner = OjaLearner;
        let cfg = HebbianConfig::builder()
            .min_weight(0.1)
            .max_weight(0.9)
            .learning_rate(10.0)
            .build();
        let result = learner.update_weight(wt(0.5), act(0.99), act(0.99), &cfg);
        assert!(result.get() >= 0.1 && result.get() <= 0.9);
    }

    #[test]
    fn oja_nan_safety() {
        let learner = OjaLearner;
        let cfg = default_config();
        let result = learner.update_weight(wt(0.5), act(0.0), act(0.0), &cfg);
        assert!(!result.get().is_nan());
        assert!(result.get() >= cfg.min_weight);
    }

    #[test]
    fn extracts_pairs_above_threshold() {
        let results = vec![
            ActivationResult {
                node: NodeId::new(0),
                activation: act(0.5),
                distance: 0,
                seed_source: None,
            },
            ActivationResult {
                node: NodeId::new(1),
                activation: act(0.3),
                distance: 1,
                seed_source: None,
            },
            ActivationResult {
                node: NodeId::new(2),
                activation: act(0.1),
                distance: 2,
                seed_source: None,
            },
        ];
        let seeds = HashSet::new();
        let cfg = HebbianConfig::builder().activation_threshold(0.15).build();
        let pairs = extract_co_activation_pairs(&results, &seeds, &cfg);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].node_a, NodeId::new(0));
        assert_eq!(pairs[0].node_b, NodeId::new(1));
    }

    #[test]
    fn excludes_seed_seed_pairs() {
        let results = vec![
            ActivationResult {
                node: NodeId::new(0),
                activation: act(0.8),
                distance: 0,
                seed_source: None,
            },
            ActivationResult {
                node: NodeId::new(1),
                activation: act(0.7),
                distance: 0,
                seed_source: None,
            },
            ActivationResult {
                node: NodeId::new(2),
                activation: act(0.6),
                distance: 1,
                seed_source: None,
            },
        ];
        let seeds: HashSet<NodeId> = [NodeId::new(0), NodeId::new(1)].into();
        let cfg = HebbianConfig::builder().activation_threshold(0.15).build();
        let pairs = extract_co_activation_pairs(&results, &seeds, &cfg);
        assert_eq!(pairs.len(), 2);
        assert!(
            pairs
                .iter()
                .all(|p| !(seeds.contains(&p.node_a) && seeds.contains(&p.node_b)))
        );
    }

    #[test]
    fn canonical_ordering_a_less_than_b() {
        let results = vec![
            ActivationResult {
                node: NodeId::new(5),
                activation: act(0.5),
                distance: 1,
                seed_source: None,
            },
            ActivationResult {
                node: NodeId::new(2),
                activation: act(0.4),
                distance: 1,
                seed_source: None,
            },
        ];
        let seeds = HashSet::new();
        let cfg = HebbianConfig::builder().activation_threshold(0.15).build();
        let pairs = extract_co_activation_pairs(&results, &seeds, &cfg);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].node_a < pairs[0].node_b);
    }
}
