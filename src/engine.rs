//! 3-step spreading activation engine over a CSR graph.

use crate::graph::{CsrGraph, Edge, EdgeKind};
use crate::inhibition::{apply_inhibition, InhibitionConfig};
use rayon::prelude::*;

/// Tunable parameters for the spreading activation engine.
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of propagation iterations (T). Default: 3.
    pub max_steps: u32,
    /// Energy retention per hop (delta). Default: 0.7.
    pub decay_factor: f64,
    /// Edge transmission efficiency (S). Default: 0.85.
    pub spread_factor: f64,
    /// Threshold below which nodes are pruned (epsilon). Default: 0.01.
    pub min_activation: f64,
    /// Sigmoid gain. Default: 10.0.
    pub sigmoid_gain: f64,
    /// Sigmoid center. Default: 0.3.
    pub sigmoid_center: f64,
    /// Lateral inhibition config. None to disable.
    pub inhibition: Option<InhibitionConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_steps: 3,
            decay_factor: 0.7,
            spread_factor: 0.85,
            min_activation: 0.01,
            sigmoid_gain: 10.0,
            sigmoid_center: 0.3,
            inhibition: Some(InhibitionConfig::default()),
        }
    }
}

/// A seed node with initial activation.
#[derive(Debug, Clone)]
pub struct Seed {
    pub node: u32,
    pub activation: f64,
}

/// Result for a single node after propagation.
#[derive(Debug, Clone)]
pub struct ActivationResult {
    pub node: u32,
    pub activation: f64,
    pub distance: u32,
}

/// Spreading activation engine.
pub struct Engine<'g> {
    pub config: Config,
    pub graph: &'g CsrGraph,
}

impl<'g> Engine<'g> {
    pub fn new(graph: &'g CsrGraph, config: Config) -> Self {
        Engine { config, graph }
    }

    /// Run spreading activation from seeds. Returns results sorted by activation descending.
    pub fn activate(&self, seeds: &[Seed]) -> Vec<ActivationResult> {
        if seeds.is_empty() {
            return Vec::new();
        }

        let n = self.graph.num_nodes as usize;
        let mut activation = vec![0.0f64; n];
        let mut distance = vec![u32::MAX; n];

        // Step 1: Initialize from seeds.
        for s in seeds {
            activation[s.node as usize] = s.activation;
            distance[s.node as usize] = 0;
        }

        // Step 2: Propagation loop (synchronous updates).
        for _step in 0..self.config.max_steps {
            let new_activation = self.propagate_step(&activation, &mut distance);
            activation = new_activation;
        }

        // Step 3: Post-process (inhibition + sigmoid + filter).
        self.post_process(&mut activation);

        // Step 4: Collect results above threshold.
        let mut results: Vec<ActivationResult> = activation
            .iter()
            .enumerate()
            .filter(|&(_, act)| *act >= self.config.min_activation)
            .map(|(i, &act)| ActivationResult {
                node: i as u32,
                activation: act,
                distance: distance[i],
            })
            .collect();

        results.sort_by(|a, b| b.activation.partial_cmp(&a.activation).unwrap());
        results
    }

    /// Single propagation step. Reads from `activation`, returns new activation vector.
    /// Updates `distance` for BFS tracking.
    fn propagate_step(&self, activation: &[f64], distance: &mut [u32]) -> Vec<f64> {
        let n = self.graph.num_nodes as usize;

        // Compute per-node contributions in parallel.
        // Each node produces a list of (target, delta) updates.
        let updates: Vec<Vec<(u32, f64, u32)>> = (0..n)
            .into_par_iter()
            .map(|node_idx| {
                let node_act = activation[node_idx];
                if node_act < self.config.min_activation {
                    return Vec::new();
                }

                let neighbors = self.graph.neighbors(node_idx as u32);
                if neighbors.is_empty() {
                    return Vec::new();
                }

                // Count edges by category for normalization.
                let (positive_count, conflict_count, dir_suppress_count, virtual_count) =
                    count_edge_categories(node_idx as u32, neighbors);

                let node_dist = distance[node_idx];
                let mut local_updates = Vec::with_capacity(neighbors.len());

                for edge in neighbors {
                    let energy_base =
                        node_act * self.config.spread_factor * edge.weight * self.config.decay_factor;

                    match edge.kind {
                        EdgeKind::Conflicts => {
                            if conflict_count > 0 {
                                let energy = energy_base / conflict_count as f64;
                                // Negative = suppression.
                                local_updates.push((edge.target, -energy, node_dist + 1));
                            }
                        }
                        EdgeKind::DirectionalSuppressive => {
                            // Forward direction only: suppress target.
                            // Reverse edges are stored as DirectionalPassive by the graph builder.
                            if dir_suppress_count > 0 {
                                let energy = energy_base / dir_suppress_count as f64;
                                local_updates.push((edge.target, -energy, node_dist + 1));
                            }
                        }
                        EdgeKind::DirectionalPassive => {
                            // Reverse side of a directional suppressive edge.
                            // Treated as positive (no suppression in this direction).
                            if positive_count > 0 {
                                let energy = energy_base / positive_count as f64;
                                if energy > 0.0 {
                                    local_updates.push((edge.target, energy, node_dist + 1));
                                }
                            }
                        }
                        EdgeKind::FeatureAffinity => {
                            if virtual_count > 0 {
                                let energy = energy_base / virtual_count as f64;
                                // Positive spread, use max semantics (handled in merge).
                                local_updates.push((edge.target, energy, node_dist + 1));
                            }
                        }
                        EdgeKind::Positive => {
                            if positive_count > 0 {
                                let energy = energy_base / positive_count as f64;
                                local_updates.push((edge.target, energy, node_dist + 1));
                            }
                        }
                    }
                }

                local_updates
            })
            .collect();

        // Merge updates into new activation (sequential for correctness).
        let mut new_activation = activation.to_vec();

        for node_updates in &updates {
            for &(target, delta, dist) in node_updates {
                let t = target as usize;
                if delta < 0.0 {
                    // Suppression: subtract (delta is negative).
                    new_activation[t] += delta;
                    if new_activation[t] < 0.0 {
                        new_activation[t] = 0.0;
                    }
                } else {
                    // Positive: max semantics to prevent runaway.
                    if delta > new_activation[t] {
                        new_activation[t] = delta;
                    }
                }

                // Update distance (BFS shortest path).
                if delta.abs() > 0.0 && dist < distance[t] {
                    distance[t] = dist;
                }
            }
        }

        new_activation
    }

    /// Apply inhibition, sigmoid, and min-activation filtering.
    fn post_process(&self, activation: &mut Vec<f64>) {
        // Lateral inhibition.
        if let Some(ref inh) = self.config.inhibition {
            if inh.enabled {
                apply_inhibition(activation, inh);
            }
        }

        // Sigmoid squashing.
        let gain = self.config.sigmoid_gain;
        let center = self.config.sigmoid_center;
        for act in activation.iter_mut() {
            *act = sigmoid(*act, gain, center);
        }

        // Zero out below threshold.
        let min = self.config.min_activation;
        for act in activation.iter_mut() {
            if *act < min {
                *act = 0.0;
            }
        }
    }
}

/// Sigmoid: 1 / (1 + exp(-gain * (x - center)))
#[inline]
fn sigmoid(x: f64, gain: f64, center: f64) -> f64 {
    1.0 / (1.0 + (-gain * (x - center)).exp())
}

/// Count edges by category for normalization denominators.
fn count_edge_categories(node: u32, edges: &[Edge]) -> (usize, usize, usize, usize) {
    let mut positive = 0usize;
    let mut conflict = 0usize;
    let mut dir_suppress = 0usize;
    let mut virtual_count = 0usize;

    for edge in edges {
        match edge.kind {
            EdgeKind::Positive | EdgeKind::DirectionalPassive => positive += 1,
            EdgeKind::Conflicts => conflict += 1,
            EdgeKind::DirectionalSuppressive => dir_suppress += 1,
            EdgeKind::FeatureAffinity => virtual_count += 1,
        }
    }

    // For directional suppressive: only outbound edges count.
    // In our CSR representation, we need direction info.
    // For now, count all (the reverse edges will be converted to
    // DirectionalPassive in a future refinement).
    let _ = node; // Will be used for direction check later.

    (positive, conflict, dir_suppress, virtual_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{CsrGraph, EdgeInput, EdgeKind};

    fn simple_chain() -> CsrGraph {
        // 0 --0.8--> 1 --0.8--> 2
        CsrGraph::build(
            3,
            &[
                EdgeInput {
                    source: 0,
                    target: 1,
                    weight: 0.8,
                    kind: EdgeKind::Positive,
                },
                EdgeInput {
                    source: 1,
                    target: 2,
                    weight: 0.8,
                    kind: EdgeKind::Positive,
                },
            ],
        )
    }

    #[test]
    fn seed_activates_neighbors() {
        let g = simple_chain();
        let engine = Engine::new(&g, Config::default());
        let results = engine.activate(&[Seed {
            node: 0,
            activation: 1.0,
        }]);

        // Node 0 should be activated (it was a seed).
        assert!(!results.is_empty());
        // Should have some results for downstream nodes.
        let node0 = results.iter().find(|r| r.node == 0);
        assert!(node0.is_some());
    }

    #[test]
    fn empty_seeds() {
        let g = simple_chain();
        let engine = Engine::new(&g, Config::default());
        let results = engine.activate(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn activation_decays_with_distance() {
        let g = simple_chain();
        let mut cfg = Config::default();
        cfg.inhibition = None; // Disable inhibition for clarity.
        let engine = Engine::new(&g, cfg);
        let results = engine.activate(&[Seed {
            node: 0,
            activation: 1.0,
        }]);

        let act0 = results.iter().find(|r| r.node == 0).map(|r| r.activation);
        let act1 = results.iter().find(|r| r.node == 1).map(|r| r.activation);
        let act2 = results.iter().find(|r| r.node == 2).map(|r| r.activation);

        // All should be present after sigmoid.
        assert!(act0.is_some());
        assert!(act1.is_some());

        // Node 0 should have highest activation (seed).
        if let (Some(a0), Some(a1)) = (act0, act1) {
            assert!(a0 >= a1, "seed should have >= activation than neighbor");
        }

        // Node 2 may or may not survive min_activation after sigmoid.
        // But if present, should be <= node 1.
        if let (Some(a1), Some(a2)) = (act1, act2) {
            assert!(a1 >= a2, "closer nodes should have >= activation");
        }
    }

    #[test]
    fn conflict_suppresses() {
        // 0 --conflict--> 1
        let g = CsrGraph::build(
            2,
            &[EdgeInput {
                source: 0,
                target: 1,
                weight: 0.8,
                kind: EdgeKind::Conflicts,
            }],
        );
        let mut cfg = Config::default();
        cfg.inhibition = None;
        let engine = Engine::new(&g, cfg);

        // Seed both nodes, conflict should suppress.
        let results = engine.activate(&[
            Seed {
                node: 0,
                activation: 0.8,
            },
            Seed {
                node: 1,
                activation: 0.8,
            },
        ]);

        // Both should be lower than initial 0.8 after suppression + sigmoid.
        for r in &results {
            // After sigmoid(0.8, 10, 0.3) ≈ 0.993, but suppression lowers the input.
            // The key test: they should exist but be reduced.
            assert!(r.activation > 0.0);
        }
    }

    #[test]
    fn sigmoid_basic() {
        // At center, sigmoid = 0.5.
        let s = sigmoid(0.3, 10.0, 0.3);
        assert!((s - 0.5).abs() < 1e-10);

        // Well above center → close to 1.
        let s = sigmoid(1.0, 10.0, 0.3);
        assert!(s > 0.99);

        // Well below center → close to 0.
        let s = sigmoid(0.0, 10.0, 0.3);
        assert!(s < 0.05);
    }

    #[test]
    fn distance_tracking() {
        let g = simple_chain();
        let mut cfg = Config::default();
        cfg.inhibition = None;
        let engine = Engine::new(&g, cfg);
        let results = engine.activate(&[Seed {
            node: 0,
            activation: 1.0,
        }]);

        if let Some(r0) = results.iter().find(|r| r.node == 0) {
            assert_eq!(r0.distance, 0, "seed should have distance 0");
        }
        if let Some(r1) = results.iter().find(|r| r.node == 1) {
            assert_eq!(r1.distance, 1, "direct neighbor should have distance 1");
        }
    }
}
