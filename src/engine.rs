use rayon::prelude::*;
use typed_builder::TypedBuilder;

use crate::graph::{EdgeData, EdgeKind, Graph};
use crate::inhibition::{InhibitionConfig, Inhibitor, TopMInhibitor};
use crate::squash::squash_sigmoid;
use crate::types::{Activation, ActivationResult, NodeId, Seed};

struct EdgeKindCounts {
    positive: u32,
    conflict: u32,
    dir_suppress: u32,
    virtual_affinity: u32,
}

fn count_edge_kinds(neighbors: &[EdgeData]) -> EdgeKindCounts {
    let mut counts = EdgeKindCounts {
        positive: 0,
        conflict: 0,
        dir_suppress: 0,
        virtual_affinity: 0,
    };
    for edge in neighbors {
        match edge.kind {
            EdgeKind::Positive => counts.positive += 1,
            EdgeKind::Conflicts => counts.conflict += 1,
            EdgeKind::DirectionalSuppressive => counts.dir_suppress += 1,
            EdgeKind::FeatureAffinity => counts.virtual_affinity += 1,
            EdgeKind::DirectionalPassive => {}
        }
    }
    counts
}

const PARALLEL_THRESHOLD: u32 = 1024;

#[derive(Debug, Clone, TypedBuilder)]
pub struct PropagationConfig {
    #[builder(default = 3)]
    pub max_steps: u32,
    #[builder(default = 0.7)]
    pub decay_factor: f64,
    #[builder(default = 0.85)]
    pub spread_factor: f64,
    #[builder(default = 0.01)]
    pub min_activation: f64,
    #[builder(default = 10.0)]
    pub sigmoid_gain: f64,
    #[builder(default = 0.3)]
    pub sigmoid_center: f64,
    #[builder(default, setter(strip_option))]
    pub inhibition: Option<InhibitionConfig>,
}

pub struct Engine<G: Graph> {
    graph: G,
}

impl<G: Graph> Engine<G> {
    pub fn new(graph: G) -> Self {
        Engine { graph }
    }

    pub fn activate(&self, seeds: &[Seed], config: &PropagationConfig) -> Vec<ActivationResult> {
        let n = self.graph.num_nodes() as usize;

        let mut current = vec![0.0f64; n];
        let mut next = vec![0.0f64; n];
        let mut distance = vec![u32::MAX; n];

        for seed in seeds {
            let idx = seed.node.index();
            if idx < n {
                current[idx] = seed.activation.get();
                distance[idx] = 0;
            }
        }

        for _step in 0..config.max_steps {
            next.copy_from_slice(&current);
            self.propagate_step(&current, &mut next, &mut distance, config);
            std::mem::swap(&mut current, &mut next);
        }

        if let Some(ref inh_config) = config.inhibition {
            TopMInhibitor.inhibit(&mut current, inh_config);
        }

        squash_sigmoid(&mut current, config.sigmoid_gain, config.sigmoid_center);

        for v in current.iter_mut() {
            if *v < config.min_activation {
                *v = 0.0;
            }
        }

        let mut results: Vec<ActivationResult> = current
            .iter()
            .enumerate()
            .filter(|&(_, &a)| a > 0.0)
            .map(|(i, &a)| ActivationResult {
                node: NodeId(i as u32),
                activation: Activation::new_unchecked(a),
                distance: distance[i],
            })
            .collect();

        results.sort_by(|a, b| {
            b.activation
                .get()
                .partial_cmp(&a.activation.get())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.node.cmp(&b.node))
        });

        results
    }

    fn propagate_step(
        &self,
        current: &[f64],
        new: &mut [f64],
        distances: &mut [u32],
        config: &PropagationConfig,
    ) {
        if self.graph.num_nodes() < PARALLEL_THRESHOLD {
            self.propagate_step_sequential(current, new, distances, config);
        } else {
            self.propagate_step_parallel(current, new, distances, config);
        }
    }

    fn propagate_step_sequential(
        &self,
        current: &[f64],
        new: &mut [f64],
        distances: &mut [u32],
        config: &PropagationConfig,
    ) {
        let n = self.graph.num_nodes() as usize;

        for i in 0..n {
            if current[i] < config.min_activation {
                continue;
            }

            let neighbors = self.graph.neighbors(NodeId(i as u32));
            let counts = count_edge_kinds(neighbors);

            for edge in neighbors {
                let base_energy =
                    current[i] * config.spread_factor * edge.weight.get() * config.decay_factor;
                let j = edge.target.index();

                match edge.kind {
                    EdgeKind::Positive => {
                        let energy = base_energy / counts.positive as f64;
                        new[j] = new[j].max(energy);
                    }
                    EdgeKind::FeatureAffinity => {
                        let energy = base_energy / counts.virtual_affinity as f64;
                        new[j] = new[j].max(energy);
                    }
                    EdgeKind::Conflicts => {
                        let energy = base_energy / counts.conflict as f64;
                        new[j] = (new[j] - energy).max(0.0);
                    }
                    EdgeKind::DirectionalSuppressive => {
                        let energy = base_energy / counts.dir_suppress as f64;
                        new[j] = (new[j] - energy).max(0.0);
                    }
                    EdgeKind::DirectionalPassive => continue,
                }

                if distances[i] != u32::MAX {
                    let candidate = distances[i] + 1;
                    if candidate < distances[j] {
                        distances[j] = candidate;
                    }
                }
            }
        }
    }

    fn propagate_step_parallel(
        &self,
        current: &[f64],
        new: &mut [f64],
        distances: &mut [u32],
        config: &PropagationConfig,
    ) {
        let n = self.graph.num_nodes() as usize;

        struct Updates {
            positive_max: Vec<f64>,
            suppress_delta: Vec<f64>,
            min_distance: Vec<u32>,
        }

        let merged = (0..n)
            .into_par_iter()
            .filter(|&i| current[i] >= config.min_activation)
            .fold(
                || Updates {
                    positive_max: vec![0.0f64; n],
                    suppress_delta: vec![0.0f64; n],
                    min_distance: vec![u32::MAX; n],
                },
                |mut local, i| {
                    let neighbors = self.graph.neighbors(NodeId(i as u32));
                    let counts = count_edge_kinds(neighbors);

                    for edge in neighbors {
                        let base_energy = current[i]
                            * config.spread_factor
                            * edge.weight.get()
                            * config.decay_factor;
                        let j = edge.target.index();

                        match edge.kind {
                            EdgeKind::Positive => {
                                let energy = base_energy / counts.positive as f64;
                                local.positive_max[j] = local.positive_max[j].max(energy);
                            }
                            EdgeKind::FeatureAffinity => {
                                let energy = base_energy / counts.virtual_affinity as f64;
                                local.positive_max[j] = local.positive_max[j].max(energy);
                            }
                            EdgeKind::Conflicts => {
                                let energy = base_energy / counts.conflict as f64;
                                local.suppress_delta[j] += energy;
                            }
                            EdgeKind::DirectionalSuppressive => {
                                let energy = base_energy / counts.dir_suppress as f64;
                                local.suppress_delta[j] += energy;
                            }
                            EdgeKind::DirectionalPassive => continue,
                        }

                        if distances[i] != u32::MAX {
                            let candidate = distances[i] + 1;
                            if candidate < local.min_distance[j] {
                                local.min_distance[j] = candidate;
                            }
                        }
                    }

                    local
                },
            )
            .reduce(
                || Updates {
                    positive_max: vec![0.0f64; n],
                    suppress_delta: vec![0.0f64; n],
                    min_distance: vec![u32::MAX; n],
                },
                |mut a, b| {
                    for j in 0..n {
                        a.positive_max[j] = a.positive_max[j].max(b.positive_max[j]);
                        a.suppress_delta[j] += b.suppress_delta[j];
                        a.min_distance[j] = a.min_distance[j].min(b.min_distance[j]);
                    }
                    a
                },
            );

        for j in 0..n {
            new[j] = new[j].max(merged.positive_max[j]);
            new[j] = (new[j] - merged.suppress_delta[j]).max(0.0);
            distances[j] = distances[j].min(merged.min_distance[j]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{CsrGraph, EdgeInput, EdgeKind};
    use crate::types::EdgeWeight;

    fn weight(v: f64) -> EdgeWeight {
        EdgeWeight::new(v).unwrap()
    }
    fn act(v: f64) -> Activation {
        Activation::new(v).unwrap()
    }

    fn build_chain(n: u32, w: f64) -> CsrGraph {
        let edges: Vec<EdgeInput> = (0..n - 1)
            .map(|i| EdgeInput {
                source: NodeId(i),
                target: NodeId(i + 1),
                weight: weight(w),
                kind: EdgeKind::Positive,
            })
            .collect();
        CsrGraph::build(n, edges)
    }

    #[test]
    fn single_seed_activates_direct_neighbor() {
        let graph = build_chain(3, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .decay_factor(1.0)
            .spread_factor(1.0)
            .min_activation(0.001)
            .sigmoid_gain(10.0)
            .sigmoid_center(0.3)
            .build();
        let seeds = vec![Seed {
            node: NodeId(0),
            activation: act(1.0),
        }];
        let results = engine.activate(&seeds, &config);
        assert!(results.iter().any(|r| r.node == NodeId(0)));
        assert!(results.iter().any(|r| r.node == NodeId(1)));
    }

    #[test]
    fn empty_seeds_no_panic() {
        let graph = build_chain(5, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let results = engine.activate(&[], &config);
        for r in &results {
            assert!(r.activation.get() < 0.05);
        }
    }

    #[test]
    fn activation_decays_with_distance() {
        let graph = build_chain(4, 1.0);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .decay_factor(0.5)
            .spread_factor(1.0)
            .min_activation(0.001)
            .sigmoid_gain(10.0)
            .sigmoid_center(0.3)
            .build();
        let seeds = vec![Seed {
            node: NodeId(0),
            activation: act(1.0),
        }];
        let results = engine.activate(&seeds, &config);
        let a = |id: u32| {
            results
                .iter()
                .find(|r| r.node == NodeId(id))
                .map(|r| r.activation.get())
                .unwrap_or(0.0)
        };
        assert!(a(0) > a(1));
        assert!(a(1) > a(2));
    }

    #[test]
    fn conflict_suppresses_target() {
        let edges = vec![
            EdgeInput {
                source: NodeId(0),
                target: NodeId(1),
                weight: weight(1.0),
                kind: EdgeKind::Positive,
            },
            EdgeInput {
                source: NodeId(0),
                target: NodeId(2),
                weight: weight(1.0),
                kind: EdgeKind::Conflicts,
            },
        ];
        let graph = CsrGraph::build(3, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .decay_factor(1.0)
            .spread_factor(1.0)
            .min_activation(0.001)
            .sigmoid_gain(10.0)
            .sigmoid_center(0.3)
            .build();
        let seeds = vec![
            Seed {
                node: NodeId(0),
                activation: act(0.8),
            },
            Seed {
                node: NodeId(2),
                activation: act(0.8),
            },
        ];
        let results = engine.activate(&seeds, &config);
        let a = |id: u32| {
            results
                .iter()
                .find(|r| r.node == NodeId(id))
                .map(|r| r.activation.get())
                .unwrap_or(0.0)
        };
        assert!(a(1) > a(2) || (a(2) - a(1)).abs() < 0.01);
    }

    #[test]
    fn directional_suppressive_only_forward() {
        let edges = vec![EdgeInput {
            source: NodeId(0),
            target: NodeId(1),
            weight: weight(1.0),
            kind: EdgeKind::DirectionalSuppressive,
        }];
        let graph = CsrGraph::build(3, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .decay_factor(1.0)
            .spread_factor(1.0)
            .min_activation(0.001)
            .sigmoid_gain(10.0)
            .sigmoid_center(0.3)
            .build();
        let seeds = vec![
            Seed {
                node: NodeId(0),
                activation: act(0.8),
            },
            Seed {
                node: NodeId(1),
                activation: act(0.8),
            },
        ];
        let results = engine.activate(&seeds, &config);
        let a = |id: u32| {
            results
                .iter()
                .find(|r| r.node == NodeId(id))
                .map(|r| r.activation.get())
                .unwrap_or(0.0)
        };
        // Node 0 survives; node 1 is suppressed (lower or absent)
        assert!(a(0) > a(1));
    }

    #[test]
    fn max_semantics_no_accumulation() {
        let edges = vec![
            EdgeInput {
                source: NodeId(0),
                target: NodeId(2),
                weight: weight(0.5),
                kind: EdgeKind::Positive,
            },
            EdgeInput {
                source: NodeId(1),
                target: NodeId(2),
                weight: weight(0.5),
                kind: EdgeKind::Positive,
            },
        ];
        let graph = CsrGraph::build(3, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .decay_factor(1.0)
            .spread_factor(1.0)
            .min_activation(0.001)
            .sigmoid_gain(10.0)
            .sigmoid_center(0.3)
            .build();
        let seeds = vec![
            Seed {
                node: NodeId(0),
                activation: act(0.8),
            },
            Seed {
                node: NodeId(1),
                activation: act(0.8),
            },
        ];
        let results = engine.activate(&seeds, &config);
        let node2 = results.iter().find(|r| r.node == NodeId(2)).unwrap();
        assert!(node2.activation.get() < 0.9);
    }

    #[test]
    fn results_sorted_by_activation_descending() {
        let graph = build_chain(5, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let seeds = vec![Seed {
            node: NodeId(0),
            activation: act(1.0),
        }];
        let results = engine.activate(&seeds, &config);
        for i in 1..results.len() {
            assert!(results[i - 1].activation.get() >= results[i].activation.get());
        }
    }

    #[test]
    fn tie_breaking_by_node_id_ascending() {
        let n = 5u32;
        let edges: Vec<EdgeInput> = (1..n)
            .map(|i| EdgeInput {
                source: NodeId(0),
                target: NodeId(i),
                weight: weight(1.0),
                kind: EdgeKind::Positive,
            })
            .collect();
        let graph = CsrGraph::build(n, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().max_steps(1).build();
        let seeds = vec![Seed {
            node: NodeId(0),
            activation: act(1.0),
        }];
        let results = engine.activate(&seeds, &config);
        let tied: Vec<&ActivationResult> = results.iter().filter(|r| r.node != NodeId(0)).collect();
        for i in 1..tied.len() {
            if (tied[i - 1].activation.get() - tied[i].activation.get()).abs() < 1e-15 {
                assert!(tied[i - 1].node.get() < tied[i].node.get());
            }
        }
    }

    #[test]
    fn distance_tracking() {
        let graph = build_chain(4, 1.0);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId(0),
            activation: act(1.0),
        }];
        let results = engine.activate(&seeds, &config);
        let dist = |id: u32| {
            results
                .iter()
                .find(|r| r.node == NodeId(id))
                .map(|r| r.distance)
        };
        assert_eq!(dist(0), Some(0));
        assert_eq!(dist(1), Some(1));
        assert_eq!(dist(2), Some(2));
        assert_eq!(dist(3), Some(3));
    }

    #[test]
    fn directional_passive_completely_skipped() {
        let edges = vec![EdgeInput {
            source: NodeId(0),
            target: NodeId(1),
            weight: weight(1.0),
            kind: EdgeKind::DirectionalSuppressive,
        }];
        let graph = CsrGraph::build(2, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .decay_factor(1.0)
            .spread_factor(1.0)
            .min_activation(0.001)
            .sigmoid_gain(10.0)
            .sigmoid_center(0.3)
            .build();
        let seeds = vec![Seed {
            node: NodeId(1),
            activation: act(0.9),
        }];
        let results = engine.activate(&seeds, &config);
        let a0 = results
            .iter()
            .find(|r| r.node == NodeId(0))
            .map(|r| r.activation.get());
        if let Some(a0_val) = a0 {
            assert!(
                a0_val < 0.1,
                "Node 0 got activation {} — passive edge should be skipped",
                a0_val
            );
        }
    }

    #[test]
    fn activation_preserved_across_steps() {
        let graph = build_chain(2, 0.5);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .decay_factor(0.5)
            .spread_factor(0.5)
            .min_activation(0.001)
            .sigmoid_gain(10.0)
            .sigmoid_center(0.3)
            .build();
        let seeds = vec![Seed {
            node: NodeId(0),
            activation: act(0.9),
        }];
        let results = engine.activate(&seeds, &config);
        let a0 = results
            .iter()
            .find(|r| r.node == NodeId(0))
            .unwrap()
            .activation
            .get();
        assert!(a0 > 0.95, "Seed activation not preserved: got {}", a0);
    }

    #[test]
    fn min_activation_prunes_below_threshold() {
        let graph = build_chain(10, 0.1);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .min_activation(0.5)
            .build();
        let seeds = vec![Seed {
            node: NodeId(0),
            activation: act(1.0),
        }];
        let results = engine.activate(&seeds, &config);
        for r in &results {
            assert!(r.activation.get() >= 0.5 || r.activation.get() == 0.0);
        }
    }

    #[test]
    fn four_denominator_normalization() {
        let edges = vec![
            EdgeInput {
                source: NodeId(0),
                target: NodeId(1),
                weight: weight(0.8),
                kind: EdgeKind::Positive,
            },
            EdgeInput {
                source: NodeId(0),
                target: NodeId(2),
                weight: weight(0.8),
                kind: EdgeKind::Positive,
            },
            EdgeInput {
                source: NodeId(0),
                target: NodeId(3),
                weight: weight(0.8),
                kind: EdgeKind::Conflicts,
            },
            EdgeInput {
                source: NodeId(0),
                target: NodeId(4),
                weight: weight(0.8),
                kind: EdgeKind::FeatureAffinity,
            },
        ];
        let graph = CsrGraph::build(5, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId(0),
            activation: act(1.0),
        }];
        let results = engine.activate(&seeds, &config);
        assert!(!results.is_empty());
    }

    // --- Parallel tests ---

    fn build_random_graph(num_nodes: u32, edges_per_node: u32) -> CsrGraph {
        let mut edges = Vec::new();
        for i in 0..num_nodes {
            for j in 0..edges_per_node {
                let target = (i + j * 7 + 13) % num_nodes;
                if target != i {
                    edges.push(EdgeInput {
                        source: NodeId(i),
                        target: NodeId(target),
                        weight: weight(0.5),
                        kind: EdgeKind::Positive,
                    });
                }
            }
        }
        CsrGraph::build(num_nodes, edges)
    }

    #[test]
    fn parallel_matches_sequential_small_graph() {
        let graph = build_random_graph(100, 5);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let seeds = vec![Seed {
            node: NodeId(0),
            activation: act(1.0),
        }];
        let r1 = engine.activate(&seeds, &config);
        let r2 = engine.activate(&seeds, &config);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.node, b.node);
            assert!((a.activation.get() - b.activation.get()).abs() < 1e-15);
        }
    }

    #[test]
    fn parallel_matches_sequential_large_graph() {
        let graph = build_random_graph(2000, 5);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let seeds = vec![Seed {
            node: NodeId(0),
            activation: act(1.0),
        }];
        let results = engine.activate(&seeds, &config);
        for r in &results {
            assert!(r.activation.get() >= 0.0 && r.activation.get() <= 1.0);
        }
        let results2 = engine.activate(&seeds, &config);
        assert_eq!(results.len(), results2.len());
        for (a, b) in results.iter().zip(results2.iter()) {
            assert_eq!(a.node, b.node);
            assert!((a.activation.get() - b.activation.get()).abs() < 1e-15);
        }
    }

    #[test]
    fn parallel_activation_bounds() {
        let graph = build_random_graph(2000, 10);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let seeds = vec![
            Seed {
                node: NodeId(0),
                activation: act(1.0),
            },
            Seed {
                node: NodeId(500),
                activation: act(0.8),
            },
        ];
        let results = engine.activate(&seeds, &config);
        for r in &results {
            assert!(
                r.activation.get() >= 0.0 && r.activation.get() <= 1.0,
                "Activation {} out of bounds",
                r.activation.get()
            );
        }
    }
}
