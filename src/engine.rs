use rayon::prelude::*;
use typed_builder::TypedBuilder;

use crate::affinity::{AffinityConfig, jaccard_similarity};
use crate::error::SproinkError;
use crate::graph::{EdgeData, EdgeKind, Graph};
use crate::inhibition::{InhibitionConfig, Inhibitor, TopMInhibitor};
use crate::squash::squash_sigmoid;
use crate::types::{Activation, ActivationResult, NodeId, Seed, TagId};

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

fn temporal_weight(base_weight: f64, last_activated: Option<f64>, rho: f64, t_now: f64) -> f64 {
    match last_activated {
        Some(t_edge) => {
            let elapsed = (t_now - t_edge).max(0.0);
            base_weight * (-rho * elapsed).exp()
        }
        None => base_weight,
    }
}

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
    #[builder(default, setter(strip_option))]
    pub temporal_decay_rate: Option<f64>,
    #[builder(default, setter(strip_option))]
    pub current_time: Option<f64>,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

#[derive(Debug, Clone)]
pub struct StepSnapshot {
    pub step: u32,
    pub activations: Vec<(NodeId, f64)>,
    pub is_final: bool,
}

pub struct Engine<G: Graph> {
    graph: G,
}

impl<G: Graph> Engine<G> {
    #[must_use]
    pub fn new(graph: G) -> Self {
        Engine { graph }
    }

    pub fn activate(
        &self,
        seeds: &[Seed],
        config: &PropagationConfig,
    ) -> Result<Vec<ActivationResult>, SproinkError> {
        let (results, _) = self.activate_inner(seeds, config, None, false)?;
        Ok(results)
    }

    pub fn activate_with_steps(
        &self,
        seeds: &[Seed],
        config: &PropagationConfig,
    ) -> Result<Vec<StepSnapshot>, SproinkError> {
        let (_, snapshots) = self.activate_inner(seeds, config, None, true)?;
        Ok(snapshots.unwrap())
    }

    pub fn activate_with_affinity(
        &self,
        seeds: &[Seed],
        config: &PropagationConfig,
        tag_sets: &[Vec<TagId>],
        affinity_config: &AffinityConfig,
    ) -> Result<Vec<ActivationResult>, SproinkError> {
        let (results, _) =
            self.activate_inner(seeds, config, Some((tag_sets, affinity_config)), false)?;
        Ok(results)
    }

    pub fn activate_with_steps_and_affinity(
        &self,
        seeds: &[Seed],
        config: &PropagationConfig,
        tag_sets: &[Vec<TagId>],
        affinity_config: &AffinityConfig,
    ) -> Result<Vec<StepSnapshot>, SproinkError> {
        let (_, snapshots) =
            self.activate_inner(seeds, config, Some((tag_sets, affinity_config)), true)?;
        Ok(snapshots.unwrap())
    }

    fn activate_inner(
        &self,
        seeds: &[Seed],
        config: &PropagationConfig,
        affinity: Option<(&[Vec<TagId>], &AffinityConfig)>,
        collect_steps: bool,
    ) -> Result<(Vec<ActivationResult>, Option<Vec<StepSnapshot>>), SproinkError> {
        Self::validate_config(config)?;
        if let Some((tag_sets, _)) = affinity
            && tag_sets.len() != self.graph.num_nodes() as usize
        {
            return Err(SproinkError::InvalidValue {
                field: "tag_sets",
                value: tag_sets.len() as f64,
            });
        }

        let n = self.graph.num_nodes() as usize;
        let mut current = vec![0.0f64; n];
        let mut next = vec![0.0f64; n];
        let mut distance = vec![u32::MAX; n];
        let mut seed_sources: Vec<Option<u32>> = vec![None; n];

        for seed in seeds {
            let idx = seed.node.index();
            if idx < n {
                current[idx] = seed.activation.get();
                distance[idx] = 0;
                seed_sources[idx] = seed.source;
            }
        }

        let mut snapshots = if collect_steps {
            let mut snaps = Vec::with_capacity(config.max_steps as usize + 2);
            // Step 0: snapshot seed state (ALL seeded nodes regardless of min_activation)
            let mut step0: Vec<(NodeId, f64)> = current
                .iter()
                .enumerate()
                .filter(|&(_, &a)| a > 0.0)
                .map(|(i, &a)| (NodeId::new(i as u32), a))
                .collect();
            Self::sort_snapshot(&mut step0);
            snaps.push(StepSnapshot {
                step: 0,
                activations: step0,
                is_final: false,
            });
            Some(snaps)
        } else {
            None
        };

        for step in 0..config.max_steps {
            next.copy_from_slice(&current);
            self.propagate_step(
                &current,
                &mut next,
                &mut distance,
                &mut seed_sources,
                config,
                affinity,
            );
            std::mem::swap(&mut current, &mut next);

            if let Some(ref mut snaps) = snapshots {
                let mut snap: Vec<(NodeId, f64)> = current
                    .iter()
                    .enumerate()
                    .filter(|&(_, &a)| a >= config.min_activation)
                    .map(|(i, &a)| (NodeId::new(i as u32), a))
                    .collect();
                Self::sort_snapshot(&mut snap);
                snaps.push(StepSnapshot {
                    step: step + 1,
                    activations: snap,
                    is_final: false,
                });
            }
        }

        // Post-process
        if let Some(ref inh_config) = config.inhibition {
            TopMInhibitor.inhibit(&mut current, inh_config);
        }
        squash_sigmoid(&mut current, config.sigmoid_gain, config.sigmoid_center);
        for v in current.iter_mut() {
            if *v < config.min_activation {
                *v = 0.0;
            }
        }

        if let Some(ref mut snaps) = snapshots {
            let mut final_snap: Vec<(NodeId, f64)> = current
                .iter()
                .enumerate()
                .filter(|&(_, &a)| a > 0.0)
                .map(|(i, &a)| (NodeId::new(i as u32), a))
                .collect();
            Self::sort_snapshot(&mut final_snap);
            snaps.push(StepSnapshot {
                step: config.max_steps + 1,
                activations: final_snap,
                is_final: true,
            });
        }

        let mut results: Vec<ActivationResult> = current
            .iter()
            .enumerate()
            .filter(|&(_, &a)| a > 0.0)
            .map(|(i, &a)| ActivationResult {
                node: NodeId::new(i as u32),
                activation: Activation::new_unchecked(a),
                distance: distance[i],
                seed_source: seed_sources[i],
            })
            .collect();

        results.sort_by(|a, b| {
            b.activation
                .get()
                .partial_cmp(&a.activation.get())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.node.cmp(&b.node))
        });

        Ok((results, snapshots))
    }

    fn validate_config(config: &PropagationConfig) -> Result<(), SproinkError> {
        if config.temporal_decay_rate.is_some() && config.current_time.is_none() {
            return Err(SproinkError::InvalidValue {
                field: "current_time",
                value: 0.0,
            });
        }
        Ok(())
    }

    fn sort_snapshot(snap: &mut [(NodeId, f64)]) {
        snap.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
    }

    fn propagate_step(
        &self,
        current: &[f64],
        new: &mut [f64],
        distances: &mut [u32],
        seed_sources: &mut [Option<u32>],
        config: &PropagationConfig,
        affinity: Option<(&[Vec<TagId>], &AffinityConfig)>,
    ) {
        if self.graph.num_nodes() < PARALLEL_THRESHOLD {
            self.propagate_step_sequential(current, new, distances, seed_sources, config, affinity);
        } else {
            self.propagate_step_parallel(current, new, distances, seed_sources, config, affinity);
        }
    }

    fn propagate_step_sequential(
        &self,
        current: &[f64],
        new: &mut [f64],
        distances: &mut [u32],
        seed_sources: &mut [Option<u32>],
        config: &PropagationConfig,
        affinity: Option<(&[Vec<TagId>], &AffinityConfig)>,
    ) {
        let n = self.graph.num_nodes() as usize;

        for i in 0..n {
            if current[i] < config.min_activation {
                continue;
            }

            let neighbors = self.graph.neighbors(NodeId::new(i as u32));
            let counts = count_edge_kinds(neighbors);

            for edge in neighbors {
                let w = match config.temporal_decay_rate {
                    Some(rho) => temporal_weight(
                        edge.weight.get(),
                        edge.last_activated,
                        rho,
                        config.current_time.unwrap(),
                    ),
                    None => edge.weight.get(),
                };
                let base_energy = current[i] * config.spread_factor * w * config.decay_factor;
                let j = edge.target.index();

                match edge.kind {
                    EdgeKind::Positive => {
                        let energy = base_energy / f64::from(counts.positive);
                        new[j] = new[j].max(energy);
                    }
                    EdgeKind::FeatureAffinity => {
                        let energy = base_energy / f64::from(counts.virtual_affinity);
                        new[j] = new[j].max(energy);
                    }
                    EdgeKind::Conflicts => {
                        let energy = base_energy / f64::from(counts.conflict);
                        new[j] = (new[j] - energy).max(0.0);
                    }
                    EdgeKind::DirectionalSuppressive => {
                        let energy = base_energy / f64::from(counts.dir_suppress);
                        new[j] = (new[j] - energy).max(0.0);
                    }
                    EdgeKind::DirectionalPassive => continue,
                }

                if distances[i] != u32::MAX {
                    let candidate = distances[i] + 1;
                    if candidate < distances[j] {
                        distances[j] = candidate;
                        seed_sources[j] = seed_sources[i];
                    }
                }
            }

            // Virtual affinity edges
            if let Some((tag_sets, aff_config)) = affinity {
                let mut virtual_edges: Vec<(usize, f64)> = Vec::new();
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let sim = jaccard_similarity(&tag_sets[i], &tag_sets[j]);
                    if sim >= aff_config.min_jaccard {
                        virtual_edges.push((j, sim * aff_config.max_weight));
                    }
                }
                let virtual_count = virtual_edges.len();
                if virtual_count > 0 {
                    for &(j, vw) in &virtual_edges {
                        let energy = current[i] * config.spread_factor * vw * config.decay_factor
                            / virtual_count as f64;
                        new[j] = new[j].max(energy);
                        if distances[i] != u32::MAX {
                            let candidate = distances[i] + 1;
                            if candidate < distances[j] {
                                distances[j] = candidate;
                                seed_sources[j] = seed_sources[i];
                            }
                        }
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
        seed_sources: &mut [Option<u32>],
        config: &PropagationConfig,
        affinity: Option<(&[Vec<TagId>], &AffinityConfig)>,
    ) {
        let n = self.graph.num_nodes() as usize;

        struct Updates {
            positive_max: Vec<f64>,
            suppress_delta: Vec<f64>,
            min_distance: Vec<u32>,
            seed_source: Vec<Option<u32>>,
        }

        let merged = (0..n)
            .into_par_iter()
            .filter(|&i| current[i] >= config.min_activation)
            .fold(
                || Updates {
                    positive_max: vec![0.0f64; n],
                    suppress_delta: vec![0.0f64; n],
                    min_distance: vec![u32::MAX; n],
                    seed_source: vec![None; n],
                },
                |mut local, i| {
                    let neighbors = self.graph.neighbors(NodeId::new(i as u32));
                    let counts = count_edge_kinds(neighbors);

                    for edge in neighbors {
                        let w = match config.temporal_decay_rate {
                            Some(rho) => temporal_weight(
                                edge.weight.get(),
                                edge.last_activated,
                                rho,
                                config.current_time.unwrap(),
                            ),
                            None => edge.weight.get(),
                        };
                        let base_energy =
                            current[i] * config.spread_factor * w * config.decay_factor;
                        let j = edge.target.index();

                        match edge.kind {
                            EdgeKind::Positive => {
                                let energy = base_energy / f64::from(counts.positive);
                                local.positive_max[j] = local.positive_max[j].max(energy);
                            }
                            EdgeKind::FeatureAffinity => {
                                let energy = base_energy / f64::from(counts.virtual_affinity);
                                local.positive_max[j] = local.positive_max[j].max(energy);
                            }
                            EdgeKind::Conflicts => {
                                let energy = base_energy / f64::from(counts.conflict);
                                local.suppress_delta[j] += energy;
                            }
                            EdgeKind::DirectionalSuppressive => {
                                let energy = base_energy / f64::from(counts.dir_suppress);
                                local.suppress_delta[j] += energy;
                            }
                            EdgeKind::DirectionalPassive => continue,
                        }

                        if distances[i] != u32::MAX {
                            let candidate = distances[i] + 1;
                            if candidate < local.min_distance[j] {
                                local.min_distance[j] = candidate;
                                local.seed_source[j] = seed_sources[i];
                            }
                        }
                    }

                    // Virtual affinity edges
                    if let Some((tag_sets, aff_config)) = affinity {
                        let mut virtual_edges: Vec<(usize, f64)> = Vec::new();
                        for j in 0..n {
                            if j == i {
                                continue;
                            }
                            let sim = jaccard_similarity(&tag_sets[i], &tag_sets[j]);
                            if sim >= aff_config.min_jaccard {
                                virtual_edges.push((j, sim * aff_config.max_weight));
                            }
                        }
                        let virtual_count = virtual_edges.len();
                        if virtual_count > 0 {
                            for &(j, vw) in &virtual_edges {
                                let energy =
                                    current[i] * config.spread_factor * vw * config.decay_factor
                                        / virtual_count as f64;
                                local.positive_max[j] = local.positive_max[j].max(energy);
                                if distances[i] != u32::MAX {
                                    let candidate = distances[i] + 1;
                                    if candidate < local.min_distance[j] {
                                        local.min_distance[j] = candidate;
                                        local.seed_source[j] = seed_sources[i];
                                    }
                                }
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
                    seed_source: vec![None; n],
                },
                |mut a, b| {
                    for j in 0..n {
                        a.positive_max[j] = a.positive_max[j].max(b.positive_max[j]);
                        a.suppress_delta[j] += b.suppress_delta[j];
                        if b.min_distance[j] < a.min_distance[j] {
                            a.min_distance[j] = b.min_distance[j];
                            a.seed_source[j] = b.seed_source[j];
                        } else if b.min_distance[j] == a.min_distance[j] {
                            // Tie-breaking: lower seed source ID wins (None > Some)
                            a.seed_source[j] = match (a.seed_source[j], b.seed_source[j]) {
                                (Some(a_id), Some(b_id)) => Some(a_id.min(b_id)),
                                (Some(_), None) => a.seed_source[j],
                                (None, Some(_)) => b.seed_source[j],
                                (None, None) => None,
                            };
                        }
                    }
                    a
                },
            );

        for j in 0..n {
            new[j] = new[j].max(merged.positive_max[j]);
            new[j] = (new[j] - merged.suppress_delta[j]).max(0.0);
            if merged.min_distance[j] < distances[j] {
                distances[j] = merged.min_distance[j];
                seed_sources[j] = merged.seed_source[j];
            } else if merged.min_distance[j] == distances[j] && distances[j] != u32::MAX {
                seed_sources[j] = match (seed_sources[j], merged.seed_source[j]) {
                    (Some(a_id), Some(b_id)) => Some(a_id.min(b_id)),
                    (Some(_), None) => seed_sources[j],
                    (None, Some(_)) => merged.seed_source[j],
                    (None, None) => None,
                };
            }
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
                source: NodeId::new(i),
                target: NodeId::new(i + 1),
                weight: weight(w),
                kind: EdgeKind::Positive,
                last_activated: None,
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
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        assert!(results.iter().any(|r| r.node == NodeId::new(0)));
        assert!(results.iter().any(|r| r.node == NodeId::new(1)));
    }

    #[test]
    fn empty_seeds_no_panic() {
        let graph = build_chain(5, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let results = engine.activate(&[], &config).unwrap();
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
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        let a = |id: u32| {
            results
                .iter()
                .find(|r| r.node == NodeId::new(id))
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
                source: NodeId::new(0),
                target: NodeId::new(1),
                weight: weight(1.0),
                kind: EdgeKind::Positive,
                last_activated: None,
            },
            EdgeInput {
                source: NodeId::new(0),
                target: NodeId::new(2),
                weight: weight(1.0),
                kind: EdgeKind::Conflicts,
                last_activated: None,
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
                node: NodeId::new(0),
                activation: act(0.8),
                source: None,
            },
            Seed {
                node: NodeId::new(2),
                activation: act(0.8),
                source: None,
            },
        ];
        let results = engine.activate(&seeds, &config).unwrap();
        let a = |id: u32| {
            results
                .iter()
                .find(|r| r.node == NodeId::new(id))
                .map(|r| r.activation.get())
                .unwrap_or(0.0)
        };
        assert!(a(1) > a(2) || (a(2) - a(1)).abs() < 0.01);
    }

    #[test]
    fn directional_suppressive_only_forward() {
        let edges = vec![EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(1.0),
            kind: EdgeKind::DirectionalSuppressive,
            last_activated: None,
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
                node: NodeId::new(0),
                activation: act(0.8),
                source: None,
            },
            Seed {
                node: NodeId::new(1),
                activation: act(0.8),
                source: None,
            },
        ];
        let results = engine.activate(&seeds, &config).unwrap();
        let a = |id: u32| {
            results
                .iter()
                .find(|r| r.node == NodeId::new(id))
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
                source: NodeId::new(0),
                target: NodeId::new(2),
                weight: weight(0.5),
                kind: EdgeKind::Positive,
                last_activated: None,
            },
            EdgeInput {
                source: NodeId::new(1),
                target: NodeId::new(2),
                weight: weight(0.5),
                kind: EdgeKind::Positive,
                last_activated: None,
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
                node: NodeId::new(0),
                activation: act(0.8),
                source: None,
            },
            Seed {
                node: NodeId::new(1),
                activation: act(0.8),
                source: None,
            },
        ];
        let results = engine.activate(&seeds, &config).unwrap();
        let node2 = results.iter().find(|r| r.node == NodeId::new(2)).unwrap();
        assert!(node2.activation.get() < 0.9);
    }

    #[test]
    fn results_sorted_by_activation_descending() {
        let graph = build_chain(5, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        for i in 1..results.len() {
            assert!(results[i - 1].activation.get() >= results[i].activation.get());
        }
    }

    #[test]
    fn tie_breaking_by_node_id_ascending() {
        let n = 5u32;
        let edges: Vec<EdgeInput> = (1..n)
            .map(|i| EdgeInput {
                source: NodeId::new(0),
                target: NodeId::new(i),
                weight: weight(1.0),
                kind: EdgeKind::Positive,
                last_activated: None,
            })
            .collect();
        let graph = CsrGraph::build(n, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().max_steps(1).build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        let tied: Vec<&ActivationResult> = results
            .iter()
            .filter(|r| r.node != NodeId::new(0))
            .collect();
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
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        let dist = |id: u32| {
            results
                .iter()
                .find(|r| r.node == NodeId::new(id))
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
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(1.0),
            kind: EdgeKind::DirectionalSuppressive,
            last_activated: None,
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
            node: NodeId::new(1),
            activation: act(0.9),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        let a0 = results
            .iter()
            .find(|r| r.node == NodeId::new(0))
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
            node: NodeId::new(0),
            activation: act(0.9),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        let a0 = results
            .iter()
            .find(|r| r.node == NodeId::new(0))
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
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        for r in &results {
            assert!(r.activation.get() >= 0.5 || r.activation.get() == 0.0);
        }
    }

    #[test]
    fn four_denominator_normalization() {
        let edges = vec![
            EdgeInput {
                source: NodeId::new(0),
                target: NodeId::new(1),
                weight: weight(0.8),
                kind: EdgeKind::Positive,
                last_activated: None,
            },
            EdgeInput {
                source: NodeId::new(0),
                target: NodeId::new(2),
                weight: weight(0.8),
                kind: EdgeKind::Positive,
                last_activated: None,
            },
            EdgeInput {
                source: NodeId::new(0),
                target: NodeId::new(3),
                weight: weight(0.8),
                kind: EdgeKind::Conflicts,
                last_activated: None,
            },
            EdgeInput {
                source: NodeId::new(0),
                target: NodeId::new(4),
                weight: weight(0.8),
                kind: EdgeKind::FeatureAffinity,
                last_activated: None,
            },
        ];
        let graph = CsrGraph::build(5, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
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
                        source: NodeId::new(i),
                        target: NodeId::new(target),
                        weight: weight(0.5),
                        kind: EdgeKind::Positive,
                        last_activated: None,
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
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let r1 = engine.activate(&seeds, &config).unwrap();
        let r2 = engine.activate(&seeds, &config).unwrap();
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
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        for r in &results {
            assert!(r.activation.get() >= 0.0 && r.activation.get() <= 1.0);
        }
        let results2 = engine.activate(&seeds, &config).unwrap();
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
                node: NodeId::new(0),
                activation: act(1.0),
                source: None,
            },
            Seed {
                node: NodeId::new(500),
                activation: act(0.8),
                source: None,
            },
        ];
        let results = engine.activate(&seeds, &config).unwrap();
        for r in &results {
            assert!(
                (0.0..=1.0).contains(&r.activation.get()),
                "Activation {} out of bounds",
                r.activation.get()
            );
        }
    }

    // --- Seed source tracking ---

    #[test]
    fn single_seed_source_propagates_to_all() {
        let graph = build_chain(4, 1.0);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: Some(42),
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        for r in &results {
            assert_eq!(
                r.seed_source,
                Some(42),
                "Node {:?} should have seed_source 42",
                r.node
            );
        }
    }

    #[test]
    fn no_seed_source_means_none() {
        let graph = build_chain(4, 1.0);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        for r in &results {
            assert_eq!(
                r.seed_source, None,
                "Node {:?} should have seed_source None",
                r.node
            );
        }
    }

    // --- Step snapshots ---

    #[test]
    fn step_snapshot_count() {
        let graph = build_chain(4, 1.0);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let snapshots = engine.activate_with_steps(&seeds, &config).unwrap();
        assert_eq!(snapshots.len(), 3 + 2); // max_steps + 2
    }

    #[test]
    fn step_snapshot_final_flag() {
        let graph = build_chain(3, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(2)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let snapshots = engine.activate_with_steps(&seeds, &config).unwrap();
        for snap in &snapshots[..snapshots.len() - 1] {
            assert!(!snap.is_final);
        }
        assert!(snapshots.last().unwrap().is_final);
    }

    #[test]
    fn step_snapshot_step0_contains_seeds() {
        let graph = build_chain(5, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(2)
            .min_activation(0.5)
            .build();
        let seeds = vec![
            Seed {
                node: NodeId::new(0),
                activation: act(0.3),
                source: None,
            },
            Seed {
                node: NodeId::new(1),
                activation: act(0.9),
                source: None,
            },
        ];
        let snapshots = engine.activate_with_steps(&seeds, &config).unwrap();
        let step0 = &snapshots[0];
        assert_eq!(step0.step, 0);
        // Step 0 should contain ALL seeded nodes regardless of min_activation
        let step0_nodes: Vec<u32> = step0.activations.iter().map(|(n, _)| n.get()).collect();
        assert!(step0_nodes.contains(&0));
        assert!(step0_nodes.contains(&1));
    }

    #[test]
    fn step_snapshot_final_matches_activate() {
        let graph = build_chain(4, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        let snapshots = engine.activate_with_steps(&seeds, &config).unwrap();
        let final_snap = snapshots.last().unwrap();
        assert!(final_snap.is_final);
        assert_eq!(final_snap.activations.len(), results.len());
        for (i, (node, act_val)) in final_snap.activations.iter().enumerate() {
            assert_eq!(*node, results[i].node);
            assert!(
                (act_val - results[i].activation.get()).abs() < 1e-15,
                "Mismatch at {}: {} vs {}",
                i,
                act_val,
                results[i].activation.get()
            );
        }
    }

    #[test]
    fn step_snapshot_filters_by_min_activation() {
        let graph = build_chain(5, 0.1);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(2)
            .min_activation(0.5)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let snapshots = engine.activate_with_steps(&seeds, &config).unwrap();
        // Steps 1..max_steps should only include nodes >= min_activation
        for snap in &snapshots[1..snapshots.len() - 1] {
            for &(_, a) in &snap.activations {
                assert!(a >= 0.5, "Activation {} below min_activation", a);
            }
        }
    }

    // --- Temporal decay ---

    #[test]
    fn temporal_weight_no_decay_returns_base() {
        assert_eq!(temporal_weight(0.8, None, 0.01, 100.0), 0.8);
    }

    #[test]
    fn temporal_weight_with_decay_24h() {
        let result = temporal_weight(1.0, Some(76.0), 0.01, 100.0);
        // elapsed = 24h, exp(-0.01 * 24) ≈ 0.7866
        assert!((result - 0.7866).abs() < 1e-3, "Got {result}");
    }

    #[test]
    fn temporal_weight_no_last_activated_returns_base() {
        assert_eq!(temporal_weight(0.5, None, 0.01, 100.0), 0.5);
    }

    #[test]
    fn temporal_decay_recent_edge_stronger() {
        // Two-node graph: 0->1 via recent edge, 0->2 via old edge
        let edges = vec![
            EdgeInput {
                source: NodeId::new(0),
                target: NodeId::new(1),
                weight: weight(0.8),
                kind: EdgeKind::Positive,
                last_activated: Some(99.0), // recent: 1h ago
            },
            EdgeInput {
                source: NodeId::new(0),
                target: NodeId::new(2),
                weight: weight(0.8),
                kind: EdgeKind::Positive,
                last_activated: Some(0.0), // old: 100h ago
            },
        ];
        let graph = CsrGraph::build(3, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .min_activation(0.001)
            .temporal_decay_rate(0.01)
            .current_time(100.0)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();
        let a1 = results
            .iter()
            .find(|r| r.node == NodeId::new(1))
            .map(|r| r.activation.get())
            .unwrap_or(0.0);
        let a2 = results
            .iter()
            .find(|r| r.node == NodeId::new(2))
            .map(|r| r.activation.get())
            .unwrap_or(0.0);
        assert!(
            a1 > a2,
            "Recent edge node ({a1}) should be stronger than old edge node ({a2})"
        );
    }

    #[test]
    fn temporal_decay_validation_error() {
        let graph = build_chain(3, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .temporal_decay_rate(0.01)
            // current_time not set -> should error
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        assert!(engine.activate(&seeds, &config).is_err());
    }

    // --- Dynamic affinity ---

    #[test]
    fn dynamic_affinity_activates_isolated_node() {
        // Chain: 0 -- 1 -- 2, node 3 isolated
        // Node 2 and node 3 share tags -> virtual edge
        let edges = vec![
            EdgeInput {
                source: NodeId::new(0),
                target: NodeId::new(1),
                weight: weight(0.8),
                kind: EdgeKind::Positive,
                last_activated: None,
            },
            EdgeInput {
                source: NodeId::new(1),
                target: NodeId::new(2),
                weight: weight(0.8),
                kind: EdgeKind::Positive,
                last_activated: None,
            },
        ];
        let graph = CsrGraph::build(4, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];

        // Without affinity: node 3 should have no activation
        let results_no_aff = engine.activate(&seeds, &config).unwrap();
        let a3_none = results_no_aff
            .iter()
            .find(|r| r.node == NodeId::new(3))
            .map(|r| r.activation.get())
            .unwrap_or(0.0);
        assert!(
            a3_none < 0.001,
            "Node 3 should be inactive without affinity"
        );

        // With affinity: node 3 should get activation via virtual edge from node 2
        let tag_sets = vec![
            vec![],                                            // node 0
            vec![],                                            // node 1
            vec![TagId::new(1), TagId::new(2), TagId::new(3)], // node 2
            vec![TagId::new(1), TagId::new(2), TagId::new(3)], // node 3: same tags as node 2
        ];
        let aff_config = AffinityConfig::builder()
            .min_jaccard(0.3)
            .max_weight(0.8)
            .build();
        let results_aff = engine
            .activate_with_affinity(&seeds, &config, &tag_sets, &aff_config)
            .unwrap();
        let a3_aff = results_aff
            .iter()
            .find(|r| r.node == NodeId::new(3))
            .map(|r| r.activation.get())
            .unwrap_or(0.0);
        assert!(
            a3_aff > 0.001,
            "Node 3 should be active with affinity, got {a3_aff}"
        );
    }

    #[test]
    fn dynamic_affinity_separate_denominator() {
        // Node 0 -> Node 1 (real), Node 0 -> Node 2 (virtual via tags)
        // Virtual edges should use their own denominator
        let edges = vec![EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
            last_activated: None,
        }];
        let graph = CsrGraph::build(3, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];

        let tag_sets = vec![
            vec![TagId::new(1), TagId::new(2)], // node 0
            vec![],                             // node 1
            vec![TagId::new(1), TagId::new(2)], // node 2: same tags as node 0
        ];
        let aff_config = AffinityConfig::builder()
            .min_jaccard(0.3)
            .max_weight(0.5)
            .build();
        let results = engine
            .activate_with_affinity(&seeds, &config, &tag_sets, &aff_config)
            .unwrap();

        // Both nodes should have activation
        assert!(results.iter().any(|r| r.node == NodeId::new(1)));
        assert!(results.iter().any(|r| r.node == NodeId::new(2)));
    }

    #[test]
    fn dynamic_affinity_empty_tags_no_virtual_edges() {
        let graph = build_chain(3, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];

        // All empty tag sets -> no virtual edges -> same as activate()
        let tag_sets: Vec<Vec<TagId>> = vec![vec![], vec![], vec![]];
        let aff_config = AffinityConfig::builder().build();
        let results_aff = engine
            .activate_with_affinity(&seeds, &config, &tag_sets, &aff_config)
            .unwrap();
        let results_plain = engine.activate(&seeds, &config).unwrap();
        assert_eq!(results_aff.len(), results_plain.len());
    }

    #[test]
    fn dynamic_affinity_wrong_tag_sets_len_errors() {
        let graph = build_chain(3, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];

        // tag_sets length doesn't match num_nodes
        let tag_sets: Vec<Vec<TagId>> = vec![vec![], vec![]]; // 2 instead of 3
        let aff_config = AffinityConfig::builder().build();
        assert!(
            engine
                .activate_with_affinity(&seeds, &config, &tag_sets, &aff_config)
                .is_err()
        );
    }

    #[test]
    fn activate_with_steps_and_affinity_final_matches() {
        let edges = vec![EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
            last_activated: None,
        }];
        let graph = CsrGraph::build(3, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(2)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        }];
        let tag_sets = vec![vec![TagId::new(1)], vec![], vec![TagId::new(1)]];
        let aff_config = AffinityConfig::builder()
            .min_jaccard(0.3)
            .max_weight(0.5)
            .build();

        let results = engine
            .activate_with_affinity(&seeds, &config, &tag_sets, &aff_config)
            .unwrap();
        let snapshots = engine
            .activate_with_steps_and_affinity(&seeds, &config, &tag_sets, &aff_config)
            .unwrap();

        let final_snap = snapshots.last().unwrap();
        assert!(final_snap.is_final);
        assert_eq!(final_snap.activations.len(), results.len());
        for (i, (node, act_val)) in final_snap.activations.iter().enumerate() {
            assert_eq!(*node, results[i].node);
            assert!(
                (act_val - results[i].activation.get()).abs() < 1e-15,
                "Mismatch at {i}: {act_val} vs {}",
                results[i].activation.get()
            );
        }
    }
}
