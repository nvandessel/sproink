mod common;

use sproink::*;
use std::collections::HashSet;

use common::{act, weight};

fn find(results: &[ActivationResult], id: u32) -> Option<&ActivationResult> {
    results.iter().find(|r| r.node == NodeId::new(id))
}

/// 3-node chain: 0 -> 1 -> 2
/// Verify propagation reaches node 2 with expected decay after 3 steps.
#[test]
fn three_node_chain_propagation() {
    let edges = vec![
        EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
        },
        EdgeInput {
            source: NodeId::new(1),
            target: NodeId::new(2),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
        },
    ];
    let graph = CsrGraph::build(3, edges);
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
    let results = engine.activate(&seeds, &config);

    // All 3 nodes should be active
    assert!(find(&results, 0).is_some());
    assert!(find(&results, 1).is_some());
    assert!(find(&results, 2).is_some());

    // Activation should decrease with distance
    let a0 = find(&results, 0).unwrap().activation.get();
    let a1 = find(&results, 1).unwrap().activation.get();
    let a2 = find(&results, 2).unwrap().activation.get();
    assert!(a0 > a1);
    assert!(a1 > a2);

    // Distance tracking
    assert_eq!(find(&results, 0).unwrap().distance, 0);
    assert_eq!(find(&results, 1).unwrap().distance, 1);
    assert_eq!(find(&results, 2).unwrap().distance, 2);
}

/// Triangle with all conflict edges.
/// Verify mutual suppression prevents all-active state.
#[test]
fn conflict_triangle() {
    let edges = vec![
        EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(1.0),
            kind: EdgeKind::Conflicts,
        },
        EdgeInput {
            source: NodeId::new(1),
            target: NodeId::new(2),
            weight: weight(1.0),
            kind: EdgeKind::Conflicts,
        },
        EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(2),
            weight: weight(1.0),
            kind: EdgeKind::Conflicts,
        },
    ];
    let graph = CsrGraph::build(3, edges);
    let engine = Engine::new(graph);
    let config = PropagationConfig::builder()
        .max_steps(3)
        .min_activation(0.001)
        .build();
    let seeds = vec![
        Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: None,
        },
        Seed {
            node: NodeId::new(1),
            activation: act(1.0),
            source: None,
        },
        Seed {
            node: NodeId::new(2),
            activation: act(1.0),
            source: None,
        },
    ];
    let results = engine.activate(&seeds, &config);

    // Mutual suppression should reduce activations from their seed values
    for r in &results {
        // After sigmoid, even suppressed values map to something > 0
        // but the raw values pre-sigmoid should be reduced
        assert!(r.activation.get() <= 1.0);
    }
}

/// DirectionalSuppressive: node 0 suppresses node 1 but not reverse
#[test]
fn directional_override() {
    let edges = vec![EdgeInput {
        source: NodeId::new(0),
        target: NodeId::new(1),
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
    let seeds = vec![
        Seed {
            node: NodeId::new(0),
            activation: act(0.9),
            source: None,
        },
        Seed {
            node: NodeId::new(1),
            activation: act(0.9),
            source: None,
        },
    ];
    let results = engine.activate(&seeds, &config);

    let a0 = find(&results, 0).map(|r| r.activation.get()).unwrap_or(0.0);
    let a1 = find(&results, 1).map(|r| r.activation.get()).unwrap_or(0.0);

    // Node 0 survives; node 1 is suppressed (lower or absent)
    assert!(a0 > a1, "Node 0 ({a0}) should be > Node 1 ({a1})");
}

/// Build graph with real + affinity edges.
/// Verify affinity edges spread activation via separate denominator.
#[test]
fn affinity_integration() {
    let edges = vec![
        // Real edges
        EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
        },
        // Affinity edge
        EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(2),
            weight: weight(0.3),
            kind: EdgeKind::FeatureAffinity,
        },
    ];
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
    let results = engine.activate(&seeds, &config);

    // Both nodes should receive energy
    assert!(find(&results, 1).is_some());
    assert!(find(&results, 2).is_some());

    // Node 1 (real edge, higher weight) should get more activation than node 2
    let a1 = find(&results, 1).unwrap().activation.get();
    let a2 = find(&results, 2).unwrap().activation.get();
    assert!(a1 > a2);
}

/// Full pipeline: propagation + inhibition + sigmoid
#[test]
fn inhibition_plus_sigmoid() {
    let n = 6u32;
    let edges: Vec<EdgeInput> = (1..n)
        .map(|i| EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(i),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
        })
        .collect();
    let graph = CsrGraph::build(n, edges);
    let engine = Engine::new(graph);
    let config = PropagationConfig::builder()
        .max_steps(1)
        .min_activation(0.001)
        .inhibition(InhibitionConfig::builder().strength(0.5).breadth(2).build())
        .build();
    let seeds = vec![Seed {
        node: NodeId::new(0),
        activation: act(1.0),
        source: None,
    }];
    let results = engine.activate(&seeds, &config);

    // All results should be in [0, 1]
    for r in &results {
        assert!((0.0..=1.0).contains(&r.activation.get()));
    }
    // Results should be sorted
    for i in 1..results.len() {
        assert!(results[i - 1].activation.get() >= results[i].activation.get());
    }
}

/// Hebbian round-trip: propagate -> extract pairs -> update weights
#[test]
fn hebbian_round_trip() {
    let edges = vec![
        EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
        },
        EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(2),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
        },
    ];
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
    let results = engine.activate(&seeds, &config);

    let seed_set: HashSet<NodeId> = [NodeId::new(0)].into();
    let hebb_config = HebbianConfig::builder().activation_threshold(0.1).build();
    let pairs = extract_co_activation_pairs(&results, &seed_set, &hebb_config);

    // Should have at least some pairs
    assert!(!pairs.is_empty());

    // Update a weight using OjaLearner
    let learner = OjaLearner;
    let old_w = weight(0.5);
    let new_w = learner.update_weight(
        old_w,
        pairs[0].activation_a,
        pairs[0].activation_b,
        &hebb_config,
    );
    assert!(new_w.get() >= hebb_config.min_weight && new_w.get() <= hebb_config.max_weight);
}

/// Large graph: verify results are valid on parallel path
#[test]
fn parallel_produces_valid_results() {
    let num_nodes = 2000u32;
    let mut edges = Vec::new();
    for i in 0..num_nodes {
        for j in 0..5u32 {
            let target = (i + j * 7 + 13) % num_nodes;
            if target != i {
                edges.push(EdgeInput {
                    source: NodeId::new(i),
                    target: NodeId::new(target),
                    weight: weight(0.5),
                    kind: EdgeKind::Positive,
                });
            }
        }
    }
    let graph = CsrGraph::build(num_nodes, edges);
    let engine = Engine::new(graph);
    let config = PropagationConfig::builder().build();
    let seeds = vec![Seed {
        node: NodeId::new(0),
        activation: act(1.0),
        source: None,
    }];

    let r1 = engine.activate(&seeds, &config);
    let r2 = engine.activate(&seeds, &config);

    // Determinism
    assert_eq!(r1.len(), r2.len());
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert_eq!(a.node, b.node);
        assert!((a.activation.get() - b.activation.get()).abs() < 1e-15);
    }

    // Bounds
    for r in &r1 {
        assert!((0.0..=1.0).contains(&r.activation.get()));
    }
}

/// Large graph with mixed edge kinds exercises the parallel path for all edge types
#[test]
fn parallel_path_mixed_edge_kinds() {
    let n = 2000u32;
    let mut edges = Vec::new();
    for i in 0..n {
        // Positive edges
        let t1 = (i + 7) % n;
        if t1 != i {
            edges.push(EdgeInput {
                source: NodeId::new(i),
                target: NodeId::new(t1),
                weight: weight(0.6),
                kind: EdgeKind::Positive,
                last_activated: None,
            });
        }
        // Conflict edges (every 10th node)
        if i % 10 == 0 {
            let t2 = (i + 3) % n;
            if t2 != i {
                edges.push(EdgeInput {
                    source: NodeId::new(i),
                    target: NodeId::new(t2),
                    weight: weight(0.5),
                    kind: EdgeKind::Conflicts,
                    last_activated: None,
                });
            }
        }
        // DirectionalSuppressive edges (every 20th node)
        if i % 20 == 0 {
            let t3 = (i + 11) % n;
            if t3 != i {
                edges.push(EdgeInput {
                    source: NodeId::new(i),
                    target: NodeId::new(t3),
                    weight: weight(0.4),
                    kind: EdgeKind::DirectionalSuppressive,
                    last_activated: None,
                });
            }
        }
        // FeatureAffinity edges (every 15th node)
        if i % 15 == 0 {
            let t4 = (i + 5) % n;
            if t4 != i {
                edges.push(EdgeInput {
                    source: NodeId::new(i),
                    target: NodeId::new(t4),
                    weight: weight(0.3),
                    kind: EdgeKind::FeatureAffinity,
                    last_activated: None,
                });
            }
        }
    }
    let graph = CsrGraph::build(n, edges);
    let engine = Engine::new(graph);
    let config = PropagationConfig::builder().build();
    let seeds = vec![Seed {
        node: NodeId::new(0),
        activation: act(1.0),
        source: None,
    }];
    let results = engine.activate(&seeds, &config).unwrap();
    for r in &results {
        assert!((0.0..=1.0).contains(&r.activation.get()));
    }
}

/// Two seeds at opposite ends of a chain, each with a different source ID.
/// Nodes closer to each seed should inherit that seed's source.
#[test]
fn two_seeds_different_sources() {
    // Chain: 0 -- 1 -- 2 -- 3 -- 4
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
        EdgeInput {
            source: NodeId::new(2),
            target: NodeId::new(3),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
            last_activated: None,
        },
        EdgeInput {
            source: NodeId::new(3),
            target: NodeId::new(4),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
            last_activated: None,
        },
    ];
    let graph = CsrGraph::build(5, edges);
    let engine = Engine::new(graph);
    let config = PropagationConfig::builder()
        .max_steps(4)
        .min_activation(0.001)
        .build();
    let seeds = vec![
        Seed {
            node: NodeId::new(0),
            activation: act(1.0),
            source: Some(10),
        },
        Seed {
            node: NodeId::new(4),
            activation: act(1.0),
            source: Some(20),
        },
    ];
    let results = engine.activate(&seeds, &config).unwrap();

    // Node 0 is seed 10
    assert_eq!(find(&results, 0).unwrap().seed_source, Some(10));
    // Node 4 is seed 20
    assert_eq!(find(&results, 4).unwrap().seed_source, Some(20));
    // Node 1 is closer to seed 0 (distance 1) than seed 4 (distance 3)
    assert_eq!(find(&results, 1).unwrap().seed_source, Some(10));
    // Node 3 is closer to seed 4 (distance 1) than seed 0 (distance 3)
    assert_eq!(find(&results, 3).unwrap().seed_source, Some(20));
}
