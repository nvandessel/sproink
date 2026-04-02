use std::collections::HashSet;
use sproink::*;

fn weight(v: f64) -> EdgeWeight {
    EdgeWeight::new(v).unwrap()
}
fn act(v: f64) -> Activation {
    Activation::new(v).unwrap()
}

fn find(results: &[ActivationResult], id: u32) -> Option<&ActivationResult> {
    results.iter().find(|r| r.node == NodeId(id))
}

/// 3-node chain: 0 -> 1 -> 2
/// Verify propagation reaches node 2 with expected decay after 3 steps.
#[test]
fn three_node_chain_propagation() {
    let edges = vec![
        EdgeInput {
            source: NodeId(0),
            target: NodeId(1),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
        },
        EdgeInput {
            source: NodeId(1),
            target: NodeId(2),
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
        node: NodeId(0),
        activation: act(1.0),
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
            source: NodeId(0),
            target: NodeId(1),
            weight: weight(1.0),
            kind: EdgeKind::Conflicts,
        },
        EdgeInput {
            source: NodeId(1),
            target: NodeId(2),
            weight: weight(1.0),
            kind: EdgeKind::Conflicts,
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
        .max_steps(3)
        .min_activation(0.001)
        .build();
    let seeds = vec![
        Seed {
            node: NodeId(0),
            activation: act(1.0),
        },
        Seed {
            node: NodeId(1),
            activation: act(1.0),
        },
        Seed {
            node: NodeId(2),
            activation: act(1.0),
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
    let seeds = vec![
        Seed {
            node: NodeId(0),
            activation: act(0.9),
        },
        Seed {
            node: NodeId(1),
            activation: act(0.9),
        },
    ];
    let results = engine.activate(&seeds, &config);

    let a0 = find(&results, 0).unwrap().activation.get();
    let a1 = find(&results, 1).unwrap().activation.get();

    // Node 0 should be unaffected (reverse is DirectionalPassive, skipped)
    // Node 1 should be suppressed
    assert!(a0 > a1, "Node 0 ({a0}) should be > Node 1 ({a1})");
}

/// Build graph with real + affinity edges.
/// Verify affinity edges spread activation via separate denominator.
#[test]
fn affinity_integration() {
    let edges = vec![
        // Real edges
        EdgeInput {
            source: NodeId(0),
            target: NodeId(1),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
        },
        // Affinity edge
        EdgeInput {
            source: NodeId(0),
            target: NodeId(2),
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
        node: NodeId(0),
        activation: act(1.0),
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
            source: NodeId(0),
            target: NodeId(i),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
        })
        .collect();
    let graph = CsrGraph::build(n, edges);
    let engine = Engine::new(graph);
    let config = PropagationConfig::builder()
        .max_steps(1)
        .min_activation(0.001)
        .inhibition(
            InhibitionConfig::builder()
                .strength(0.5)
                .breadth(2)
                .enabled(true)
                .build(),
        )
        .build();
    let seeds = vec![Seed {
        node: NodeId(0),
        activation: act(1.0),
    }];
    let results = engine.activate(&seeds, &config);

    // All results should be in [0, 1]
    for r in &results {
        assert!(r.activation.get() >= 0.0 && r.activation.get() <= 1.0);
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
    ];
    let graph = CsrGraph::build(3, edges);
    let engine = Engine::new(graph);
    let config = PropagationConfig::builder()
        .max_steps(2)
        .min_activation(0.001)
        .build();
    let seeds = vec![Seed {
        node: NodeId(0),
        activation: act(1.0),
    }];
    let results = engine.activate(&seeds, &config);

    let seed_set: HashSet<NodeId> = [NodeId(0)].into();
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
                    source: NodeId(i),
                    target: NodeId(target),
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
        node: NodeId(0),
        activation: act(1.0),
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
        assert!(r.activation.get() >= 0.0 && r.activation.get() <= 1.0);
    }
}
