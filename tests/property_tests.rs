mod common;

use proptest::prelude::*;
use sproink::*;

use common::build_chain;

fn build_random_graph(num_nodes: u32, num_edges: usize) -> CsrGraph {
    let mut edges = Vec::new();
    for i in 0..num_edges {
        let source = (i as u32) % num_nodes;
        let target = ((i as u32) * 7 + 13) % num_nodes;
        if source != target {
            edges.push(EdgeInput {
                source: NodeId::new(source),
                target: NodeId::new(target),
                weight: EdgeWeight::new(0.5).unwrap(),
                kind: EdgeKind::Positive,
                last_activated: None,
            });
        }
    }
    CsrGraph::build(num_nodes, edges)
}

proptest! {
    /// All output activations must be in [0.0, 1.0].
    #[test]
    fn activation_bounds(
        num_nodes in 2u32..50,
        num_edges in 1usize..100,
        num_seeds in 1usize..5,
    ) {
        let graph = build_random_graph(num_nodes, num_edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let seeds: Vec<Seed> = (0..num_seeds)
            .map(|i| Seed {
                node: NodeId::new((i as u32) % num_nodes),
                activation: Activation::new(0.8).unwrap(),
            source: None,
            })
            .collect();
        let results = engine.activate(&seeds, &config).unwrap();
        for r in &results {
            prop_assert!((0.0..=1.0).contains(&r.activation.get()),
                "Activation {} out of [0,1]", r.activation.get());
        }
    }

    /// Same inputs always produce same outputs (determinism).
    #[test]
    fn deterministic_output(
        num_nodes in 2u32..20,
        num_edges in 1usize..30,
    ) {
        let graph1 = build_random_graph(num_nodes, num_edges);
        let graph2 = build_random_graph(num_nodes, num_edges);
        let config = PropagationConfig::builder().build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: Activation::new(1.0).unwrap(),
            source: None,
        }];
        let r1 = Engine::new(graph1).activate(&seeds, &config).unwrap();
        let r2 = Engine::new(graph2).activate(&seeds, &config).unwrap();
        prop_assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            prop_assert_eq!(a.node, b.node);
            prop_assert!((a.activation.get() - b.activation.get()).abs() < 1e-15);
        }
    }

    /// For positive-only graphs, activation at distance d+1 <= distance d.
    #[test]
    fn monotonic_decay_positive_only(chain_len in 3u32..20) {
        let graph = build_chain(chain_len, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(chain_len)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed {
            node: NodeId::new(0),
            activation: Activation::new(1.0).unwrap(),
            source: None,
        }];
        let results = engine.activate(&seeds, &config).unwrap();

        // Group by distance and check monotonic decay
        let mut by_distance: Vec<(u32, f64)> = results.iter()
            .map(|r| (r.distance, r.activation.get()))
            .collect();
        by_distance.sort_by_key(|&(d, _)| d);

        // Max activation at each distance should be non-increasing
        let mut prev_max = f64::MAX;
        let mut current_dist = 0;
        let mut current_max = 0.0f64;
        for &(d, a) in &by_distance {
            if d != current_dist {
                if current_max > 0.0 {
                    prop_assert!(current_max <= prev_max + 1e-10,
                        "Distance {} has activation {} > previous distance max {}",
                        current_dist, current_max, prev_max);
                    prev_max = current_max;
                }
                current_dist = d;
                current_max = a;
            } else {
                current_max = current_max.max(a);
            }
        }
    }

    /// Repeated Oja updates converge (not NaN/Inf).
    #[test]
    fn oja_convergence(
        w in 0.01f64..0.95,
        a_i in 0.01f64..1.0,
        a_j in 0.01f64..1.0,
        iterations in 1usize..500,
    ) {
        let learner = OjaLearner;
        let config = HebbianConfig::builder().build();
        let act_a = Activation::new(a_i).unwrap();
        let act_b = Activation::new(a_j).unwrap();
        let mut current = EdgeWeight::new(w).unwrap();
        for _ in 0..iterations {
            current = learner.update_weight(current, act_a, act_b, &config);
            prop_assert!(!current.get().is_nan(), "Weight became NaN");
            prop_assert!(!current.get().is_infinite(), "Weight became Inf");
            prop_assert!(current.get() >= config.min_weight);
            prop_assert!(current.get() <= config.max_weight);
        }
    }

    /// Jaccard always in [0.0, 1.0], symmetric, J(A,A)=1.0.
    #[test]
    fn jaccard_properties(
        a_raw in prop::collection::vec(0u32..100, 0..20),
        b_raw in prop::collection::vec(0u32..100, 0..20),
    ) {
        // Deduplicate and sort for valid Jaccard input
        let mut a: Vec<TagId> = a_raw.into_iter().collect::<std::collections::BTreeSet<_>>()
            .into_iter().map(TagId::new).collect();
        let mut b: Vec<TagId> = b_raw.into_iter().collect::<std::collections::BTreeSet<_>>()
            .into_iter().map(TagId::new).collect();
        a.sort();
        b.sort();

        let j = sproink::jaccard_similarity(&a, &b);
        prop_assert!(j >= 0.0 && j <= 1.0, "Jaccard {} out of [0,1]", j);

        // Symmetric
        let j2 = sproink::jaccard_similarity(&b, &a);
        prop_assert!((j - j2).abs() < 1e-15, "Jaccard not symmetric: {} vs {}", j, j2);

        // Self-similarity
        if !a.is_empty() {
            let j_self = sproink::jaccard_similarity(&a, &a);
            prop_assert!((j_self - 1.0).abs() < 1e-10, "J(A,A) = {} != 1.0", j_self);
        }
    }
}
