#[allow(dead_code)]
#[path = "../tests/common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use sproink::*;
use std::hint::black_box;

fn build_random_graph(num_nodes: u32, edges_per_node: u32) -> CsrGraph {
    let mut edges = Vec::new();
    for i in 0..num_nodes {
        for j in 0..edges_per_node {
            let target = (i + j * 7 + 13) % num_nodes;
            if target != i {
                edges.push(EdgeInput {
                    source: NodeId::new(i),
                    target: NodeId::new(target),
                    weight: EdgeWeight::new(0.5 + 0.3 * ((i + j) % 3) as f64 / 2.0).unwrap(),
                    kind: EdgeKind::Positive,
                    last_activated: None,
                });
            }
        }
    }
    CsrGraph::build(num_nodes, &edges).unwrap()
}

fn build_mixed_edge_graph(num_nodes: u32, edges_per_node: u32) -> CsrGraph {
    let kinds = [
        EdgeKind::Positive,
        EdgeKind::Conflicts,
        EdgeKind::DirectionalSuppressive,
        EdgeKind::FeatureAffinity,
    ];
    let mut edges = Vec::new();
    for i in 0..num_nodes {
        for j in 0..edges_per_node {
            let target = (i + j * 7 + 13) % num_nodes;
            if target != i {
                let kind = kinds[((i + j) % 4) as usize];
                edges.push(EdgeInput {
                    source: NodeId::new(i),
                    target: NodeId::new(target),
                    weight: EdgeWeight::new(0.5 + 0.3 * ((i + j) % 3) as f64 / 2.0).unwrap(),
                    kind,
                    last_activated: None,
                });
            }
        }
    }
    CsrGraph::build(num_nodes, &edges).unwrap()
}

fn bench_propagation(c: &mut Criterion) {
    let g100 = common::build_chain(100, 0.8);
    let e100 = Engine::new(g100);
    let config = PropagationConfig::builder().build();
    let seeds = vec![Seed {
        node: NodeId::new(0),
        activation: Activation::new(1.0).unwrap(),
        source: None,
    }];

    c.bench_function("chain_100", |b| {
        b.iter(|| {
            e100.activate(black_box(&seeds), black_box(&config))
                .unwrap()
        })
    });

    let g1k = build_random_graph(1_000, 5);
    let e1k = Engine::new(g1k);
    c.bench_function("random_1k_5e", |b| {
        b.iter(|| e1k.activate(black_box(&seeds), black_box(&config)).unwrap())
    });

    let g10k = build_random_graph(10_000, 10);
    let e10k = Engine::new(g10k);
    c.bench_function("random_10k_10e", |b| {
        b.iter(|| {
            e10k.activate(black_box(&seeds), black_box(&config))
                .unwrap()
        })
    });

    // T14: Mixed edge kinds benchmarks
    let g_mixed_1k = build_mixed_edge_graph(1_000, 5);
    let e_mixed_1k = Engine::new(g_mixed_1k);
    c.bench_function("mixed_1k_5e", |b| {
        b.iter(|| {
            e_mixed_1k
                .activate(black_box(&seeds), black_box(&config))
                .unwrap()
        })
    });

    let g_mixed_10k = build_mixed_edge_graph(10_000, 10);
    let e_mixed_10k = Engine::new(g_mixed_10k);
    c.bench_function("mixed_10k_10e", |b| {
        b.iter(|| {
            e_mixed_10k
                .activate(black_box(&seeds), black_box(&config))
                .unwrap()
        })
    });

    // Multiple seeds benchmark
    let multi_seeds = vec![
        Seed {
            node: NodeId::new(0),
            activation: Activation::new(1.0).unwrap(),
            source: Some(0),
        },
        Seed {
            node: NodeId::new(500),
            activation: Activation::new(0.8).unwrap(),
            source: Some(1),
        },
        Seed {
            node: NodeId::new(999),
            activation: Activation::new(0.9).unwrap(),
            source: Some(2),
        },
    ];
    c.bench_function("mixed_1k_5e_3seeds", |b| {
        let g = build_mixed_edge_graph(1_000, 5);
        let e = Engine::new(g);
        b.iter(|| {
            e.activate(black_box(&multi_seeds), black_box(&config))
                .unwrap()
        })
    });
}

criterion_group!(benches, bench_propagation);
criterion_main!(benches);
