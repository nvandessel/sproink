use criterion::{Criterion, criterion_group, criterion_main};
use sproink::engine::{Config, Engine, Seed};
use sproink::graph::{CsrGraph, EdgeInput, EdgeKind};

fn build_chain(n: u32) -> CsrGraph {
    let edges: Vec<EdgeInput> = (0..n - 1)
        .map(|i| EdgeInput {
            source: i,
            target: i + 1,
            weight: 0.8,
            kind: EdgeKind::Positive,
        })
        .collect();
    CsrGraph::build(n, &edges)
}

fn build_random_graph(num_nodes: u32, edges_per_node: u32) -> CsrGraph {
    let mut edges = Vec::new();
    for i in 0..num_nodes {
        for j in 0..edges_per_node {
            let target = (i + j + 1) % num_nodes;
            if target != i {
                edges.push(EdgeInput {
                    source: i,
                    target,
                    weight: 0.8,
                    kind: EdgeKind::Positive,
                });
            }
        }
    }
    CsrGraph::build(num_nodes, &edges)
}

fn bench_propagation(c: &mut Criterion) {
    let g_100 = build_chain(100);
    let g_1000 = build_random_graph(1000, 5);
    let g_10000 = build_random_graph(10000, 10);

    c.bench_function("chain_100", |b| {
        let engine = Engine::new(&g_100, Config::default());
        b.iter(|| engine.activate(&[Seed { node: 0, activation: 1.0 }]));
    });

    c.bench_function("random_1k_5e", |b| {
        let engine = Engine::new(&g_1000, Config::default());
        b.iter(|| engine.activate(&[Seed { node: 0, activation: 1.0 }]));
    });

    c.bench_function("random_10k_10e", |b| {
        let engine = Engine::new(&g_10000, Config::default());
        b.iter(|| engine.activate(&[Seed { node: 0, activation: 1.0 }]));
    });
}

criterion_group!(benches, bench_propagation);
criterion_main!(benches);
