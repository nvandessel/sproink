use criterion::{Criterion, black_box, criterion_group, criterion_main};
use sproink::*;

fn build_chain(n: u32) -> CsrGraph {
    let edges: Vec<EdgeInput> = (0..n - 1)
        .map(|i| EdgeInput {
            source: NodeId(i),
            target: NodeId(i + 1),
            weight: EdgeWeight::new(0.8).unwrap(),
            kind: EdgeKind::Positive,
        })
        .collect();
    CsrGraph::build(n, edges)
}

fn build_random_graph(num_nodes: u32, edges_per_node: u32) -> CsrGraph {
    let mut edges = Vec::new();
    for i in 0..num_nodes {
        for j in 0..edges_per_node {
            let target = (i + j * 7 + 13) % num_nodes;
            if target != i {
                edges.push(EdgeInput {
                    source: NodeId(i),
                    target: NodeId(target),
                    weight: EdgeWeight::new(0.5 + 0.3 * ((i + j) % 3) as f64 / 2.0).unwrap(),
                    kind: EdgeKind::Positive,
                });
            }
        }
    }
    CsrGraph::build(num_nodes, edges)
}

fn bench_propagation(c: &mut Criterion) {
    let g100 = build_chain(100);
    let e100 = Engine::new(g100);
    let config = PropagationConfig::builder().build();
    let seeds = vec![Seed {
        node: NodeId(0),
        activation: Activation::new(1.0).unwrap(),
    }];

    c.bench_function("chain_100", |b| {
        b.iter(|| e100.activate(black_box(&seeds), black_box(&config)))
    });

    let g1k = build_random_graph(1_000, 5);
    let e1k = Engine::new(g1k);
    c.bench_function("random_1k_5e", |b| {
        b.iter(|| e1k.activate(black_box(&seeds), black_box(&config)))
    });

    let g10k = build_random_graph(10_000, 10);
    let e10k = Engine::new(g10k);
    c.bench_function("random_10k_10e", |b| {
        b.iter(|| e10k.activate(black_box(&seeds), black_box(&config)))
    });
}

criterion_group!(benches, bench_propagation);
criterion_main!(benches);
