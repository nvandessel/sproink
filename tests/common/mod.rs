use sproink::*;

pub fn build_chain(n: u32, w: f64) -> CsrGraph {
    let edges: Vec<EdgeInput> = (0..n - 1)
        .map(|i| EdgeInput {
            source: NodeId::new(i),
            target: NodeId::new(i + 1),
            weight: EdgeWeight::new(w).unwrap(),
            kind: EdgeKind::Positive,
        })
        .collect();
    CsrGraph::build(n, edges)
}

pub fn weight(v: f64) -> EdgeWeight {
    EdgeWeight::new(v).unwrap()
}

pub fn act(v: f64) -> Activation {
    Activation::new(v).unwrap()
}
