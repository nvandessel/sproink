use crate::types::{EdgeWeight, NodeId};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    Positive = 0,
    Conflicts = 1,
    DirectionalSuppressive = 2,
    DirectionalPassive = 3,
    FeatureAffinity = 4,
}

impl EdgeKind {
    fn reverse(self) -> Self {
        match self {
            EdgeKind::DirectionalSuppressive => EdgeKind::DirectionalPassive,
            other => other,
        }
    }
}

pub struct EdgeData {
    pub target: NodeId,
    pub weight: EdgeWeight,
    pub kind: EdgeKind,
}

pub struct EdgeInput {
    pub source: NodeId,
    pub target: NodeId,
    pub weight: EdgeWeight,
    pub kind: EdgeKind,
}

pub trait Graph: Send + Sync {
    fn num_nodes(&self) -> u32;
    fn neighbors(&self, node: NodeId) -> &[EdgeData];
}

pub struct CsrGraph {
    num_nodes: u32,
    row_offsets: Vec<u32>,
    edges: Vec<EdgeData>,
}

impl CsrGraph {
    pub fn build(num_nodes: u32, inputs: Vec<EdgeInput>) -> Self {
        // Count edges per node (each input produces 2 stored edges)
        let mut counts = vec![0u32; num_nodes as usize];
        for e in &inputs {
            counts[e.source.index()] += 1;
            counts[e.target.index()] += 1;
        }

        // Build prefix-sum row_offsets
        let mut row_offsets = vec![0u32; num_nodes as usize + 1];
        for i in 0..num_nodes as usize {
            row_offsets[i + 1] = row_offsets[i] + counts[i];
        }

        let total_edges = row_offsets[num_nodes as usize] as usize;

        // Use write positions to place edges
        let mut write_pos = row_offsets[..num_nodes as usize].to_vec();

        // Initialize edges vec with placeholder data
        let mut edges: Vec<EdgeData> = (0..total_edges)
            .map(|_| EdgeData {
                target: NodeId(0),
                weight: EdgeWeight::new_unchecked(0.0),
                kind: EdgeKind::Positive,
            })
            .collect();

        for e in &inputs {
            // Forward edge: source -> target
            let pos = write_pos[e.source.index()] as usize;
            edges[pos] = EdgeData {
                target: e.target,
                weight: e.weight,
                kind: e.kind,
            };
            write_pos[e.source.index()] += 1;

            // Reverse edge: target -> source
            let pos = write_pos[e.target.index()] as usize;
            edges[pos] = EdgeData {
                target: e.source,
                weight: e.weight,
                kind: e.kind.reverse(),
            };
            write_pos[e.target.index()] += 1;
        }

        CsrGraph {
            num_nodes,
            row_offsets,
            edges,
        }
    }
}

impl<G: Graph> Graph for &G {
    fn num_nodes(&self) -> u32 {
        (*self).num_nodes()
    }

    fn neighbors(&self, node: NodeId) -> &[EdgeData] {
        (*self).neighbors(node)
    }
}

impl Graph for CsrGraph {
    fn num_nodes(&self) -> u32 {
        self.num_nodes
    }

    fn neighbors(&self, node: NodeId) -> &[EdgeData] {
        let start = self.row_offsets[node.index()] as usize;
        let end = self.row_offsets[node.index() + 1] as usize;
        &self.edges[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weight(v: f64) -> EdgeWeight {
        EdgeWeight::new(v).unwrap()
    }

    #[test]
    fn empty_graph() {
        let g = CsrGraph::build(5, vec![]);
        assert_eq!(g.num_nodes(), 5);
        for i in 0..5 {
            assert!(g.neighbors(NodeId(i)).is_empty());
        }
    }

    #[test]
    fn single_positive_edge_stored_bidirectionally() {
        let edges = vec![EdgeInput {
            source: NodeId(0),
            target: NodeId(1),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
        }];
        let g = CsrGraph::build(2, edges);
        let n0 = g.neighbors(NodeId(0));
        assert_eq!(n0.len(), 1);
        assert_eq!(n0[0].target, NodeId(1));
        assert_eq!(n0[0].kind, EdgeKind::Positive);
        let n1 = g.neighbors(NodeId(1));
        assert_eq!(n1.len(), 1);
        assert_eq!(n1[0].target, NodeId(0));
        assert_eq!(n1[0].kind, EdgeKind::Positive);
    }

    #[test]
    fn directional_suppressive_reverse_is_passive() {
        let edges = vec![EdgeInput {
            source: NodeId(0),
            target: NodeId(1),
            weight: weight(0.5),
            kind: EdgeKind::DirectionalSuppressive,
        }];
        let g = CsrGraph::build(2, edges);
        let n0 = g.neighbors(NodeId(0));
        assert_eq!(n0[0].kind, EdgeKind::DirectionalSuppressive);
        let n1 = g.neighbors(NodeId(1));
        assert_eq!(n1[0].kind, EdgeKind::DirectionalPassive);
    }

    #[test]
    fn conflict_edges_are_symmetric() {
        let edges = vec![EdgeInput {
            source: NodeId(0),
            target: NodeId(1),
            weight: weight(0.6),
            kind: EdgeKind::Conflicts,
        }];
        let g = CsrGraph::build(2, edges);
        assert_eq!(g.neighbors(NodeId(0))[0].kind, EdgeKind::Conflicts);
        assert_eq!(g.neighbors(NodeId(1))[0].kind, EdgeKind::Conflicts);
    }

    #[test]
    fn feature_affinity_edges_are_symmetric() {
        let edges = vec![EdgeInput {
            source: NodeId(0),
            target: NodeId(1),
            weight: weight(0.3),
            kind: EdgeKind::FeatureAffinity,
        }];
        let g = CsrGraph::build(2, edges);
        assert_eq!(g.neighbors(NodeId(0))[0].kind, EdgeKind::FeatureAffinity);
        assert_eq!(g.neighbors(NodeId(1))[0].kind, EdgeKind::FeatureAffinity);
    }

    #[test]
    fn multiple_edges_from_same_node() {
        let edges = vec![
            EdgeInput {
                source: NodeId(0),
                target: NodeId(1),
                weight: weight(0.5),
                kind: EdgeKind::Positive,
            },
            EdgeInput {
                source: NodeId(0),
                target: NodeId(2),
                weight: weight(0.7),
                kind: EdgeKind::Positive,
            },
        ];
        let g = CsrGraph::build(3, edges);
        assert_eq!(g.neighbors(NodeId(0)).len(), 2);
        assert_eq!(g.neighbors(NodeId(1)).len(), 1);
        assert_eq!(g.neighbors(NodeId(2)).len(), 1);
    }

    #[test]
    fn self_loop_stored_correctly() {
        let edges = vec![EdgeInput {
            source: NodeId(0),
            target: NodeId(0),
            weight: weight(0.5),
            kind: EdgeKind::Positive,
        }];
        let g = CsrGraph::build(1, edges);
        let n0 = g.neighbors(NodeId(0));
        assert_eq!(n0.len(), 2);
        assert!(n0.iter().all(|e| e.target == NodeId(0)));
    }

    #[test]
    fn graph_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CsrGraph>();
    }
}
