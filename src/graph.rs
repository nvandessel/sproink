//! CSR (Compressed Sparse Row) graph representation and edge types.
//!
//! The graph stores edges bidirectionally: each [`EdgeInput`] produces a forward
//! and reverse edge. Directional-suppressive edges reverse to directional-passive.

use std::fmt;

use crate::types::{EdgeWeight, NodeId};

/// The kind of relationship an edge represents.
///
/// Edge kinds determine how activation energy is propagated or suppressed
/// during spreading activation.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EdgeKind {
    /// Standard excitatory edge — energy flows and accumulates via max.
    Positive = 0,
    /// Mutual suppression — both endpoints reduce each other's activation.
    Conflicts = 1,
    /// One-way suppression from source to target (reverse becomes [`DirectionalPassive`](EdgeKind::DirectionalPassive)).
    DirectionalSuppressive = 2,
    /// Passive end of a directional-suppressive edge; no energy propagates.
    DirectionalPassive = 3,
    /// Tag-similarity affinity edge; behaves like [`Positive`](EdgeKind::Positive) during propagation.
    FeatureAffinity = 4,
}

impl fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeKind::Positive => write!(f, "positive"),
            EdgeKind::Conflicts => write!(f, "conflicts"),
            EdgeKind::DirectionalSuppressive => write!(f, "directional-suppressive"),
            EdgeKind::DirectionalPassive => write!(f, "directional-passive"),
            EdgeKind::FeatureAffinity => write!(f, "feature-affinity"),
        }
    }
}

impl EdgeKind {
    fn reverse(self) -> Self {
        match self {
            EdgeKind::DirectionalSuppressive => EdgeKind::DirectionalPassive,
            other => other,
        }
    }
}

/// A stored edge in the CSR graph (target endpoint only; source is implicit).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeData {
    /// The neighbor node this edge points to.
    pub target: NodeId,
    /// Edge weight in `[0.0, 1.0]`.
    pub weight: EdgeWeight,
    /// The kind of relationship this edge represents.
    pub kind: EdgeKind,
    /// Optional timestamp of last activation (used for temporal decay).
    pub last_activated: Option<f64>,
}

/// An edge to insert when building a graph.
///
/// Each `EdgeInput` produces two stored edges (forward and reverse).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeInput {
    /// Source node of the edge.
    pub source: NodeId,
    /// Target node of the edge.
    pub target: NodeId,
    /// Edge weight in `[0.0, 1.0]`.
    pub weight: EdgeWeight,
    /// The kind of relationship.
    pub kind: EdgeKind,
    /// Optional timestamp of last activation (for temporal decay).
    pub last_activated: Option<f64>,
}

/// Trait abstracting graph access for the propagation engine.
pub trait Graph: Send + Sync {
    /// Returns the total number of nodes in the graph.
    fn num_nodes(&self) -> u32;
    /// Returns the outgoing edges for the given node.
    fn neighbors(&self, node: NodeId) -> &[EdgeData];
}

/// Compressed Sparse Row graph — the default high-performance graph backend.
///
/// Built once via [`CsrGraph::build()`] and then immutably shared with the engine.
///
/// # Examples
///
/// ```
/// use sproink::{CsrGraph, EdgeInput, EdgeKind, EdgeWeight, Graph, NodeId};
///
/// let graph = CsrGraph::build(2, vec![EdgeInput {
///     source: NodeId::new(0),
///     target: NodeId::new(1),
///     weight: EdgeWeight::new(0.5).unwrap(),
///     kind: EdgeKind::Positive,
///     last_activated: None,
/// }]);
///
/// assert_eq!(graph.num_nodes(), 2);
/// ```
pub struct CsrGraph {
    num_nodes: u32,
    row_offsets: Vec<u32>,
    edges: Vec<EdgeData>,
}

impl fmt::Debug for CsrGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CsrGraph")
            .field("num_nodes", &self.num_nodes)
            .field("num_edges", &self.edges.len())
            .finish()
    }
}

impl CsrGraph {
    /// Builds a CSR graph from a list of edge inputs.
    ///
    /// Each input edge is stored bidirectionally. Directional-suppressive edges
    /// have their reverse stored as directional-passive.
    #[must_use]
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
                target: NodeId::new(0),
                weight: EdgeWeight::new_unchecked(0.0),
                kind: EdgeKind::Positive,
                last_activated: None,
            })
            .collect();

        for e in &inputs {
            // Forward edge: source -> target
            let pos = write_pos[e.source.index()] as usize;
            edges[pos] = EdgeData {
                target: e.target,
                weight: e.weight,
                kind: e.kind,
                last_activated: e.last_activated,
            };
            write_pos[e.source.index()] += 1;

            // Reverse edge: target -> source
            let pos = write_pos[e.target.index()] as usize;
            edges[pos] = EdgeData {
                target: e.source,
                weight: e.weight,
                kind: e.kind.reverse(),
                last_activated: e.last_activated,
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
            assert!(g.neighbors(NodeId::new(i)).is_empty());
        }
    }

    #[test]
    fn single_positive_edge_stored_bidirectionally() {
        let edges = vec![EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(0.8),
            kind: EdgeKind::Positive,
            last_activated: None,
        }];
        let g = CsrGraph::build(2, edges);
        let n0 = g.neighbors(NodeId::new(0));
        assert_eq!(n0.len(), 1);
        assert_eq!(n0[0].target, NodeId::new(1));
        assert_eq!(n0[0].kind, EdgeKind::Positive);
        let n1 = g.neighbors(NodeId::new(1));
        assert_eq!(n1.len(), 1);
        assert_eq!(n1[0].target, NodeId::new(0));
        assert_eq!(n1[0].kind, EdgeKind::Positive);
    }

    #[test]
    fn directional_suppressive_reverse_is_passive() {
        let edges = vec![EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(0.5),
            kind: EdgeKind::DirectionalSuppressive,
            last_activated: None,
        }];
        let g = CsrGraph::build(2, edges);
        let n0 = g.neighbors(NodeId::new(0));
        assert_eq!(n0[0].kind, EdgeKind::DirectionalSuppressive);
        let n1 = g.neighbors(NodeId::new(1));
        assert_eq!(n1[0].kind, EdgeKind::DirectionalPassive);
    }

    #[test]
    fn conflict_edges_are_symmetric() {
        let edges = vec![EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(0.6),
            kind: EdgeKind::Conflicts,
            last_activated: None,
        }];
        let g = CsrGraph::build(2, edges);
        assert_eq!(g.neighbors(NodeId::new(0))[0].kind, EdgeKind::Conflicts);
        assert_eq!(g.neighbors(NodeId::new(1))[0].kind, EdgeKind::Conflicts);
    }

    #[test]
    fn feature_affinity_edges_are_symmetric() {
        let edges = vec![EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight: weight(0.3),
            kind: EdgeKind::FeatureAffinity,
            last_activated: None,
        }];
        let g = CsrGraph::build(2, edges);
        assert_eq!(
            g.neighbors(NodeId::new(0))[0].kind,
            EdgeKind::FeatureAffinity
        );
        assert_eq!(
            g.neighbors(NodeId::new(1))[0].kind,
            EdgeKind::FeatureAffinity
        );
    }

    #[test]
    fn multiple_edges_from_same_node() {
        let edges = vec![
            EdgeInput {
                source: NodeId::new(0),
                target: NodeId::new(1),
                weight: weight(0.5),
                kind: EdgeKind::Positive,
                last_activated: None,
            },
            EdgeInput {
                source: NodeId::new(0),
                target: NodeId::new(2),
                weight: weight(0.7),
                kind: EdgeKind::Positive,
                last_activated: None,
            },
        ];
        let g = CsrGraph::build(3, edges);
        assert_eq!(g.neighbors(NodeId::new(0)).len(), 2);
        assert_eq!(g.neighbors(NodeId::new(1)).len(), 1);
        assert_eq!(g.neighbors(NodeId::new(2)).len(), 1);
    }

    #[test]
    fn self_loop_stored_correctly() {
        let edges = vec![EdgeInput {
            source: NodeId::new(0),
            target: NodeId::new(0),
            weight: weight(0.5),
            kind: EdgeKind::Positive,
            last_activated: None,
        }];
        let g = CsrGraph::build(1, edges);
        let n0 = g.neighbors(NodeId::new(0));
        assert_eq!(n0.len(), 2);
        assert!(n0.iter().all(|e| e.target == NodeId::new(0)));
    }

    #[test]
    fn graph_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CsrGraph>();
    }
}
