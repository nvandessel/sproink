//! Compressed Sparse Row (CSR) graph for cache-friendly adjacency traversal.

/// Edge kind determines propagation semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EdgeKind {
    /// Normal positive edge — spreads activation.
    Positive = 0,
    /// Symmetric conflict — suppresses in both directions.
    Conflicts = 1,
    /// Directional suppression (overrides, deprecated-to, merged-into).
    /// Only suppresses from source → target.
    DirectionalSuppressive = 2,
    /// Reverse side of a directional suppressive edge — does NOT suppress.
    /// Stored internally when building the CSR graph. Not used in EdgeInput.
    DirectionalPassive = 3,
    /// Virtual affinity edge from tag overlap.
    FeatureAffinity = 4,
}

/// A single directed edge in the CSR graph.
#[derive(Debug, Clone)]
pub struct Edge {
    pub target: u32,
    pub weight: f64,
    pub kind: EdgeKind,
}

/// CSR (Compressed Sparse Row) adjacency graph.
///
/// Stores both outbound and inbound edges for bidirectional traversal.
/// Node IDs are dense integers [0, num_nodes). The caller maps string IDs
/// to dense indices before constructing the graph.
#[derive(Debug, Clone)]
pub struct CsrGraph {
    /// Number of nodes.
    pub num_nodes: u32,
    /// row_offsets[i] .. row_offsets[i+1] indexes into `edges` for node i's neighbors.
    pub row_offsets: Vec<u32>,
    /// Packed edge data, ordered by source node.
    pub edges: Vec<Edge>,
}

/// Intermediate edge used during graph construction.
pub struct EdgeInput {
    pub source: u32,
    pub target: u32,
    pub weight: f64,
    pub kind: EdgeKind,
}

impl CsrGraph {
    /// Build a CSR graph from edge inputs.
    ///
    /// Each edge is stored twice (once per endpoint) so that propagation
    /// can traverse both directions. The `Edge::target` always points to
    /// the "other" node from the perspective of the row owner.
    pub fn build(num_nodes: u32, inputs: &[EdgeInput]) -> Self {
        let n = num_nodes as usize;

        // Count degree per node (both directions).
        let mut degree = vec![0u32; n];
        for e in inputs {
            degree[e.source as usize] += 1;
            degree[e.target as usize] += 1;
        }

        // Build row_offsets via prefix sum.
        let mut row_offsets = Vec::with_capacity(n + 1);
        row_offsets.push(0);
        for &d in &degree {
            row_offsets.push(row_offsets.last().unwrap() + d);
        }
        let total_edges = *row_offsets.last().unwrap() as usize;

        // Place edges into their slots.
        let mut edges = vec![
            Edge {
                target: 0,
                weight: 0.0,
                kind: EdgeKind::Positive,
            };
            total_edges
        ];
        let mut cursor = vec![0u32; n]; // tracks next write position per row

        for e in inputs {
            let s = e.source as usize;
            let t = e.target as usize;

            // Forward edge: source row → target.
            let pos = (row_offsets[s] + cursor[s]) as usize;
            edges[pos] = Edge {
                target: e.target,
                weight: e.weight,
                kind: e.kind,
            };
            cursor[s] += 1;

            // Reverse edge: target row → source.
            // For DirectionalSuppressive edges, the reverse direction should
            // NOT suppress — store as DirectionalPassive (treated as Positive
            // by the engine).
            let reverse_kind = match e.kind {
                EdgeKind::DirectionalSuppressive => EdgeKind::DirectionalPassive,
                other => other,
            };
            let pos = (row_offsets[t] + cursor[t]) as usize;
            edges[pos] = Edge {
                target: e.source,
                weight: e.weight,
                kind: reverse_kind,
            };
            cursor[t] += 1;
        }

        CsrGraph {
            num_nodes,
            row_offsets,
            edges,
        }
    }

    /// Returns a slice of edges adjacent to `node`.
    #[inline]
    pub fn neighbors(&self, node: u32) -> &[Edge] {
        let start = self.row_offsets[node as usize] as usize;
        let end = self.row_offsets[node as usize + 1] as usize;
        &self.edges[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_query() {
        // 0 --0.8--> 1 --0.6--> 2
        let inputs = vec![
            EdgeInput {
                source: 0,
                target: 1,
                weight: 0.8,
                kind: EdgeKind::Positive,
            },
            EdgeInput {
                source: 1,
                target: 2,
                weight: 0.6,
                kind: EdgeKind::Positive,
            },
        ];
        let g = CsrGraph::build(3, &inputs);

        // Node 0: one neighbor (1)
        assert_eq!(g.neighbors(0).len(), 1);
        assert_eq!(g.neighbors(0)[0].target, 1);

        // Node 1: two neighbors (0 and 2), bidirectional
        assert_eq!(g.neighbors(1).len(), 2);

        // Node 2: one neighbor (1)
        assert_eq!(g.neighbors(2).len(), 1);
        assert_eq!(g.neighbors(2)[0].target, 1);
    }

    #[test]
    fn conflict_edges() {
        let inputs = vec![EdgeInput {
            source: 0,
            target: 1,
            weight: 0.8,
            kind: EdgeKind::Conflicts,
        }];
        let g = CsrGraph::build(2, &inputs);

        // Both directions should have Conflicts kind.
        assert_eq!(g.neighbors(0)[0].kind, EdgeKind::Conflicts);
        assert_eq!(g.neighbors(1)[0].kind, EdgeKind::Conflicts);
    }

    #[test]
    fn directional_suppressive_preserves_direction() {
        // Source=0 → Target=1 with DirectionalSuppressive.
        // The reverse edge (stored at node 1) still records the original source
        // direction via its target field pointing back at 0. The engine uses
        // the row owner vs edge.target to determine direction.
        let inputs = vec![EdgeInput {
            source: 0,
            target: 1,
            weight: 0.5,
            kind: EdgeKind::DirectionalSuppressive,
        }];
        let g = CsrGraph::build(2, &inputs);

        // From node 0's perspective: target=1 (outbound, should suppress)
        let e0 = &g.neighbors(0)[0];
        assert_eq!(e0.target, 1);
        assert_eq!(e0.kind, EdgeKind::DirectionalSuppressive);

        // From node 1's perspective: target=0 (inbound, stored as DirectionalPassive)
        let e1 = &g.neighbors(1)[0];
        assert_eq!(e1.target, 0);
        assert_eq!(e1.kind, EdgeKind::DirectionalPassive);
    }
}
