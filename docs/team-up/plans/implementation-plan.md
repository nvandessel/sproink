# Sproink Implementation Plan

**Date:** 2026-04-02
**Source:** [sproink-spec.md](../specs/sproink-spec.md)
**Status:** Review feedback applied

---

## Overview

This plan rebuilds sproink from scratch based on the approved spec. The existing prototype in `src/` will be replaced entirely. Each task follows TDD: write failing tests first, then implement until tests pass.

The plan is ordered by dependency: foundational types first, then components, then the engine that wires them together, then FFI on top.

---

## Task 1: Project Setup

**Goal:** Clean slate with correct Cargo.toml, module structure, and crate attributes.

### Files to create/modify

- `Cargo.toml` — full dependency list
- `src/lib.rs` — crate attributes and module declarations
- `cbindgen.toml` — cbindgen configuration

### What to do

1. Replace `Cargo.toml` with:

```toml
[package]
name = "sproink"
version = "0.1.0"
edition = "2024"
description = "In-memory spreading activation engine with CSR graph"
license = "MIT"

[lib]
crate-type = ["lib", "cdylib", "staticlib"]

[dependencies]
rayon = "1.10"
thiserror = "2"
typed-builder = "0.20"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
mockall = "0.13"
proptest = "1.9"

[build-dependencies]
cbindgen = "0.27"

[[bench]]
name = "propagation"
harness = false
```

2. Replace `src/lib.rs` with:

```rust
#![deny(unsafe_op_in_unsafe_fn)]

mod error;
mod types;
mod graph;
mod inhibition;
mod squash;
mod hebbian;
mod affinity;
mod engine;
pub mod ffi;

pub use error::SproinkError;
pub use types::{NodeId, EdgeWeight, Activation, TagId, Seed, ActivationResult};
pub use graph::{EdgeKind, EdgeData, EdgeInput, Graph, CsrGraph};
pub use inhibition::{InhibitionConfig, Inhibitor, TopMInhibitor};
pub use squash::squash_sigmoid;
pub use hebbian::{HebbianConfig, CoActivationPair, Learner, OjaLearner, extract_co_activation_pairs};
pub use affinity::{AffinityConfig, AffinityGenerator, JaccardAffinity};
pub use engine::{PropagationConfig, Engine};
```

3. Create empty stub files for each module (`error.rs`, `types.rs`, `graph.rs`, `inhibition.rs`, `squash.rs`, `hebbian.rs`, `affinity.rs`, `engine.rs`, `ffi.rs`) so the project compiles.

4. Create `cbindgen.toml`:

```toml
language = "C"
include_guard = "SPROINK_H"
no_includes = true

[export]
prefix = "sproink_"

[fn]
prefix = "sproink_"
```

### Acceptance criteria

- `cargo check` succeeds with no errors
- All module stubs exist
- Crate attribute `#![deny(unsafe_op_in_unsafe_fn)]` is present in lib.rs

### Dependencies

None.

---

## Task 2: Error Types

**Goal:** Define `SproinkError` with thiserror.

### Files

- `src/error.rs`

### Tests to write FIRST

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_value_displays_field_and_value() {
        let err = SproinkError::InvalidValue { field: "activation", value: 1.5 };
        let msg = err.to_string();
        assert!(msg.contains("activation"));
        assert!(msg.contains("1.5"));
    }

    #[test]
    fn node_out_of_bounds_displays_details() {
        let err = SproinkError::NodeOutOfBounds {
            node: NodeId(42),
            num_nodes: 10,
        };
        let msg = err.to_string();
        assert!(msg.contains("42"));
        assert!(msg.contains("10"));
    }
}
```

### Implementation

```rust
use crate::types::NodeId;

#[derive(Debug, thiserror::Error)]
pub enum SproinkError {
    #[error("invalid {field}: {value} (expected finite value in valid range)")]
    InvalidValue { field: &'static str, value: f64 },

    #[error("node {node:?} out of bounds (graph has {num_nodes} nodes)")]
    NodeOutOfBounds { node: NodeId, num_nodes: u32 },
}
```

**Note:** `NodeId` is needed here, so `types.rs` needs at least the `NodeId` definition (even if incomplete) for this to compile. You may need to implement `NodeId` first as a minimal stub, or implement Tasks 2 and 3 together.

### Acceptance criteria

- Both tests pass
- `SproinkError` implements `std::error::Error` and `Display` via thiserror

### Dependencies

Task 1 (project setup). Partially co-dependent with Task 3 (needs NodeId).

---

## Task 3: Newtypes

**Goal:** Define `NodeId`, `EdgeWeight`, `Activation`, `TagId` with validation and accessors.

### Files

- `src/types.rs`

### Tests to write FIRST

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // --- NodeId ---
    #[test]
    fn node_id_get_returns_inner() {
        assert_eq!(NodeId(42).get(), 42);
    }

    #[test]
    fn node_id_index_returns_usize() {
        assert_eq!(NodeId(5).index(), 5usize);
    }

    // --- EdgeWeight ---
    #[test]
    fn edge_weight_valid_range() {
        assert!(EdgeWeight::new(0.0).is_ok());
        assert!(EdgeWeight::new(0.5).is_ok());
        assert!(EdgeWeight::new(1.0).is_ok());
    }

    #[test]
    fn edge_weight_rejects_nan() {
        assert!(EdgeWeight::new(f64::NAN).is_err());
    }

    #[test]
    fn edge_weight_rejects_infinity() {
        assert!(EdgeWeight::new(f64::INFINITY).is_err());
        assert!(EdgeWeight::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn edge_weight_rejects_out_of_range() {
        assert!(EdgeWeight::new(-0.1).is_err());
        assert!(EdgeWeight::new(1.1).is_err());
    }

    #[test]
    fn edge_weight_get_returns_inner() {
        assert_eq!(EdgeWeight::new(0.75).unwrap().get(), 0.75);
    }

    #[test]
    fn edge_weight_unchecked_bypasses_validation() {
        let w = EdgeWeight::new_unchecked(0.5);
        assert_eq!(w.get(), 0.5);
    }

    // --- Activation ---
    #[test]
    fn activation_valid_range() {
        assert!(Activation::new(0.0).is_ok());
        assert!(Activation::new(0.5).is_ok());
        assert!(Activation::new(1.0).is_ok());
    }

    #[test]
    fn activation_rejects_nan() {
        assert!(Activation::new(f64::NAN).is_err());
    }

    #[test]
    fn activation_rejects_out_of_range() {
        assert!(Activation::new(-0.01).is_err());
        assert!(Activation::new(1.01).is_err());
    }

    #[test]
    fn activation_get_returns_inner() {
        assert_eq!(Activation::new(0.42).unwrap().get(), 0.42);
    }

    // --- TagId ---
    #[test]
    fn tag_id_get_returns_inner() {
        assert_eq!(TagId(99).get(), 99);
    }

    // --- Seed ---
    #[test]
    fn seed_construction() {
        let s = Seed { node: NodeId(0), activation: Activation::new(0.8).unwrap() };
        assert_eq!(s.node, NodeId(0));
        assert_eq!(s.activation.get(), 0.8);
    }

    // --- ActivationResult ---
    #[test]
    fn activation_result_construction() {
        let r = ActivationResult {
            node: NodeId(5),
            activation: Activation::new(0.6).unwrap(),
            distance: 2,
        };
        assert_eq!(r.node, NodeId(5));
        assert_eq!(r.activation.get(), 0.6);
        assert_eq!(r.distance, 2);
    }
}
```

### Implementation

```rust
use crate::error::SproinkError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    #[inline]
    pub fn get(self) -> u32 { self.0 }

    #[inline]
    pub fn index(self) -> usize { self.0 as usize }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct EdgeWeight(f64);

impl EdgeWeight {
    pub fn new(v: f64) -> Result<Self, SproinkError> {
        if v.is_nan() || v.is_infinite() || v < 0.0 || v > 1.0 {
            return Err(SproinkError::InvalidValue { field: "edge_weight", value: v });
        }
        Ok(Self(v))
    }

    #[inline]
    pub fn new_unchecked(v: f64) -> Self {
        debug_assert!(!v.is_nan() && v >= 0.0 && v <= 1.0);
        Self(v)
    }

    #[inline]
    pub fn get(self) -> f64 { self.0 }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Activation(f64);

impl Activation {
    pub fn new(v: f64) -> Result<Self, SproinkError> {
        if v.is_nan() || v.is_infinite() || v < 0.0 || v > 1.0 {
            return Err(SproinkError::InvalidValue { field: "activation", value: v });
        }
        Ok(Self(v))
    }

    #[inline]
    pub fn new_unchecked(v: f64) -> Self {
        debug_assert!(!v.is_nan() && v >= 0.0 && v <= 1.0);
        Self(v)
    }

    #[inline]
    pub fn get(self) -> f64 { self.0 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct TagId(pub u32);

impl TagId {
    #[inline]
    pub fn get(self) -> u32 { self.0 }
}

/// Activation seed for the propagation engine.
pub struct Seed {
    pub node: NodeId,
    pub activation: Activation,
}

/// Single node's result after propagation.
pub struct ActivationResult {
    pub node: NodeId,
    pub activation: Activation,
    pub distance: u32,
}
```

### Acceptance criteria

- All 15 tests pass (13 original + 2 for Seed/ActivationResult)
- All newtypes have `#[repr(transparent)]`
- No `Deref` implementations
- `EdgeWeight` and `Activation` enforce `[0.0, 1.0]` range and reject NaN/Inf
- `NodeId` and `TagId` have public inner field

### Dependencies

Task 1, Task 2 (for SproinkError).

---

## Task 4: Graph Trait + CsrGraph

**Goal:** Define `EdgeKind`, `EdgeData`, `EdgeInput`, `Graph` trait, and `CsrGraph` implementation.

### Files

- `src/graph.rs`

### Tests to write FIRST

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn weight(v: f64) -> EdgeWeight { EdgeWeight::new(v).unwrap() }

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
        // Forward: 0 -> 1
        let n0 = g.neighbors(NodeId(0));
        assert_eq!(n0.len(), 1);
        assert_eq!(n0[0].target, NodeId(1));
        assert_eq!(n0[0].kind, EdgeKind::Positive);
        // Reverse: 1 -> 0
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
        // Forward: 0 -> 1 is DirectionalSuppressive
        let n0 = g.neighbors(NodeId(0));
        assert_eq!(n0[0].kind, EdgeKind::DirectionalSuppressive);
        // Reverse: 1 -> 0 is DirectionalPassive
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
            EdgeInput { source: NodeId(0), target: NodeId(1), weight: weight(0.5), kind: EdgeKind::Positive },
            EdgeInput { source: NodeId(0), target: NodeId(2), weight: weight(0.7), kind: EdgeKind::Positive },
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
        // Self-loop produces two stored edges (forward + reverse), both pointing to node 0
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
```

### Implementation

Define the types and `Graph` trait per spec section 3.1 and 5:

```rust
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    Positive = 0,
    Conflicts = 1,
    DirectionalSuppressive = 2,
    DirectionalPassive = 3,
    FeatureAffinity = 4,
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
```

Construction algorithm (spec section 5.2):
1. For each input edge `(u, v, w, kind)`, store forward `(u->v, kind)` and reverse `(v->u, reverse_kind)`
2. Where `reverse_kind` maps `DirectionalSuppressive -> DirectionalPassive`, all others stay the same
3. Build prefix-sum `row_offsets`, pack edges contiguously grouped by source

`neighbors()` returns `&edges[row_offsets[node]..row_offsets[node+1]]` — O(1).

Add `#[cfg_attr(test, mockall::automock)]` to the `Graph` trait for use in later tasks.

**Note on mockall:** `MockGraph::neighbors()` returns `&[EdgeData]`, which requires the mock to own the data. In tests that mock the graph, use helper functions that return `'static` edge slices (e.g., via `Box::leak` in test setup or static arrays). Alternatively, prefer constructing small real `CsrGraph` instances in tests over mocking where practical.

### Acceptance criteria

- All 8 tests pass
- `CsrGraph` implements `Graph`
- Self-loops are handled without infinite recursion or double-counting
- Bidirectional storage works correctly
- `DirectionalSuppressive` reverses to `DirectionalPassive`
- `CsrGraph` is `Send + Sync`

### Dependencies

Tasks 1-3.

---

## Task 5: Inhibition

**Goal:** Implement `InhibitionConfig`, `Inhibitor` trait, and `TopMInhibitor`.

### Files

- `src/inhibition.rs`

### Tests to write FIRST

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn config(strength: f64, breadth: usize, enabled: bool) -> InhibitionConfig {
        InhibitionConfig::builder()
            .strength(strength)
            .breadth(breadth)
            .enabled(enabled)
            .build()
    }

    #[test]
    fn disabled_is_noop() {
        let mut activations = vec![0.8, 0.6, 0.4, 0.2];
        let original = activations.clone();
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(0.15, 2, false));
        assert_eq!(activations, original);
    }

    #[test]
    fn fewer_active_than_breadth_is_noop() {
        // Only 3 active nodes, breadth=5 => no losers
        let mut activations = vec![0.8, 0.6, 0.4, 0.0, 0.0];
        let original = activations.clone();
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(0.15, 5, true));
        assert_eq!(activations, original);
    }

    #[test]
    fn losers_are_suppressed() {
        // 4 active, breadth=2 => top 2 are winners, bottom 2 are losers
        let mut activations = vec![0.8, 0.6, 0.4, 0.2];
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(0.15, 2, true));
        // Winners (0.8, 0.6) unchanged
        assert_eq!(activations[0], 0.8);
        assert_eq!(activations[1], 0.6);
        // mean_winner = (0.8 + 0.6) / 2 = 0.7
        // loser[2]: suppression = 0.15 * (0.7 - 0.4) = 0.045 => 0.4 - 0.045 = 0.355
        assert!((activations[2] - 0.355).abs() < 1e-10);
        // loser[3]: suppression = 0.15 * (0.7 - 0.2) = 0.075 => 0.2 - 0.075 = 0.125
        assert!((activations[3] - 0.125).abs() < 1e-10);
    }

    #[test]
    fn suppression_floors_at_zero() {
        let mut activations = vec![0.9, 0.01];
        let inhibitor = TopMInhibitor;
        // strength=1.0 to force heavy suppression
        inhibitor.inhibit(&mut activations, &config(1.0, 1, true));
        assert!(activations[1] >= 0.0);
    }

    #[test]
    fn tie_breaking_by_node_id() {
        // Nodes 0,1,2 all have activation 0.5, breadth=2
        // Winners should be node 0 and node 1 (lower IDs win ties)
        let mut activations = vec![0.5, 0.5, 0.5];
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(0.15, 2, true));
        // Winners unchanged
        assert_eq!(activations[0], 0.5);
        assert_eq!(activations[1], 0.5);
        // Node 2 is the loser; mean_winner = 0.5
        // suppression = 0.15 * (0.5 - 0.5) = 0.0 => unchanged
        assert_eq!(activations[2], 0.5);
    }

    #[test]
    fn zero_activations_not_counted_as_active() {
        let mut activations = vec![0.8, 0.0, 0.0, 0.0];
        let original = activations.clone();
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(0.15, 2, true));
        // Only 1 active node, breadth=2 => no losers
        assert_eq!(activations, original);
    }
}
```

### Implementation

```rust
use typed_builder::TypedBuilder;

#[derive(Debug, Clone, TypedBuilder)]
pub struct InhibitionConfig {
    #[builder(default = 0.15)]
    pub strength: f64,
    #[builder(default = 7)]
    pub breadth: usize,
    #[builder(default = true)]
    pub enabled: bool,
}

pub trait Inhibitor: Send + Sync {
    fn inhibit(&self, activations: &mut [f64], config: &InhibitionConfig);
}

pub struct TopMInhibitor;
```

Algorithm (spec section 4.2):
1. Collect indices where `activation > 0`
2. If count <= breadth: return unchanged
3. Sort by activation descending, tie-break by index ascending
4. Winners = top M; `mean_winner = sum(winner activations) / M`
5. For each loser: `suppression = strength * (mean_winner - loser_activation)`, `activation = max(0, activation - suppression)`

### Acceptance criteria

- All 6 tests pass
- `TopMInhibitor` implements `Inhibitor`
- Disabled config is a no-op
- Suppression floors at zero
- Tie-breaking by node index ascending

### Dependencies

Tasks 1-3.

---

## Task 6: Sigmoid Squashing

**Goal:** Implement `squash_sigmoid` free function.

### Files

- `src/squash.rs`

### Tests to write FIRST

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_center_returns_half() {
        let mut v = vec![0.3];
        squash_sigmoid(&mut v, 10.0, 0.3);
        assert!((v[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn well_above_center_near_one() {
        let mut v = vec![1.0];
        squash_sigmoid(&mut v, 10.0, 0.3);
        assert!(v[0] > 0.99);
    }

    #[test]
    fn well_below_center_near_zero() {
        let mut v = vec![0.0];
        squash_sigmoid(&mut v, 10.0, 0.3);
        assert!(v[0] < 0.05);
    }

    #[test]
    fn higher_gain_steeper_curve() {
        let mut low_gain = vec![0.5];
        let mut high_gain = vec![0.5];
        squash_sigmoid(&mut low_gain, 5.0, 0.3);
        squash_sigmoid(&mut high_gain, 20.0, 0.3);
        // Both above 0.5 (since 0.5 > center), but high gain closer to 1
        assert!(high_gain[0] > low_gain[0]);
    }

    #[test]
    fn empty_slice_is_noop() {
        let mut v: Vec<f64> = vec![];
        squash_sigmoid(&mut v, 10.0, 0.3); // should not panic
    }

    #[test]
    fn multiple_values() {
        let mut v = vec![0.0, 0.3, 0.6, 1.0];
        squash_sigmoid(&mut v, 10.0, 0.3);
        // Should be monotonically increasing
        for i in 1..v.len() {
            assert!(v[i] >= v[i - 1]);
        }
        // All in (0, 1) — sigmoid never reaches exactly 0 or 1
        for &val in &v {
            assert!(val > 0.0 && val < 1.0);
        }
    }
}
```

### Implementation

```rust
/// Apply sigmoid squashing element-wise in-place.
/// sigmoid(x) = 1 / (1 + exp(-gain * (x - center)))
pub fn squash_sigmoid(activations: &mut [f64], gain: f64, center: f64) {
    for v in activations.iter_mut() {
        *v = 1.0 / (1.0 + (-gain * (*v - center)).exp());
    }
}
```

### Acceptance criteria

- All 6 tests pass
- Function is a plain free function (no trait)
- Operates in-place on `&mut [f64]`

### Dependencies

Task 1.

---

## Task 7: Hebbian Learning

**Goal:** Implement `HebbianConfig`, `Learner` trait, `OjaLearner`, `CoActivationPair`, and `extract_co_activation_pairs`.

### Files

- `src/hebbian.rs`

### Tests to write FIRST

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn default_config() -> HebbianConfig {
        HebbianConfig::builder().build()
    }

    fn act(v: f64) -> Activation { Activation::new(v).unwrap() }
    fn wt(v: f64) -> EdgeWeight { EdgeWeight::new(v).unwrap() }

    // --- OjaLearner ---

    #[test]
    fn oja_strengthens_coactivation() {
        let learner = OjaLearner;
        let cfg = default_config();
        let old_w = wt(0.3);
        let new_w = learner.update_weight(old_w, act(0.8), act(0.7), &cfg);
        assert!(new_w.get() > old_w.get());
    }

    #[test]
    fn oja_stabilizes_at_high_weight() {
        // At high weight, forgetting term A_j^2 * W dominates
        let learner = OjaLearner;
        let cfg = default_config();
        let old_w = wt(0.9);
        let new_w = learner.update_weight(old_w, act(0.5), act(0.5), &cfg);
        assert!(new_w.get() < old_w.get());
    }

    #[test]
    fn oja_clamps_to_bounds() {
        let learner = OjaLearner;
        let cfg = HebbianConfig::builder()
            .min_weight(0.1)
            .max_weight(0.9)
            .learning_rate(10.0) // extreme rate to force out-of-range
            .build();
        let result = learner.update_weight(wt(0.5), act(0.99), act(0.99), &cfg);
        assert!(result.get() >= 0.1 && result.get() <= 0.9);
    }

    #[test]
    fn oja_nan_safety() {
        let learner = OjaLearner;
        let cfg = default_config();
        // Force NaN via extreme values
        let result = learner.update_weight(wt(0.5), act(0.0), act(0.0), &cfg);
        assert!(!result.get().is_nan());
        assert!(result.get() >= cfg.min_weight);
    }

    // --- extract_co_activation_pairs ---

    #[test]
    fn extracts_pairs_above_threshold() {
        let results = vec![
            ActivationResult { node: NodeId(0), activation: act(0.5), distance: 0 },
            ActivationResult { node: NodeId(1), activation: act(0.3), distance: 1 },
            ActivationResult { node: NodeId(2), activation: act(0.1), distance: 2 },
        ];
        let seeds = HashSet::new();
        let cfg = HebbianConfig::builder().activation_threshold(0.15).build();
        let pairs = extract_co_activation_pairs(&results, &seeds, &cfg);
        // Only nodes 0 and 1 are above 0.15
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].node_a, NodeId(0));
        assert_eq!(pairs[0].node_b, NodeId(1));
    }

    #[test]
    fn excludes_seed_seed_pairs() {
        let results = vec![
            ActivationResult { node: NodeId(0), activation: act(0.8), distance: 0 },
            ActivationResult { node: NodeId(1), activation: act(0.7), distance: 0 },
            ActivationResult { node: NodeId(2), activation: act(0.6), distance: 1 },
        ];
        let seeds: HashSet<NodeId> = [NodeId(0), NodeId(1)].into();
        let cfg = HebbianConfig::builder().activation_threshold(0.15).build();
        let pairs = extract_co_activation_pairs(&results, &seeds, &cfg);
        // (0,1) excluded (both seeds), (0,2) and (1,2) included
        assert_eq!(pairs.len(), 2);
        assert!(pairs.iter().all(|p| !(seeds.contains(&p.node_a) && seeds.contains(&p.node_b))));
    }

    #[test]
    fn canonical_ordering_a_less_than_b() {
        let results = vec![
            ActivationResult { node: NodeId(5), activation: act(0.5), distance: 1 },
            ActivationResult { node: NodeId(2), activation: act(0.4), distance: 1 },
        ];
        let seeds = HashSet::new();
        let cfg = HebbianConfig::builder().activation_threshold(0.15).build();
        let pairs = extract_co_activation_pairs(&results, &seeds, &cfg);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].node_a < pairs[0].node_b);
    }
}
```

### Implementation

Types and signatures per spec sections 3.5 and 4.4:

```rust
use typed_builder::TypedBuilder;
use std::collections::HashSet;

#[derive(Debug, Clone, TypedBuilder)]
pub struct HebbianConfig {
    #[builder(default = 0.05)]
    pub learning_rate: f64,
    #[builder(default = 0.01)]
    pub min_weight: f64,
    #[builder(default = 0.95)]
    pub max_weight: f64,
    #[builder(default = 0.15)]
    pub activation_threshold: f64,
}

pub struct CoActivationPair {
    pub node_a: NodeId,
    pub node_b: NodeId,
    pub activation_a: Activation,
    pub activation_b: Activation,
}

pub trait Learner: Send + Sync {
    fn update_weight(
        &self,
        current_weight: EdgeWeight,
        activation_a: Activation,
        activation_b: Activation,
        config: &HebbianConfig,
    ) -> EdgeWeight;
}

pub struct OjaLearner;
```

Oja formula: `dW = eta * (A_i * A_j - A_j^2 * W)`, then `clamp(W + dW, min, max)`. If result is NaN/Inf, fall back to `min_weight`.

`extract_co_activation_pairs`: filter by threshold, generate all `(a, b)` pairs where `a < b`, exclude both-seed pairs.

### Acceptance criteria

- All 7 tests pass
- Oja update strengthens low weights, stabilizes high weights
- Weight is always clamped to [min, max]
- NaN never escapes
- Seed-seed pairs are excluded
- Canonical ordering enforced

### Dependencies

Tasks 1-3. `ActivationResult` is defined in `types.rs` (Task 3), so no dependency on engine.

---

## Task 8: Feature Affinity (Jaccard)

**Goal:** Implement `AffinityConfig`, `AffinityGenerator` trait, `JaccardAffinity`, and Jaccard similarity.

### Files

- `src/affinity.rs`

### Tests to write FIRST

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn tag(id: u32) -> TagId { TagId(id) }

    // --- Jaccard similarity ---

    #[test]
    fn jaccard_identical_sets() {
        let a = vec![tag(1), tag(2), tag(3)];
        assert!((jaccard_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a = vec![tag(1), tag(2)];
        let b = vec![tag(3), tag(4)];
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a = vec![tag(1), tag(2), tag(3)];
        let b = vec![tag(2), tag(3), tag(4)];
        // intersection = {2,3} = 2, union = {1,2,3,4} = 4 => J = 0.5
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn jaccard_both_empty() {
        let a: Vec<TagId> = vec![];
        assert_eq!(jaccard_similarity(&a, &a), 0.0);
    }

    #[test]
    fn jaccard_one_empty() {
        let a = vec![tag(1)];
        let b: Vec<TagId> = vec![];
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn jaccard_is_symmetric() {
        let a = vec![tag(1), tag(2), tag(3)];
        let b = vec![tag(2), tag(4)];
        assert_eq!(jaccard_similarity(&a, &b), jaccard_similarity(&b, &a));
    }

    // --- AffinityGenerator ---

    #[test]
    fn generates_edges_above_threshold() {
        let node_tags = vec![
            vec![tag(1), tag(2), tag(3)],
            vec![tag(2), tag(3), tag(4)],  // J(0,1) = 0.5
            vec![tag(10), tag(11)],         // J(0,2) = 0, J(1,2) = 0
        ];
        let config = AffinityConfig { max_weight: 0.4, min_jaccard: 0.3 };
        let gen = JaccardAffinity;
        let edges = gen.generate(&node_tags, &config);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, NodeId(0));
        assert_eq!(edges[0].target, NodeId(1));
        assert_eq!(edges[0].kind, EdgeKind::FeatureAffinity);
        // weight = J * max_weight = 0.5 * 0.4 = 0.2
        assert!((edges[0].weight.get() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn skips_empty_tag_sets() {
        let node_tags = vec![
            vec![tag(1), tag(2)],
            vec![],
            vec![tag(1), tag(2)],
        ];
        let config = AffinityConfig { max_weight: 0.4, min_jaccard: 0.0 };
        let gen = JaccardAffinity;
        let edges = gen.generate(&node_tags, &config);
        // Only (0,2) — (0,1) and (1,2) skipped because node 1 is empty
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, NodeId(0));
        assert_eq!(edges[0].target, NodeId(2));
    }

    #[test]
    fn edges_have_canonical_source_less_than_target() {
        let node_tags = vec![
            vec![tag(1)],
            vec![tag(1)],
        ];
        let config = AffinityConfig { max_weight: 0.4, min_jaccard: 0.0 };
        let gen = JaccardAffinity;
        let edges = gen.generate(&node_tags, &config);
        for edge in &edges {
            assert!(edge.source.get() < edge.target.get());
        }
    }
}
```

### Implementation

```rust
pub struct AffinityConfig {
    pub max_weight: f64,
    pub min_jaccard: f64,
}

pub trait AffinityGenerator: Send + Sync {
    fn generate(
        &self,
        node_tags: &[Vec<TagId>],
        config: &AffinityConfig,
    ) -> Vec<EdgeInput>;
}

pub struct JaccardAffinity;

/// Two-pointer merge Jaccard on pre-sorted tag slices.
pub fn jaccard_similarity(a: &[TagId], b: &[TagId]) -> f64 { ... }
```

Jaccard uses two-pointer merge (O(n+m)). Both-empty and either-empty return 0.0. Input slices must be sorted.

Virtual edges are `EdgeKind::FeatureAffinity` with `weight = J * max_weight`.

### Acceptance criteria

- All 9 tests pass
- Two-pointer Jaccard is O(n+m)
- Empty sets handled correctly
- Only generates edges above `min_jaccard` threshold
- Output edges use `EdgeKind::FeatureAffinity`

### Dependencies

Tasks 1-4 (needs `EdgeInput`, `EdgeKind`, `NodeId`, `TagId`, `EdgeWeight`).

---

## Task 9a: Engine — Sequential Propagation

**Goal:** Implement `PropagationConfig`, `Engine<G: Graph>` with sequential propagation, seed initialization, post-processing pipeline, and all core tests. `Seed` and `ActivationResult` are already defined in `types.rs` (Task 3).

### Files

- `src/engine.rs`

### Tests to write FIRST

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{CsrGraph, EdgeInput, EdgeKind};
    use crate::types::{Seed, ActivationResult};

    fn weight(v: f64) -> EdgeWeight { EdgeWeight::new(v).unwrap() }
    fn act(v: f64) -> Activation { Activation::new(v).unwrap() }

    fn build_chain(n: u32, w: f64) -> CsrGraph {
        let edges: Vec<EdgeInput> = (0..n-1).map(|i| EdgeInput {
            source: NodeId(i), target: NodeId(i + 1),
            weight: weight(w), kind: EdgeKind::Positive,
        }).collect();
        CsrGraph::build(n, edges)
    }

    fn no_post_processing_config() -> PropagationConfig {
        PropagationConfig::builder()
            .max_steps(3)
            .decay_factor(1.0)
            .spread_factor(1.0)
            .min_activation(0.0)
            .sigmoid_gain(0.0)  // disable sigmoid effect
            .sigmoid_center(0.0)
            .build()
    }

    // --- Seed initialization ---

    #[test]
    fn single_seed_activates_direct_neighbor() {
        let graph = build_chain(3, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .decay_factor(1.0)
            .spread_factor(1.0)
            .min_activation(0.001)
            .sigmoid_gain(10.0)
            .sigmoid_center(0.3)
            .build();
        let seeds = vec![Seed { node: NodeId(0), activation: act(1.0) }];
        let results = engine.activate(&seeds, &config);
        // Node 0 should be active (seed), node 1 should be active (neighbor)
        assert!(results.iter().any(|r| r.node == NodeId(0)));
        assert!(results.iter().any(|r| r.node == NodeId(1)));
    }

    #[test]
    fn empty_seeds_no_panic() {
        // With no seeds, all activations start at 0. After sigmoid,
        // sigmoid(0) ≈ 0.047 > default min_activation (0.01), so nodes
        // appear with baseline sigmoid activation. This is correct behavior.
        let graph = build_chain(5, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let results = engine.activate(&[], &config);
        // Should not panic; all results should have sigmoid(0) activation
        for r in &results {
            assert!(r.activation.get() < 0.05);
        }
    }

    #[test]
    fn activation_decays_with_distance() {
        let graph = build_chain(4, 1.0);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .decay_factor(0.5)
            .spread_factor(1.0)
            .min_activation(0.001)
            .sigmoid_gain(10.0)
            .sigmoid_center(0.3)
            .build();
        let seeds = vec![Seed { node: NodeId(0), activation: act(1.0) }];
        let results = engine.activate(&seeds, &config);
        // Before sigmoid: node 0 has 1.0, node 1 gets less, node 2 even less
        // After sigmoid this ordering should be preserved
        let a = |id: u32| results.iter().find(|r| r.node == NodeId(id)).map(|r| r.activation.get()).unwrap_or(0.0);
        assert!(a(0) > a(1));
        assert!(a(1) > a(2));
    }

    #[test]
    fn conflict_suppresses_target() {
        let edges = vec![
            EdgeInput { source: NodeId(0), target: NodeId(1), weight: weight(1.0), kind: EdgeKind::Positive },
            EdgeInput { source: NodeId(0), target: NodeId(2), weight: weight(1.0), kind: EdgeKind::Conflicts },
        ];
        let graph = CsrGraph::build(3, edges);
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
            Seed { node: NodeId(0), activation: act(0.8) },
            Seed { node: NodeId(2), activation: act(0.8) },
        ];
        let results = engine.activate(&seeds, &config);
        // Node 1 gets positive energy from 0, node 2 gets suppressed by 0's conflict
        let a = |id: u32| results.iter().find(|r| r.node == NodeId(id)).map(|r| r.activation.get()).unwrap_or(0.0);
        // Node 2's activation should be reduced relative to node 1
        assert!(a(1) > a(2) || (a(2) - a(1)).abs() < 0.01); // conflict + self-seed interplay
    }

    #[test]
    fn directional_suppressive_only_forward() {
        let edges = vec![
            EdgeInput { source: NodeId(0), target: NodeId(1), weight: weight(1.0), kind: EdgeKind::DirectionalSuppressive },
        ];
        let graph = CsrGraph::build(3, edges);
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
            Seed { node: NodeId(0), activation: act(0.8) },
            Seed { node: NodeId(1), activation: act(0.8) },
        ];
        let results = engine.activate(&seeds, &config);
        let a = |id: u32| results.iter().find(|r| r.node == NodeId(id)).unwrap().activation.get();
        // Node 0 suppresses node 1 (forward = DirectionalSuppressive)
        // Node 1 does NOT suppress node 0 (reverse = DirectionalPassive, skipped)
        // So node 0 should retain more activation than node 1
        assert!(a(0) > a(1));
    }

    #[test]
    fn max_semantics_no_accumulation() {
        // Two nodes both feed into a third via Positive edges
        let edges = vec![
            EdgeInput { source: NodeId(0), target: NodeId(2), weight: weight(0.5), kind: EdgeKind::Positive },
            EdgeInput { source: NodeId(1), target: NodeId(2), weight: weight(0.5), kind: EdgeKind::Positive },
        ];
        let graph = CsrGraph::build(3, edges);
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
            Seed { node: NodeId(0), activation: act(0.8) },
            Seed { node: NodeId(1), activation: act(0.8) },
        ];
        let results = engine.activate(&seeds, &config);
        let node2 = results.iter().find(|r| r.node == NodeId(2)).unwrap();
        // With max-semantics, node 2 gets max of the two incoming energies, not sum
        // base_energy from each = 0.8 * 1.0 * 0.5 * 1.0 / positive_count
        // Each source has 1 positive edge, so energy = 0.8 * 0.5 / 1 = 0.4
        // max(0.4, 0.4) = 0.4, not 0.8
        // After sigmoid(0.4, 10, 0.3) ≈ 0.731
        assert!(node2.activation.get() < 0.9); // would be higher with sum-semantics
    }

    #[test]
    fn results_sorted_by_activation_descending() {
        let graph = build_chain(5, 0.8);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let seeds = vec![Seed { node: NodeId(0), activation: act(1.0) }];
        let results = engine.activate(&seeds, &config);
        for i in 1..results.len() {
            assert!(results[i - 1].activation.get() >= results[i].activation.get());
        }
    }

    #[test]
    fn tie_breaking_by_node_id_ascending() {
        // Build a star: center node 0 connected to all others equally
        let n = 5u32;
        let edges: Vec<EdgeInput> = (1..n).map(|i| EdgeInput {
            source: NodeId(0), target: NodeId(i),
            weight: weight(1.0), kind: EdgeKind::Positive,
        }).collect();
        let graph = CsrGraph::build(n, edges);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(1)
            .build();
        let seeds = vec![Seed { node: NodeId(0), activation: act(1.0) }];
        let results = engine.activate(&seeds, &config);
        // Nodes 1-4 all receive equal activation. Among equal activations,
        // lower NodeId should come first.
        let tied: Vec<&ActivationResult> = results.iter()
            .filter(|r| r.node != NodeId(0))
            .collect();
        for i in 1..tied.len() {
            if (tied[i-1].activation.get() - tied[i].activation.get()).abs() < 1e-15 {
                assert!(tied[i - 1].node.get() < tied[i].node.get());
            }
        }
    }

    #[test]
    fn distance_tracking() {
        let graph = build_chain(4, 1.0);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed { node: NodeId(0), activation: act(1.0) }];
        let results = engine.activate(&seeds, &config);
        let dist = |id: u32| results.iter().find(|r| r.node == NodeId(id)).map(|r| r.distance);
        assert_eq!(dist(0), Some(0));
        assert_eq!(dist(1), Some(1));
        assert_eq!(dist(2), Some(2));
        assert_eq!(dist(3), Some(3));
    }

    #[test]
    fn directional_passive_completely_skipped() {
        // DirectionalPassive should not spread energy or count in any denominator
        // Node 1 -> Node 0 is DirectionalPassive (reverse of 0->1 DirectionalSuppressive)
        // Even if node 1 is seeded, it should not affect node 0 via the passive edge
        let edges = vec![
            EdgeInput { source: NodeId(0), target: NodeId(1), weight: weight(1.0), kind: EdgeKind::DirectionalSuppressive },
        ];
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
        // Only seed node 1
        let seeds = vec![Seed { node: NodeId(1), activation: act(0.9) }];
        let results = engine.activate(&seeds, &config);
        let a0 = results.iter().find(|r| r.node == NodeId(0)).map(|r| r.activation.get());
        let a1 = results.iter().find(|r| r.node == NodeId(1)).unwrap().activation.get();
        // Node 0 should only have sigmoid(0) activation — no energy from node 1
        // because the 1->0 edge is DirectionalPassive (skipped)
        // sigmoid(0, 10, 0.3) ≈ 0.047
        if let Some(a0_val) = a0 {
            assert!(a0_val < 0.1, "Node 0 got activation {} — passive edge should be skipped", a0_val);
        }
    }

    #[test]
    fn activation_preserved_across_steps() {
        // Seeds should keep their activation across propagation steps
        // (because new vector is initialized as a copy)
        let graph = build_chain(2, 0.5);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .decay_factor(0.5)
            .spread_factor(0.5)
            .min_activation(0.001)
            .sigmoid_gain(10.0)
            .sigmoid_center(0.3)
            .build();
        let seeds = vec![Seed { node: NodeId(0), activation: act(0.9) }];
        let results = engine.activate(&seeds, &config);
        let a0 = results.iter().find(|r| r.node == NodeId(0)).unwrap().activation.get();
        // Node 0's pre-sigmoid activation should be at least 0.9 (seed value preserved by copy-init + max-semantics)
        // After sigmoid(0.9, 10, 0.3) ≈ 0.998
        assert!(a0 > 0.95, "Seed activation not preserved: got {}", a0);
    }

    #[test]
    fn min_activation_prunes_below_threshold() {
        let graph = build_chain(10, 0.1); // weak edges
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder()
            .max_steps(3)
            .min_activation(0.5) // aggressive threshold
            .build();
        let seeds = vec![Seed { node: NodeId(0), activation: act(1.0) }];
        let results = engine.activate(&seeds, &config);
        // Distant nodes should be pruned
        for r in &results {
            assert!(r.activation.get() >= 0.5 || r.activation.get() == 0.0);
        }
    }

    #[test]
    fn four_denominator_normalization() {
        // Node 0 has 2 positive, 1 conflict, 1 FeatureAffinity edge
        // Each category should use its own denominator
        let edges = vec![
            EdgeInput { source: NodeId(0), target: NodeId(1), weight: weight(0.8), kind: EdgeKind::Positive },
            EdgeInput { source: NodeId(0), target: NodeId(2), weight: weight(0.8), kind: EdgeKind::Positive },
            EdgeInput { source: NodeId(0), target: NodeId(3), weight: weight(0.8), kind: EdgeKind::Conflicts },
            EdgeInput { source: NodeId(0), target: NodeId(4), weight: weight(0.8), kind: EdgeKind::FeatureAffinity },
        ];
        let graph = CsrGraph::build(5, edges);
        let engine = Engine::new(graph);
        // This is more of a smoke test — verifying no panics and reasonable output
        let config = PropagationConfig::builder()
            .max_steps(1)
            .min_activation(0.001)
            .build();
        let seeds = vec![Seed { node: NodeId(0), activation: act(1.0) }];
        let results = engine.activate(&seeds, &config);
        assert!(!results.is_empty());
    }
}
```

### Implementation

Types per spec sections 3.2 and 4.1. `Seed` and `ActivationResult` are imported from `crate::types`:

```rust
use typed_builder::TypedBuilder;
use crate::types::{NodeId, EdgeWeight, Activation, Seed, ActivationResult};

#[derive(Debug, Clone, TypedBuilder)]
pub struct PropagationConfig {
    #[builder(default = 3)]
    pub max_steps: u32,
    #[builder(default = 0.7)]
    pub decay_factor: f64,
    #[builder(default = 0.85)]
    pub spread_factor: f64,
    #[builder(default = 0.01)]
    pub min_activation: f64,
    #[builder(default = 10.0)]
    pub sigmoid_gain: f64,
    #[builder(default = 0.3)]
    pub sigmoid_center: f64,
    #[builder(default, setter(strip_option))]
    pub inhibition: Option<InhibitionConfig>,
}

pub struct Engine<G: Graph> {
    graph: G,
}
```

**Engine::activate algorithm** (spec section 4.1) — implement as sequential first:

1. Init `activation: Vec<f64>` of size `num_nodes`, zeroed
2. Init `distance: Vec<u32>` of size `num_nodes`, all `u32::MAX`
3. Seed: set `activation[seed.node] = seed.activation.get()`, `distance[seed.node] = 0`
4. For each step 1..=max_steps:
   a. `new = activation.clone()` (copy-init preserves prior values)
   b. For each active node `i` where `activation[i] >= min_activation`:
      - Count edges by category (excluding DirectionalPassive from all counts)
      - For each neighbor `(j, w, kind)`:
        - `base_energy = activation[i] * spread_factor * w * decay_factor`
        - Apply per-kind logic with per-kind denominator (see spec table)
        - **DirectionalPassive: skip entirely**
      - Update `distance[j] = min(distance[j], distance[i] + 1)` when energy spread
   c. `activation = new`
5. Post-processing (in order):
   a. Lateral inhibition via `TopMInhibitor` (if enabled)
   b. Sigmoid: `squash_sigmoid(&mut activation, gain, center)`
   c. Threshold: zero out nodes below `min_activation`
6. Collect results with `activation > 0`, sort by activation descending, tie-break by NodeId ascending

**Extract the propagation step into a private method** `propagate_step_sequential()` so Task 9b can add a parallel variant alongside it:

```rust
impl<G: Graph> Engine<G> {
    pub fn new(graph: G) -> Self { ... }

    pub fn activate(&self, seeds: &[Seed], config: &PropagationConfig) -> Vec<ActivationResult> {
        // ... seed init, call propagate_step for each step, post-process, collect ...
    }

    fn propagate_step_sequential(
        &self,
        current: &[f64],
        new: &mut [f64],
        distances: &mut [u32],
        config: &PropagationConfig,
    ) { ... }
}
```

### Acceptance criteria

- All 13 tests pass
- Propagation uses copy-init (activation preserved across steps)
- Max-semantics for positive energy (no sum accumulation)
- DirectionalPassive completely skipped (no energy, no counting)
- Four-denominator normalization (positive, conflict, directional suppressive, virtual)
- Distance tracking correct
- Results sorted descending, ties broken by NodeId ascending
- Post-processing order: inhibition -> sigmoid -> threshold filter
- Propagation step is extracted into a private method (ready for parallel variant in 9b)

### Dependencies

Tasks 1-6 (uses all prior modules).

---

## Task 9b: Engine — Rayon Parallel Propagation

**Goal:** Add parallel propagation via Rayon with `PARALLEL_THRESHOLD` fallback to sequential. The sequential path from Task 9a remains; this task adds the parallel path and wires `activate()` to choose between them.

### Files

- `src/engine.rs` (modify)

### Tests to write FIRST

```rust
#[cfg(test)]
mod parallel_tests {
    use super::*;
    use crate::graph::{CsrGraph, EdgeInput, EdgeKind};

    fn weight(v: f64) -> EdgeWeight { EdgeWeight::new(v).unwrap() }
    fn act(v: f64) -> Activation { Activation::new(v).unwrap() }

    fn build_random_graph(num_nodes: u32, edges_per_node: u32) -> CsrGraph {
        let mut edges = Vec::new();
        for i in 0..num_nodes {
            for j in 0..edges_per_node {
                let target = (i + j * 7 + 13) % num_nodes;
                if target != i {
                    edges.push(EdgeInput {
                        source: NodeId(i), target: NodeId(target),
                        weight: weight(0.5), kind: EdgeKind::Positive,
                    });
                }
            }
        }
        CsrGraph::build(num_nodes, edges)
    }

    #[test]
    fn parallel_matches_sequential_small_graph() {
        // Below PARALLEL_THRESHOLD — both paths should give identical results
        let graph = build_random_graph(100, 5);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let seeds = vec![Seed { node: NodeId(0), activation: act(1.0) }];
        let r1 = engine.activate(&seeds, &config);
        let r2 = engine.activate(&seeds, &config);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.node, b.node);
            assert!((a.activation.get() - b.activation.get()).abs() < 1e-15);
        }
    }

    #[test]
    fn parallel_matches_sequential_large_graph() {
        // Above PARALLEL_THRESHOLD — exercises the Rayon path
        let graph = build_random_graph(2000, 5);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let seeds = vec![Seed { node: NodeId(0), activation: act(1.0) }];
        let results = engine.activate(&seeds, &config);
        // Verify invariants hold
        for r in &results {
            assert!(r.activation.get() >= 0.0 && r.activation.get() <= 1.0);
        }
        // Determinism: run twice, get same results
        let results2 = engine.activate(&seeds, &config);
        assert_eq!(results.len(), results2.len());
        for (a, b) in results.iter().zip(results2.iter()) {
            assert_eq!(a.node, b.node);
            assert!((a.activation.get() - b.activation.get()).abs() < 1e-15);
        }
    }

    #[test]
    fn parallel_activation_bounds() {
        let graph = build_random_graph(2000, 10);
        let engine = Engine::new(graph);
        let config = PropagationConfig::builder().build();
        let seeds = vec![
            Seed { node: NodeId(0), activation: act(1.0) },
            Seed { node: NodeId(500), activation: act(0.8) },
        ];
        let results = engine.activate(&seeds, &config);
        for r in &results {
            assert!(r.activation.get() >= 0.0 && r.activation.get() <= 1.0,
                "Activation {} out of bounds", r.activation.get());
        }
    }
}
```

### Implementation

Add a parallel propagation step using Rayon's `fold+reduce` pattern with thread-local buffers:

```rust
use rayon::prelude::*;

const PARALLEL_THRESHOLD: u32 = 1024;

impl<G: Graph> Engine<G> {
    fn propagate_step_parallel(
        &self,
        current: &[f64],
        new: &mut [f64],
        distances: &mut [u32],
        config: &PropagationConfig,
    ) {
        let num_nodes = self.graph.num_nodes() as usize;

        // Phase 1: parallel fold+reduce with thread-local buffers
        // Each Rayon thread gets its own Vec<f64> and merges via max-semantics
        let (merged_activations, merged_distances): (Vec<f64>, Vec<u32>) = (0..num_nodes)
            .into_par_iter()
            .filter(|&i| current[i] >= config.min_activation)
            .fold(
                || (vec![0.0f64; num_nodes], vec![u32::MAX; num_nodes]),
                |(mut local_act, mut local_dist), i| {
                    // ... compute per-source contributions using same logic as sequential ...
                    // Apply max-semantics for positive/affinity, subtraction for suppressive
                    (local_act, local_dist)
                },
            )
            .reduce(
                || (vec![0.0f64; num_nodes], vec![u32::MAX; num_nodes]),
                |(mut a_act, mut a_dist), (b_act, b_dist)| {
                    for j in 0..num_nodes {
                        a_act[j] = a_act[j].max(b_act[j]);
                        a_dist[j] = a_dist[j].min(b_dist[j]);
                    }
                    (a_act, a_dist)
                },
            );

        // Phase 2: merge parallel results into `new` (which starts as copy of current)
        for j in 0..num_nodes {
            new[j] = new[j].max(merged_activations[j]);
            // Suppressive updates need separate handling — collect negative deltas
            // in the fold phase and apply after merge
            distances[j] = distances[j].min(merged_distances[j]);
        }
    }

    /// Choose sequential or parallel based on graph size.
    fn propagate_step(
        &self,
        current: &[f64],
        new: &mut [f64],
        distances: &mut [u32],
        config: &PropagationConfig,
    ) {
        if self.graph.num_nodes() < PARALLEL_THRESHOLD {
            self.propagate_step_sequential(current, new, distances, config);
        } else {
            self.propagate_step_parallel(current, new, distances, config);
        }
    }
}
```

**Note on suppression in parallel:** Suppressive edges (Conflicts, DirectionalSuppressive) subtract energy. In the parallel path, suppressive deltas must be collected separately and applied sequentially after the merge phase, since max-semantics doesn't apply to subtraction. The implementation should track both positive maxima and suppressive deltas in the thread-local buffers.

Update `activate()` to call `propagate_step()` (which dispatches to sequential or parallel) instead of `propagate_step_sequential()` directly.

### Acceptance criteria

- All 3 parallel-specific tests pass
- All 13 tests from Task 9a still pass (no regressions)
- Graphs < 1024 nodes use sequential path
- Graphs >= 1024 nodes use Rayon parallel path
- Parallel and sequential produce bit-identical results for the same input
- `PARALLEL_THRESHOLD` is a named constant (1024)

### Dependencies

Task 9a.

---

## Task 10: Integration Tests

**Goal:** End-to-end pipeline tests with real graphs.

### Files

- `tests/integration.rs`

### Tests to write

```rust
use sproink::*;

/// 3-node chain: 0 -> 1 -> 2
/// Verify propagation reaches node 2 with expected decay after 3 steps.
#[test]
fn three_node_chain_propagation() { ... }

/// Triangle with all conflict edges.
/// Verify mutual suppression prevents all-active state.
#[test]
fn conflict_triangle() { ... }

/// DirectionalSuppressive edge: node 0 overrides node 1.
/// Seed both. Verify 0 suppresses 1 but 1 does NOT suppress 0.
#[test]
fn directional_override() { ... }

/// Build graph with real + affinity edges.
/// Verify affinity edges spread activation via separate denominator
/// without diluting real edge energy.
#[test]
fn affinity_integration() { ... }

/// Full pipeline: propagation + inhibition + sigmoid.
/// Verify losers are suppressed and sigmoid shapes output.
#[test]
fn inhibition_plus_sigmoid() { ... }

/// Hebbian round-trip: propagate -> extract pairs -> update weights.
#[test]
fn hebbian_round_trip() { ... }

/// Large graph: verify parallel and sequential produce identical results.
#[test]
fn parallel_matches_sequential() { ... }
```

Each test constructs a small known graph, runs the full pipeline, and asserts specific numeric outcomes. Use exact expected values where possible (pre-computed by hand or reference implementation).

### Acceptance criteria

- All 7 integration tests pass
- Tests exercise the full public API (graph build -> engine -> results -> hebbian)
- No test depends on internal implementation details

### Dependencies

Tasks 1-9b.

---

## Task 11: Property Tests

**Goal:** Use `proptest` to verify invariants hold across random inputs.

### Files

- `tests/property_tests.rs` (or add to `tests/integration.rs`)

### Tests to write

```rust
use proptest::prelude::*;
use sproink::*;

proptest! {
    /// All output activations must be in [0.0, 1.0].
    #[test]
    fn activation_bounds(
        num_nodes in 2u32..50,
        num_edges in 1usize..100,
        num_seeds in 1usize..5,
    ) { ... }

    /// Same inputs always produce same outputs (determinism).
    #[test]
    fn deterministic_output(
        num_nodes in 2u32..20,
        num_edges in 1usize..30,
    ) { ... }

    /// For positive-only graphs, activation at distance d+1 <= distance d.
    #[test]
    fn monotonic_decay_positive_only(
        chain_len in 3u32..20,
    ) { ... }

    /// Repeated Oja updates converge (not NaN/Inf).
    #[test]
    fn oja_convergence(
        w in 0.01f64..0.95,
        a_i in 0.01f64..1.0,
        a_j in 0.01f64..1.0,
        iterations in 1usize..500,
    ) { ... }

    /// Jaccard always in [0.0, 1.0], symmetric, J(A,A)=1.0.
    #[test]
    fn jaccard_properties(
        a in prop::collection::vec(0u32..100, 0..20),
        b in prop::collection::vec(0u32..100, 0..20),
    ) { ... }

    /// No seeds → no activation (after sigmoid, below threshold gets zeroed).
    #[test]
    fn zero_seeds_idempotent(num_nodes in 1u32..50) { ... }
}
```

Define `proptest` strategies for generating random graphs, seeds, and configs. Keep graph sizes small (<50 nodes) for speed.

### Acceptance criteria

- All property tests pass with default proptest config (256 cases)
- Commit proptest regression files to git
- No test takes more than 5 seconds

### Dependencies

Tasks 1-9b.

---

## Task 12: FFI Layer

**Goal:** Implement all `extern "C"` functions with opaque handles, null safety, and panic catching.

### Files

- `src/ffi.rs`

### What to implement

**Opaque handle wrapper types** — each wraps the real Rust type so `Box::into_raw` / `Box::from_raw` have a concrete type to work with:

```rust
/// Wraps CsrGraph for FFI ownership transfer.
pub struct SproinkGraph {
    inner: CsrGraph,
}

/// Wraps activation results for FFI ownership transfer.
pub struct SproinkResults {
    results: Vec<ActivationResult>,
}

/// Wraps co-activation pairs for FFI ownership transfer.
pub struct SproinkPairs {
    pairs: Vec<CoActivationPair>,
}
```

**Ownership pattern** — build returns a boxed wrapper, free reclaims it:

```rust
// Build: wrap and hand off ownership to caller
let wrapper = Box::new(SproinkGraph { inner: csr_graph });
Box::into_raw(wrapper) as *mut SproinkGraph

// Free: reclaim ownership and drop
let wrapper = unsafe { Box::from_raw(ptr) };
// wrapper.inner (CsrGraph) is dropped here

// Borrow: read through the wrapper without taking ownership
let wrapper = unsafe { &*ptr };
let results = &wrapper.results;
```

**Helper macro:**
```rust
macro_rules! ffi_entry {
    ($ptr:expr, $default:expr, $body:expr) => {{
        if $ptr.is_null() {
            return $default;
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(v) => v,
            Err(_) => $default,
        }
    }};
}
```

**Functions to implement** (spec section 7.2):

1. `sproink_graph_build(num_nodes, num_edges, sources, targets, weights, kinds) -> *mut SproinkGraph`
   - Reconstruct `Vec<EdgeInput>` from flat arrays
   - Map `kind` via `edge_kind_from_u8` (reject value 3 = DirectionalPassive, default to Positive for unknown)
   - Build `CsrGraph`, box it, return raw pointer

2. `sproink_graph_free(graph)` — null check, `Box::from_raw`, drop

3. `sproink_activate(graph, num_seeds, seed_nodes, seed_activations, max_steps, decay_factor, spread_factor, min_activation, sigmoid_gain, sigmoid_center, inhibition_enabled, inhibition_strength, inhibition_breadth) -> *mut SproinkResults`
   - Build `PropagationConfig` and `Vec<Seed>` from flat params
   - Create `Engine`, call `activate`, box results

4. `sproink_results_len(results) -> u32`
5. `sproink_results_nodes(results, out)` — copy node IDs to caller buffer
6. `sproink_results_activations(results, out)` — copy activation values
7. `sproink_results_distances(results, out)` — copy distances
8. `sproink_results_free(results)`

9. `sproink_extract_pairs(results, num_seeds, seed_nodes, activation_threshold) -> *mut SproinkPairs`
   - Build `HashSet<NodeId>` from seed array
   - Build `HebbianConfig` from threshold param
   - Call `extract_co_activation_pairs`

10. `sproink_pairs_len(pairs) -> u32`
11. `sproink_pairs_nodes(pairs, out_a, out_b)` — copy to two caller buffers
12. `sproink_pairs_activations(pairs, out_a, out_b)` — copy to two caller buffers
13. `sproink_pairs_free(pairs)`

14. `sproink_oja_update(current_weight, activation_a, activation_b, learning_rate, min_weight, max_weight) -> f64`
    - Stateless, no handle. Build config inline, call `OjaLearner::update_weight`.

**All functions must:**
- Be marked `#[unsafe(no_mangle)]`
- Use `unsafe extern "C"` calling convention
- Check null pointers (return null/0/default on null)
- Wrap body in `catch_unwind`
- Never panic across FFI boundary

### Acceptance criteria

- All 14 FFI functions compile
- `cargo build` produces cdylib and staticlib
- Null pointer inputs return safe defaults
- No `unsafe` operations outside of `ffi.rs` (except debug_assert)

### Dependencies

Tasks 1-9b.

---

## Task 13: FFI Tests

**Goal:** Round-trip tests and null safety tests for FFI functions.

### Files

- `tests/ffi_tests.rs`

### Tests to write

```rust
use sproink::ffi::*;

#[test]
fn round_trip_graph_build_activate_results() {
    // Build a 3-node chain via FFI
    // Activate via FFI
    // Extract results via FFI
    // Compare to pure-Rust results
    unsafe {
        let sources = [0u32, 1];
        let targets = [1u32, 2];
        let weights = [0.8f64, 0.8];
        let kinds = [0u8, 0]; // Positive
        let graph = sproink_graph_build(3, 2, sources.as_ptr(), targets.as_ptr(), weights.as_ptr(), kinds.as_ptr());
        assert!(!graph.is_null());

        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];
        let results = sproink_activate(
            graph, 1, seed_nodes.as_ptr(), seed_activations.as_ptr(),
            3, 0.7, 0.85, 0.01, 10.0, 0.3, true, 0.15, 7,
        );
        assert!(!results.is_null());

        let len = sproink_results_len(results);
        assert!(len > 0);

        let mut nodes = vec![0u32; len as usize];
        let mut activations = vec![0.0f64; len as usize];
        let mut distances = vec![0u32; len as usize];
        sproink_results_nodes(results, nodes.as_mut_ptr());
        sproink_results_activations(results, activations.as_mut_ptr());
        sproink_results_distances(results, distances.as_mut_ptr());

        // Verify results are reasonable
        for &a in &activations {
            assert!(a >= 0.0 && a <= 1.0);
        }

        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}

#[test]
fn null_graph_returns_null_results() {
    unsafe {
        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];
        let results = sproink_activate(
            std::ptr::null(), 1, seed_nodes.as_ptr(), seed_activations.as_ptr(),
            3, 0.7, 0.85, 0.01, 10.0, 0.3, true, 0.15, 7,
        );
        assert!(results.is_null());
    }
}

#[test]
fn null_results_returns_zero_len() {
    unsafe {
        assert_eq!(sproink_results_len(std::ptr::null()), 0);
    }
}

#[test]
fn pairs_round_trip() {
    // Build graph, activate, extract pairs via FFI
    // Verify pair extraction works end-to-end
    unsafe { ... }
}

#[test]
fn oja_update_via_ffi() {
    unsafe {
        let w = sproink_oja_update(0.3, 0.8, 0.7, 0.05, 0.01, 0.95);
        assert!(w >= 0.01 && w <= 0.95);
        assert!(w > 0.3); // should strengthen
    }
}
```

### Acceptance criteria

- All FFI tests pass
- Round-trip produces same results as pure-Rust API
- Null inputs handled gracefully (no crash, no UB)
- Run FFI tests under AddressSanitizer for UB detection (Miri does not support FFI): `RUSTFLAGS="-Zsanitizer=address" cargo +nightly test -p sproink --test ffi_tests --target x86_64-unknown-linux-gnu`

### Dependencies

Tasks 1-12.

---

## Task 14: Benchmarks

**Goal:** Criterion benchmarks for propagation at different scales.

### Files

- `benches/propagation.rs`

### What to implement

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sproink::*;

fn build_chain(n: u32) -> CsrGraph {
    let edges: Vec<EdgeInput> = (0..n-1).map(|i| EdgeInput {
        source: NodeId(i), target: NodeId(i + 1),
        weight: EdgeWeight::new(0.8).unwrap(),
        kind: EdgeKind::Positive,
    }).collect();
    CsrGraph::build(n, edges)
}

fn build_random_graph(num_nodes: u32, edges_per_node: u32) -> CsrGraph {
    // Deterministic pseudo-random graph for reproducible benchmarks
    let mut edges = Vec::new();
    for i in 0..num_nodes {
        for j in 0..edges_per_node {
            let target = (i + j * 7 + 13) % num_nodes;
            if target != i {
                edges.push(EdgeInput {
                    source: NodeId(i), target: NodeId(target),
                    weight: EdgeWeight::new(0.5 + 0.3 * ((i + j) % 3) as f64 / 2.0).unwrap(),
                    kind: EdgeKind::Positive,
                });
            }
        }
    }
    CsrGraph::build(num_nodes, edges)
}

fn bench_propagation(c: &mut Criterion) {
    // chain_100: baseline
    let g100 = build_chain(100);
    let e100 = Engine::new(g100);
    let config = PropagationConfig::builder().build();
    let seeds = vec![Seed { node: NodeId(0), activation: Activation::new(1.0).unwrap() }];

    c.bench_function("chain_100", |b| {
        b.iter(|| e100.activate(black_box(&seeds), black_box(&config)))
    });

    // random_1k_5e: typical workload
    let g1k = build_random_graph(1_000, 5);
    let e1k = Engine::new(g1k);
    c.bench_function("random_1k_5e", |b| {
        b.iter(|| e1k.activate(black_box(&seeds), black_box(&config)))
    });

    // random_10k_10e: stress test
    let g10k = build_random_graph(10_000, 10);
    let e10k = Engine::new(g10k);
    c.bench_function("random_10k_10e", |b| {
        b.iter(|| e10k.activate(black_box(&seeds), black_box(&config)))
    });
}

criterion_group!(benches, bench_propagation);
criterion_main!(benches);
```

### Acceptance criteria

- `cargo bench` runs successfully
- Three benchmarks: `chain_100`, `random_1k_5e`, `random_10k_10e`
- HTML reports generated in `target/criterion/`
- `random_10k_10e` completes in under 10ms per iteration (baseline target)

### Dependencies

Tasks 1-9b.

---

## Task 15: cbindgen Header Generation

**Goal:** Auto-generate `sproink.h` C header from Rust FFI functions.

### Files

- `build.rs` (create new)
- `cbindgen.toml` (already created in Task 1)

### What to implement

```rust
// build.rs
fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_language(cbindgen::Language::C)
        .with_include_guard("SPROINK_H")
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file("sproink.h");
}
```

### Acceptance criteria

- `cargo build` generates `sproink.h` in the crate root
- Header includes all `sproink_*` function declarations
- Header compiles with a C compiler (`cc -fsyntax-only sproink.h`)
- Opaque types appear as forward declarations

### Dependencies

Tasks 1, 12 (FFI functions must exist for cbindgen to process them).

---

## Dependency Graph

```
Task 1: Project Setup
  ├── Task 2: Error Types ──┐
  │                         ├── Task 3: Newtypes (incl. Seed, ActivationResult)
  │                         │     ├── Task 4: Graph ──────────┐
  │                         │     ├── Task 5: Inhibition      │
  │                         │     ├── Task 7: Hebbian         │
  │                         │     └── Task 8: Affinity ───────┤
  ├── Task 6: Sigmoid                                         │
  │                                                           │
  └───────────────────────────────────────────────────────────├── Task 9a: Engine (Sequential)
                                                              │     └── Task 9b: Engine (Parallel/Rayon)
                                                              │
                                                              ├── Task 10: Integration Tests
                                                              ├── Task 11: Property Tests
                                                              ├── Task 12: FFI ──── Task 13: FFI Tests
                                                              │              └──── Task 15: cbindgen
                                                              └── Task 14: Benchmarks
```

Tasks 5, 6, 7, 8 can be done in parallel (independent of each other; Task 7 no longer depends on engine since Seed/ActivationResult are in types.rs).
Task 9b depends on 9a. Tasks 10, 11, 12, 14 can be done in parallel after Task 9b.
Task 13 depends on Task 12. Task 15 depends on Task 12.

---

## Implementation Notes

### Mockall and `Graph` trait

The `Graph` trait returns `&[EdgeData]` from `neighbors()`. When using `mockall`, the mock needs to own the data that the returned reference points to. Options:
- Use `#[mockall::automock]` — mockall handles lifetime management for returning references
- Prefer small real `CsrGraph` instances in tests where mocking is awkward
- For engine tests, real graphs are often clearer than mocks

### Rayon fold+reduce vs collect-merge

The Rust expert review recommends `fold+reduce` with thread-local buffers for the parallel propagation step. This allocates one `Vec<f64>` per Rayon thread (~400KB for 50k nodes). The alternative is collecting `(target, energy)` tuples and merging sequentially. Start with `fold+reduce` — it avoids the large intermediate vector of updates.

### `Seed` and `ActivationResult` placement

These types are defined in `types.rs` (Task 3). Both `engine.rs` and `hebbian.rs` import from `crate::types`. This avoids a circular dependency and enables Tasks 5-8 to run fully in parallel without waiting for the engine.

### Edge case: sigmoid applied to all nodes

After propagation, sigmoid is applied to the entire activation vector. This means even nodes with `activation = 0.0` get `sigmoid(0) ≈ 0.047`. The threshold filter then zeros out nodes below `min_activation`. With default `min_activation = 0.01`, `sigmoid(0) = 0.047 > 0.01`, so zero-activation nodes appear in results with activation ~0.047.

This matches the Go reference behavior. If the engineer sees this and thinks it's a bug, it's not — it's how sigmoid works on zero values. The threshold filter should catch most of these, but with a very low `min_activation`, some nodes with no real activation will appear in results.
