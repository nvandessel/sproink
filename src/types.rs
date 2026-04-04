//! Core newtypes for the sproink engine.
//!
//! All numeric values are wrapped in newtypes to prevent mixing node IDs,
//! edge weights, and activation levels at the type level.

use std::fmt;

use crate::error::SproinkError;

/// Unique identifier for a node in the graph.
///
/// Wraps a `u32` and provides [`index()`](NodeId::index) for use as a slice index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NodeId(u32);

impl NodeId {
    /// Creates a new `NodeId` from a raw `u32`.
    #[inline]
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the inner `u32` value.
    #[inline]
    #[must_use]
    pub fn get(self) -> u32 {
        self.0
    }

    /// Returns the node ID as a `usize` index for array/slice access.
    #[inline]
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for NodeId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Edge weight in the range `[0.0, 1.0]`.
///
/// Construction via [`new()`](EdgeWeight::new) validates the range; use
/// [`new_unchecked()`](EdgeWeight::new_unchecked) when the value is known-good.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct EdgeWeight(f64);

impl EdgeWeight {
    /// Creates a validated edge weight. Returns an error if the value is
    /// NaN, infinite, or outside `[0.0, 1.0]`.
    pub fn new(v: f64) -> Result<Self, SproinkError> {
        if v.is_nan() || v.is_infinite() || !(0.0..=1.0).contains(&v) {
            return Err(SproinkError::InvalidValue {
                field: "edge_weight",
                value: v,
            });
        }
        Ok(Self(v))
    }

    /// Creates an edge weight without validation. Debug builds panic on invalid values.
    #[inline]
    #[must_use]
    pub fn new_unchecked(v: f64) -> Self {
        debug_assert!(!v.is_nan() && (0.0..=1.0).contains(&v));
        Self(v)
    }

    /// Returns the inner `f64` value.
    #[inline]
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Display for EdgeWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Activation level in the range `[0.0, 1.0]`.
///
/// Represents the energy at a node during or after propagation.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Activation(f64);

impl Activation {
    /// Creates a validated activation. Returns an error if the value is
    /// NaN, infinite, or outside `[0.0, 1.0]`.
    pub fn new(v: f64) -> Result<Self, SproinkError> {
        if v.is_nan() || v.is_infinite() || !(0.0..=1.0).contains(&v) {
            return Err(SproinkError::InvalidValue {
                field: "activation",
                value: v,
            });
        }
        Ok(Self(v))
    }

    /// Creates an activation without validation. Debug builds panic on invalid values.
    #[inline]
    #[must_use]
    pub fn new_unchecked(v: f64) -> Self {
        debug_assert!(!v.is_nan() && (0.0..=1.0).contains(&v));
        Self(v)
    }

    /// Returns the inner `f64` value.
    #[inline]
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a tag used in affinity calculations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct TagId(u32);

impl TagId {
    /// Creates a new `TagId` from a raw `u32`.
    #[inline]
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the inner `u32` value.
    #[inline]
    #[must_use]
    pub fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for TagId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl fmt::Display for TagId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Activation seed for the propagation engine.
///
/// Each seed injects energy into a single node. The optional `source` field
/// tags which external entity originated this seed (for provenance tracking).
#[derive(Debug, Clone)]
pub struct Seed {
    /// The node to inject activation into.
    pub node: NodeId,
    /// Initial activation level for the seed node.
    pub activation: Activation,
    /// Optional identifier for the external source that produced this seed.
    pub source: Option<u32>,
}

/// A single node's activation after propagation completes.
///
/// Results are sorted by activation descending, with ties broken by node ID ascending.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationResult {
    /// The activated node.
    pub node: NodeId,
    /// Final activation level after all post-processing.
    pub activation: Activation,
    /// Shortest hop distance from the nearest seed node.
    pub distance: u32,
    /// Source identifier inherited from the seed that reached this node first.
    pub seed_source: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- NodeId ---
    #[test]
    fn node_id_get_returns_inner() {
        assert_eq!(NodeId::new(42).get(), 42);
    }

    #[test]
    fn node_id_index_returns_usize() {
        assert_eq!(NodeId::new(5).index(), 5usize);
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
        assert_eq!(TagId::new(99).get(), 99);
    }

    // --- Seed ---
    #[test]
    fn seed_construction() {
        let s = Seed {
            node: NodeId::new(0),
            activation: Activation::new(0.8).unwrap(),
            source: None,
        };
        assert_eq!(s.node, NodeId::new(0));
        assert_eq!(s.activation.get(), 0.8);
    }

    // --- ActivationResult ---
    #[test]
    fn activation_result_construction() {
        let r = ActivationResult {
            node: NodeId::new(5),
            activation: Activation::new(0.6).unwrap(),
            distance: 2,
            seed_source: None,
        };
        assert_eq!(r.node, NodeId::new(5));
        assert_eq!(r.activation.get(), 0.6);
        assert_eq!(r.distance, 2);
    }
}
