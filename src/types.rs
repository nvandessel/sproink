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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
///
/// Since NaN values are rejected at construction, `Eq` and `Ord` are safely
/// implemented using [`f64::total_cmp`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "f64"))]
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
        // Canonicalize -0.0 to +0.0 so PartialEq (IEEE 754) agrees with Ord
        // (total_cmp), preserving the Ord contract that `a == b` implies
        // `a.cmp(&b) == Ordering::Equal`.
        Ok(Self(v + 0.0))
    }

    /// Creates an edge weight without validation. Debug builds panic on invalid values.
    #[inline]
    #[must_use]
    pub fn new_unchecked(v: f64) -> Self {
        debug_assert!(!v.is_nan() && (0.0..=1.0).contains(&v));
        // Canonicalize -0.0 to +0.0 so the Ord contract holds — must mirror
        // `new()` or the two constructors would produce differently-ordered
        // equal values.
        Self(v + 0.0)
    }

    /// Returns the inner `f64` value.
    #[inline]
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

impl Eq for EdgeWeight {}

impl PartialOrd for EdgeWeight {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeWeight {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl TryFrom<f64> for EdgeWeight {
    type Error = SproinkError;

    fn try_from(v: f64) -> Result<Self, Self::Error> {
        Self::new(v)
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
///
/// Since NaN values are rejected at construction, `Eq` and `Ord` are safely
/// implemented using [`f64::total_cmp`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "f64"))]
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
        // Canonicalize -0.0 to +0.0 so PartialEq (IEEE 754) agrees with Ord
        // (total_cmp), preserving the Ord contract that `a == b` implies
        // `a.cmp(&b) == Ordering::Equal`.
        Ok(Self(v + 0.0))
    }

    /// Creates an activation without validation. Debug builds panic on invalid values.
    #[inline]
    #[must_use]
    pub fn new_unchecked(v: f64) -> Self {
        debug_assert!(!v.is_nan() && (0.0..=1.0).contains(&v));
        // Canonicalize -0.0 to +0.0 so the Ord contract holds — must mirror
        // `new()` or the two constructors would produce differently-ordered
        // equal values.
        Self(v + 0.0)
    }

    /// Returns the inner `f64` value.
    #[inline]
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

impl Eq for Activation {}

impl PartialOrd for Activation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Activation {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl TryFrom<f64> for Activation {
    type Error = SproinkError;

    fn try_from(v: f64) -> Result<Self, Self::Error> {
        Self::new(v)
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a tag used in affinity calculations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    #[test]
    fn edge_weight_eq_and_ord() {
        let a = EdgeWeight::new(0.3).unwrap();
        let b = EdgeWeight::new(0.7).unwrap();
        let c = EdgeWeight::new(0.3).unwrap();
        assert_eq!(a, c);
        assert!(a < b);
        assert!(b > a);
        let mut v = [b, a, c];
        v.sort();
        assert_eq!(v[0].get(), 0.3);
        assert_eq!(v[2].get(), 0.7);
    }

    #[test]
    fn edge_weight_unchecked_canonicalizes_negative_zero() {
        let w = EdgeWeight::new_unchecked(-0.0);
        assert_eq!(w.get(), 0.0);
        assert!(!w.get().is_sign_negative());
        let pos = EdgeWeight::new_unchecked(0.0);
        assert_eq!(w, pos);
        assert_eq!(w.cmp(&pos), std::cmp::Ordering::Equal);
    }

    #[test]
    fn edge_weight_canonicalizes_negative_zero() {
        // -0.0 must round-trip to +0.0 so PartialEq and Ord agree.
        let w = EdgeWeight::new(-0.0).unwrap();
        assert_eq!(w.get(), 0.0);
        assert!(!w.get().is_sign_negative());

        // PartialEq/Ord contract: equal values must compare Equal.
        let pos = EdgeWeight::new(0.0).unwrap();
        let neg = EdgeWeight::new(-0.0).unwrap();
        assert_eq!(pos, neg);
        assert_eq!(pos.cmp(&neg), std::cmp::Ordering::Equal);
    }

    #[test]
    fn edge_weight_try_from() {
        assert!(EdgeWeight::try_from(0.5).is_ok());
        assert!(EdgeWeight::try_from(1.5).is_err());
        assert!(EdgeWeight::try_from(f64::NAN).is_err());
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

    #[test]
    fn activation_eq_and_ord() {
        let a = Activation::new(0.2).unwrap();
        let b = Activation::new(0.9).unwrap();
        let c = Activation::new(0.2).unwrap();
        assert_eq!(a, c);
        assert!(a < b);
        let mut v = [b, a, c];
        v.sort();
        assert_eq!(v[0].get(), 0.2);
        assert_eq!(v[2].get(), 0.9);
    }

    #[test]
    fn activation_unchecked_canonicalizes_negative_zero() {
        let a = Activation::new_unchecked(-0.0);
        assert_eq!(a.get(), 0.0);
        assert!(!a.get().is_sign_negative());
        let pos = Activation::new_unchecked(0.0);
        assert_eq!(a, pos);
        assert_eq!(a.cmp(&pos), std::cmp::Ordering::Equal);
    }

    #[test]
    fn activation_canonicalizes_negative_zero() {
        // -0.0 must round-trip to +0.0 so PartialEq and Ord agree.
        let a = Activation::new(-0.0).unwrap();
        assert_eq!(a.get(), 0.0);
        assert!(!a.get().is_sign_negative());

        // PartialEq/Ord contract: equal values must compare Equal.
        let pos = Activation::new(0.0).unwrap();
        let neg = Activation::new(-0.0).unwrap();
        assert_eq!(pos, neg);
        assert_eq!(pos.cmp(&neg), std::cmp::Ordering::Equal);
    }

    #[test]
    fn activation_try_from() {
        assert!(Activation::try_from(0.5).is_ok());
        assert!(Activation::try_from(-0.1).is_err());
        assert!(Activation::try_from(f64::NAN).is_err());
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

    #[test]
    fn seed_is_copy() {
        let s = Seed {
            node: NodeId::new(0),
            activation: Activation::new(0.5).unwrap(),
            source: None,
        };
        let s2 = s;
        let _s3 = s; // use original after copy
        assert_eq!(s2.node, NodeId::new(0));
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

    #[test]
    fn activation_result_is_copy() {
        let r = ActivationResult {
            node: NodeId::new(1),
            activation: Activation::new(0.5).unwrap(),
            distance: 1,
            seed_source: None,
        };
        let r2 = r;
        let _r3 = r; // use original after copy
        assert_eq!(r2.node, NodeId::new(1));
    }

    // --- Serde deserialization validates ---
    #[cfg(feature = "serde")]
    #[test]
    fn serde_edge_weight_rejects_out_of_range() {
        let bad_json = "1.5";
        let result: Result<EdgeWeight, _> = serde_json::from_str(bad_json);
        assert!(result.is_err(), "EdgeWeight should reject 1.5 via serde");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_activation_rejects_out_of_range() {
        let result: Result<Activation, _> = serde_json::from_str("-0.1");
        assert!(result.is_err(), "Activation should reject -0.1 via serde");

        let result2: Result<Activation, _> = serde_json::from_str("1.5");
        assert!(result2.is_err(), "Activation should reject 1.5 via serde");
    }

    // --- Serde round-trip ---
    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip() {
        let result = ActivationResult {
            node: NodeId::new(7),
            activation: Activation::new(0.85).unwrap(),
            distance: 3,
            seed_source: Some(42),
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ActivationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, deserialized);

        // Also verify Seed round-trip
        let seed = Seed {
            node: NodeId::new(0),
            activation: Activation::new(1.0).unwrap(),
            source: Some(1),
        };
        let json = serde_json::to_string(&seed).unwrap();
        let deserialized: Seed = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.node, seed.node);
        assert_eq!(deserialized.activation.get(), seed.activation.get());
        assert_eq!(deserialized.source, seed.source);
    }
}
