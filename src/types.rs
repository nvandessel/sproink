use std::fmt;

use crate::error::SproinkError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NodeId(u32);

impl NodeId {
    #[inline]
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    #[must_use]
    pub fn get(self) -> u32 {
        self.0
    }

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

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct EdgeWeight(f64);

impl EdgeWeight {
    pub fn new(v: f64) -> Result<Self, SproinkError> {
        if v.is_nan() || v.is_infinite() || !(0.0..=1.0).contains(&v) {
            return Err(SproinkError::InvalidValue {
                field: "edge_weight",
                value: v,
            });
        }
        Ok(Self(v))
    }

    #[inline]
    #[must_use]
    pub fn new_unchecked(v: f64) -> Self {
        debug_assert!(!v.is_nan() && (0.0..=1.0).contains(&v));
        Self(v)
    }

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

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Activation(f64);

impl Activation {
    pub fn new(v: f64) -> Result<Self, SproinkError> {
        if v.is_nan() || v.is_infinite() || !(0.0..=1.0).contains(&v) {
            return Err(SproinkError::InvalidValue {
                field: "activation",
                value: v,
            });
        }
        Ok(Self(v))
    }

    #[inline]
    #[must_use]
    pub fn new_unchecked(v: f64) -> Self {
        debug_assert!(!v.is_nan() && (0.0..=1.0).contains(&v));
        Self(v)
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct TagId(u32);

impl TagId {
    #[inline]
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

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
#[derive(Debug, Clone)]
pub struct Seed {
    pub node: NodeId,
    pub activation: Activation,
}

/// Single node's result after propagation.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationResult {
    pub node: NodeId,
    pub activation: Activation,
    pub distance: u32,
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
        };
        assert_eq!(r.node, NodeId::new(5));
        assert_eq!(r.activation.get(), 0.6);
        assert_eq!(r.distance, 2);
    }
}
