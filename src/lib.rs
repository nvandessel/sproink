//! # Sproink
//!
//! In-memory spreading activation engine built on a compressed sparse row (CSR) graph.
//!
//! Sproink propagates activation energy from seed nodes through weighted edges,
//! supporting positive, conflict, directional-suppressive, and feature-affinity
//! edge types. Post-processing includes lateral inhibition and sigmoid squashing.
//!
//! # Quick Start
//!
//! ```
//! use sproink::{
//!     Activation, CsrGraph, EdgeInput, EdgeKind, EdgeWeight, Engine,
//!     NodeId, PropagationConfig, Seed,
//! };
//!
//! // Build a simple 3-node chain: 0 → 1 → 2
//! let graph = CsrGraph::build(3, vec![
//!     EdgeInput {
//!         source: NodeId::new(0),
//!         target: NodeId::new(1),
//!         weight: EdgeWeight::new(0.8).unwrap(),
//!         kind: EdgeKind::Positive,
//!         last_activated: None,
//!     },
//!     EdgeInput {
//!         source: NodeId::new(1),
//!         target: NodeId::new(2),
//!         weight: EdgeWeight::new(0.6).unwrap(),
//!         kind: EdgeKind::Positive,
//!         last_activated: None,
//!     },
//! ]);
//!
//! let engine = Engine::new(&graph);
//! let seeds = vec![Seed {
//!     node: NodeId::new(0),
//!     activation: Activation::new(1.0).unwrap(),
//!     source: None,
//! }];
//! let config = PropagationConfig::default();
//! let results = engine.activate(&seeds, &config).unwrap();
//!
//! assert!(!results.is_empty());
//! ```

#![deny(unsafe_op_in_unsafe_fn)]

mod affinity;
mod engine;
mod error;
pub mod ffi;
mod graph;
mod hebbian;
mod inhibition;
mod squash;
mod types;

pub use affinity::{AffinityConfig, AffinityGenerator, JaccardAffinity, jaccard_similarity};
pub use engine::{Engine, PropagationConfig, StepSnapshot};
pub use error::SproinkError;
pub use graph::{CsrGraph, EdgeData, EdgeInput, EdgeKind, Graph};
pub use hebbian::{
    CoActivationPair, HebbianConfig, Learner, OjaLearner, extract_co_activation_pairs,
};
pub use inhibition::{InhibitionConfig, Inhibitor, TopMInhibitor};
pub use squash::squash_sigmoid;
pub use types::{Activation, ActivationResult, EdgeWeight, NodeId, Seed, TagId};
