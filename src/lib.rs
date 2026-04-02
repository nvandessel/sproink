#![deny(unsafe_op_in_unsafe_fn)]

mod affinity;
mod engine;
mod error;
#[allow(clippy::missing_safety_doc)]
pub mod ffi;
mod graph;
mod hebbian;
mod inhibition;
mod squash;
mod types;

pub use affinity::{AffinityConfig, AffinityGenerator, JaccardAffinity, jaccard_similarity};
pub use engine::{Engine, PropagationConfig};
pub use error::SproinkError;
pub use graph::{CsrGraph, EdgeData, EdgeInput, EdgeKind, Graph};
pub use hebbian::{
    CoActivationPair, HebbianConfig, Learner, OjaLearner, extract_co_activation_pairs,
};
pub use inhibition::{InhibitionConfig, Inhibitor, TopMInhibitor};
pub use squash::squash_sigmoid;
pub use types::{Activation, ActivationResult, EdgeWeight, NodeId, Seed, TagId};
