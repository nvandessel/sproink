#![deny(unsafe_op_in_unsafe_fn)]

mod error;
mod types;
mod graph;
mod inhibition;
mod squash;
mod hebbian;
mod affinity;
mod engine;
#[allow(clippy::missing_safety_doc)]
pub mod ffi;

pub use error::SproinkError;
pub use types::{Activation, ActivationResult, EdgeWeight, NodeId, Seed, TagId};
pub use graph::{CsrGraph, EdgeData, EdgeInput, EdgeKind, Graph};
pub use inhibition::{InhibitionConfig, Inhibitor, TopMInhibitor};
pub use squash::squash_sigmoid;
pub use hebbian::{
    CoActivationPair, HebbianConfig, Learner, OjaLearner, extract_co_activation_pairs,
};
pub use affinity::{AffinityConfig, AffinityGenerator, JaccardAffinity, jaccard_similarity};
pub use engine::{Engine, PropagationConfig};
