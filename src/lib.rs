//! sproink — in-memory spreading activation engine.
//!
//! A standalone spreading activation engine using CSR (compressed sparse row)
//! graph representation for cache-friendly traversal. Designed as the compute
//! kernel for floop's behavior activation, replacing per-step SQLite I/O with
//! pure in-memory graph operations.
//!
//! # Architecture
//!
//! - **CSR graph**: Cache-friendly adjacency via `graph::CsrGraph`
//! - **3-step propagation**: Synchronous discrete-time spreading with decay
//! - **Lateral inhibition**: Top-M winners suppress competitors
//! - **Sigmoid squashing**: Sharp threshold at configurable center
//! - **Oja's rule**: Self-stabilizing Hebbian weight updates
//! - **Jaccard affinity**: Virtual edges from tag overlap
//! - **C FFI**: Flat arrays in, activation scores out (for CGO consumption)
//!
//! # Usage (Rust)
//!
//! ```
//! use sproink::graph::{CsrGraph, EdgeInput, EdgeKind};
//! use sproink::engine::{Engine, Config, Seed};
//!
//! let graph = CsrGraph::build(3, &[
//!     EdgeInput { source: 0, target: 1, weight: 0.8, kind: EdgeKind::Positive },
//!     EdgeInput { source: 1, target: 2, weight: 0.6, kind: EdgeKind::Positive },
//! ]);
//!
//! let engine = Engine::new(&graph, Config::default());
//! let results = engine.activate(&[Seed { node: 0, activation: 1.0 }]);
//!
//! for r in &results {
//!     println!("node={} activation={:.3} distance={}", r.node, r.activation, r.distance);
//! }
//! ```

pub mod affinity;
pub mod engine;
pub mod ffi;
pub mod graph;
pub mod hebbian;
pub mod inhibition;
