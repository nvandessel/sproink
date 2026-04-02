//! C FFI: flat arrays in, activation scores out.
//!
//! floop consumes this via CGO. All pointers are borrowed for the duration
//! of each call — the caller owns the memory.

use crate::engine::{ActivationResult, Config, Engine, Seed};
use crate::graph::{CsrGraph, EdgeInput, EdgeKind};
use crate::inhibition::InhibitionConfig;
use std::slice;

/// Opaque handle to a built CSR graph.
pub struct SproinkGraph {
    inner: CsrGraph,
}

/// Opaque handle to activation results.
pub struct SproinkResults {
    results: Vec<ActivationResult>,
}

/// Build a CSR graph from flat arrays.
///
/// # Safety
/// - `sources`, `targets` must point to `num_edges` elements.
/// - `weights` must point to `num_edges` elements.
/// - `kinds` must point to `num_edges` elements (0=Positive, 1=Conflicts,
///    2=DirectionalSuppressive, 3=FeatureAffinity).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_graph_build(
    num_nodes: u32,
    num_edges: u32,
    sources: *const u32,
    targets: *const u32,
    weights: *const f64,
    kinds: *const u8,
) -> *mut SproinkGraph {
    let n = num_edges as usize;
    let sources = unsafe { slice::from_raw_parts(sources, n) };
    let targets = unsafe { slice::from_raw_parts(targets, n) };
    let weights = unsafe { slice::from_raw_parts(weights, n) };
    let kinds = unsafe { slice::from_raw_parts(kinds, n) };

    let inputs: Vec<EdgeInput> = (0..n)
        .map(|i| EdgeInput {
            source: sources[i],
            target: targets[i],
            weight: weights[i],
            kind: match kinds[i] {
                1 => EdgeKind::Conflicts,
                2 => EdgeKind::DirectionalSuppressive,
                // 3 = DirectionalPassive (internal only, not expected from FFI)
                4 => EdgeKind::FeatureAffinity,
                _ => EdgeKind::Positive,
            },
        })
        .collect();

    let graph = CsrGraph::build(num_nodes, &inputs);
    Box::into_raw(Box::new(SproinkGraph { inner: graph }))
}

/// Free a graph handle.
///
/// # Safety
/// `graph` must be a valid pointer from `sproink_graph_build`, and must not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_graph_free(graph: *mut SproinkGraph) {
    if !graph.is_null() {
        drop(unsafe { Box::from_raw(graph) });
    }
}

/// Run spreading activation.
///
/// # Safety
/// - `graph` must be a valid pointer from `sproink_graph_build`.
/// - `seed_nodes` and `seed_activations` must point to `num_seeds` elements.
///
/// Returns a results handle. Caller must free with `sproink_results_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_activate(
    graph: *const SproinkGraph,
    num_seeds: u32,
    seed_nodes: *const u32,
    seed_activations: *const f64,
    // Config params — flat for CGO simplicity.
    max_steps: u32,
    decay_factor: f64,
    spread_factor: f64,
    min_activation: f64,
    sigmoid_gain: f64,
    sigmoid_center: f64,
    inhibition_enabled: bool,
    inhibition_strength: f64,
    inhibition_breadth: u32,
) -> *mut SproinkResults {
    let graph = unsafe { &(*graph).inner };
    let ns = num_seeds as usize;
    let nodes = unsafe { slice::from_raw_parts(seed_nodes, ns) };
    let acts = unsafe { slice::from_raw_parts(seed_activations, ns) };

    let seeds: Vec<Seed> = (0..ns)
        .map(|i| Seed {
            node: nodes[i],
            activation: acts[i],
        })
        .collect();

    let config = Config {
        max_steps,
        decay_factor,
        spread_factor,
        min_activation,
        sigmoid_gain,
        sigmoid_center,
        inhibition: if inhibition_enabled {
            Some(InhibitionConfig {
                strength: inhibition_strength,
                breadth: inhibition_breadth as usize,
                enabled: true,
            })
        } else {
            None
        },
    };

    let engine = Engine::new(graph, config);
    let results = engine.activate(&seeds);

    Box::into_raw(Box::new(SproinkResults { results }))
}

/// Get the number of results.
///
/// # Safety
/// `results` must be a valid pointer from `sproink_activate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_len(results: *const SproinkResults) -> u32 {
    unsafe { (*results).results.len() as u32 }
}

/// Copy result node IDs into caller-owned buffer.
///
/// # Safety
/// - `results` must be valid.
/// - `out_nodes` must point to at least `sproink_results_len()` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_nodes(results: *const SproinkResults, out_nodes: *mut u32) {
    let r = unsafe { &(*results).results };
    let out = unsafe { slice::from_raw_parts_mut(out_nodes, r.len()) };
    for (i, res) in r.iter().enumerate() {
        out[i] = res.node;
    }
}

/// Copy result activations into caller-owned buffer.
///
/// # Safety
/// - `results` must be valid.
/// - `out_activations` must point to at least `sproink_results_len()` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_activations(
    results: *const SproinkResults,
    out_activations: *mut f64,
) {
    let r = unsafe { &(*results).results };
    let out = unsafe { slice::from_raw_parts_mut(out_activations, r.len()) };
    for (i, res) in r.iter().enumerate() {
        out[i] = res.activation;
    }
}

/// Copy result distances into caller-owned buffer.
///
/// # Safety
/// - `results` must be valid.
/// - `out_distances` must point to at least `sproink_results_len()` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_distances(
    results: *const SproinkResults,
    out_distances: *mut u32,
) {
    let r = unsafe { &(*results).results };
    let out = unsafe { slice::from_raw_parts_mut(out_distances, r.len()) };
    for (i, res) in r.iter().enumerate() {
        out[i] = res.distance;
    }
}

/// Free a results handle.
///
/// # Safety
/// `results` must be a valid pointer from `sproink_activate`, and must not be used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_free(results: *mut SproinkResults) {
    if !results.is_null() {
        drop(unsafe { Box::from_raw(results) });
    }
}

/// Compute Oja weight update (pure function, no handle needed).
#[unsafe(no_mangle)]
pub extern "C" fn sproink_oja_update(
    current_weight: f64,
    activation_a: f64,
    activation_b: f64,
    learning_rate: f64,
    min_weight: f64,
    max_weight: f64,
) -> f64 {
    let cfg = crate::hebbian::HebbianConfig {
        learning_rate,
        min_weight,
        max_weight,
        activation_threshold: 0.0, // not used by oja_update
    };
    crate::hebbian::oja_update(current_weight, activation_a, activation_b, &cfg)
}
