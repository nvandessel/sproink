//! C-compatible FFI bindings for the sproink engine.
//!
//! All functions use opaque pointer handles and catch panics at the boundary,
//! returning null pointers on failure.
//!
//! # Thread Safety
//!
//! - `SproinkGraph` is `Send + Sync` — safe to share across threads after construction.
//! - `SproinkResults` and `SproinkPairs` are NOT thread-safe — each must be
//!   accessed from a single thread at a time.
//! - Free functions (`sproink_*_free`) must be called exactly once per allocation.
//!   Double-free is undefined behavior.

use std::collections::HashSet;

use crate::engine::{Engine, PropagationConfig};
use crate::graph::{CsrGraph, EdgeInput, EdgeKind};
use crate::hebbian::{
    CoActivationPair, HebbianConfig, Learner, OjaLearner, extract_co_activation_pairs,
};
use crate::inhibition::InhibitionConfig;
use crate::types::{Activation, ActivationResult, EdgeWeight, NodeId, Seed};

pub struct SproinkGraph {
    inner: CsrGraph,
}

pub struct SproinkResults {
    results: Vec<ActivationResult>,
}

pub struct SproinkPairs {
    pairs: Vec<CoActivationPair>,
}

fn edge_kind_from_u8(v: u8) -> Option<EdgeKind> {
    match v {
        0 => Some(EdgeKind::Positive),
        1 => Some(EdgeKind::Conflicts),
        2 => Some(EdgeKind::DirectionalSuppressive),
        4 => Some(EdgeKind::FeatureAffinity),
        _ => None, // 3 (DirectionalPassive) and unknown values rejected
    }
}

// --- Graph ---

/// Builds a CSR graph from parallel arrays of edge data.
///
/// Returns a heap-allocated `SproinkGraph` pointer, or null on failure.
/// Free with [`sproink_graph_free()`].
///
/// # Resource Usage
///
/// Memory scales as O(num_nodes + num_edges). For very large `num_nodes`
/// values (e.g., > 10M), expect significant memory allocation.
///
/// # Safety
///
/// - `sources`, `targets`, `weights`, and `kinds` must all be non-null and point
///   to arrays of at least `num_edges` elements.
/// - The caller retains no ownership of the input arrays; the graph copies all data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_graph_build(
    num_nodes: u32,
    num_edges: u32,
    sources: *const u32,
    targets: *const u32,
    weights: *const f64,
    kinds: *const u8,
) -> *mut SproinkGraph {
    if sources.is_null() || targets.is_null() || weights.is_null() || kinds.is_null() {
        return std::ptr::null_mut();
    }

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let n = num_edges as usize;
        let sources = unsafe { std::slice::from_raw_parts(sources, n) };
        let targets = unsafe { std::slice::from_raw_parts(targets, n) };
        let weights = unsafe { std::slice::from_raw_parts(weights, n) };
        let kinds = unsafe { std::slice::from_raw_parts(kinds, n) };

        let mut edges = Vec::with_capacity(n);
        for i in 0..n {
            let kind = match edge_kind_from_u8(kinds[i]) {
                Some(k) => k,
                None => return std::ptr::null_mut(),
            };
            let w = weights[i].clamp(0.0, 1.0);
            edges.push(EdgeInput {
                source: NodeId::new(sources[i]),
                target: NodeId::new(targets[i]),
                weight: EdgeWeight::new_unchecked(w),
                kind,
                last_activated: None,
            });
        }

        let graph = match CsrGraph::build(num_nodes, &edges) {
            Ok(g) => g,
            Err(_) => return std::ptr::null_mut(),
        };
        Box::into_raw(Box::new(SproinkGraph { inner: graph }))
    })) {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

/// Frees a graph previously returned by [`sproink_graph_build()`].
///
/// # Safety
///
/// - `graph` must be a pointer returned by `sproink_graph_build()`, or null.
/// - Must not be called more than once on the same pointer.
///   Passing a previously-freed pointer is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_graph_free(graph: *mut SproinkGraph) {
    if !graph.is_null() {
        drop(unsafe { Box::from_raw(graph) });
    }
}

// --- Activation ---

/// Runs spreading activation on the graph.
///
/// Returns a heap-allocated `SproinkResults` pointer, or null on failure.
/// Free with [`sproink_results_free()`].
///
/// # Resource Usage
///
/// - Memory: O(num_nodes) for activation arrays. For graphs above 1024 nodes,
///   parallel execution allocates O(threads × 4 × num_nodes) transient memory.
/// - Time: O(num_nodes × num_edges × max_steps) worst case.
///
/// # Thread Safety
///
/// `SproinkGraph` is immutable after construction and may be shared across
/// threads. However, each `sproink_activate` call must use its own
/// `SproinkResults` — results are not thread-safe.
///
/// # Safety
///
/// - `graph` must be a valid pointer returned by `sproink_graph_build()`.
/// - `seed_nodes` and `seed_activations` must point to arrays of at least
///   `num_seeds` elements, or be null when `num_seeds == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_activate(
    graph: *const SproinkGraph,
    num_seeds: u32,
    seed_nodes: *const u32,
    seed_activations: *const f64,
    max_steps: u32,
    decay_factor: f64,
    spread_factor: f64,
    min_activation: f64,
    sigmoid_gain: f64,
    sigmoid_center: f64,
    inhibition_enabled: u8,
    inhibition_strength: f64,
    inhibition_breadth: u32,
) -> *mut SproinkResults {
    if graph.is_null() {
        return std::ptr::null_mut();
    }
    // Reject silent seed drops: if the caller claims to pass seeds but either
    // array pointer is null, that's a contract violation — fail loudly.
    if num_seeds > 0 && (seed_nodes.is_null() || seed_activations.is_null()) {
        return std::ptr::null_mut();
    }

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let graph_ref = unsafe { &(*graph).inner };
        let n = num_seeds as usize;

        let seeds: Vec<Seed> = if n > 0 {
            let nodes = unsafe { std::slice::from_raw_parts(seed_nodes, n) };
            let acts = unsafe { std::slice::from_raw_parts(seed_activations, n) };
            nodes
                .iter()
                .zip(acts.iter())
                .map(|(&node, &act)| Seed {
                    node: NodeId::new(node),
                    activation: Activation::new_unchecked(act.clamp(0.0, 1.0)),
                    source: None,
                })
                .collect()
        } else {
            vec![]
        };

        let inhibition = if inhibition_enabled != 0 {
            Some(
                InhibitionConfig::builder()
                    .strength(inhibition_strength)
                    .breadth(inhibition_breadth as usize)
                    .build(),
            )
        } else {
            None
        };

        let config = PropagationConfig {
            max_steps,
            decay_factor,
            spread_factor,
            min_activation,
            sigmoid_gain,
            sigmoid_center,
            inhibition,
            temporal_decay_rate: None,
            current_time: None,
        };

        let engine = Engine::new(graph_ref);
        let results = match engine.activate(&seeds, &config) {
            Ok(r) => r,
            Err(_) => return std::ptr::null_mut(),
        };

        Box::into_raw(Box::new(SproinkResults { results }))
    })) {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

// --- Results ---

/// Returns the number of results, or 0 if `results` is null.
///
/// # Safety
///
/// - `results` must be a valid pointer from `sproink_activate()`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_len(results: *const SproinkResults) -> u32 {
    if results.is_null() {
        return 0;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        u32::try_from(unsafe { &*results }.results.len()).unwrap_or(u32::MAX)
    }))
    .unwrap_or_default()
}

/// Copies result node IDs into `out`.
///
/// Writes at most `min(buffer_len, sproink_results_len(results))` elements.
/// The number of elements actually written is returned (bounded by both the
/// buffer capacity and the result count).
///
/// # Safety
///
/// - `results` must be a valid pointer from `sproink_activate()`.
/// - `out` must point to a buffer of at least `buffer_len` `u32` elements, or
///   be null (in which case this function is a no-op returning 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_nodes(
    results: *const SproinkResults,
    out: *mut u32,
    buffer_len: u32,
) -> u32 {
    if results.is_null() || out.is_null() {
        return 0;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let r = unsafe { &*results };
        let n = r.results.len().min(buffer_len as usize);
        for (idx, result) in r.results.iter().take(n).enumerate() {
            unsafe { out.add(idx).write(result.node.get()) };
        }
        u32::try_from(n).unwrap_or(u32::MAX)
    }))
    .unwrap_or_default()
}

/// Copies result activation values into `out`.
///
/// Writes at most `min(buffer_len, sproink_results_len(results))` elements
/// and returns the count.
///
/// # Safety
///
/// - `results` must be a valid pointer from `sproink_activate()`.
/// - `out` must point to a buffer of at least `buffer_len` `f64` elements, or
///   be null (in which case this function is a no-op returning 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_activations(
    results: *const SproinkResults,
    out: *mut f64,
    buffer_len: u32,
) -> u32 {
    if results.is_null() || out.is_null() {
        return 0;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let r = unsafe { &*results };
        let n = r.results.len().min(buffer_len as usize);
        for (idx, result) in r.results.iter().take(n).enumerate() {
            unsafe { out.add(idx).write(result.activation.get()) };
        }
        u32::try_from(n).unwrap_or(u32::MAX)
    }))
    .unwrap_or_default()
}

/// Copies result hop distances into `out`.
///
/// Writes at most `min(buffer_len, sproink_results_len(results))` elements
/// and returns the count.
///
/// # Safety
///
/// - `results` must be a valid pointer from `sproink_activate()`.
/// - `out` must point to a buffer of at least `buffer_len` `u32` elements, or
///   be null (in which case this function is a no-op returning 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_distances(
    results: *const SproinkResults,
    out: *mut u32,
    buffer_len: u32,
) -> u32 {
    if results.is_null() || out.is_null() {
        return 0;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let r = unsafe { &*results };
        let n = r.results.len().min(buffer_len as usize);
        for (idx, result) in r.results.iter().take(n).enumerate() {
            unsafe { out.add(idx).write(result.distance) };
        }
        u32::try_from(n).unwrap_or(u32::MAX)
    }))
    .unwrap_or_default()
}

/// Frees results previously returned by [`sproink_activate()`].
///
/// # Safety
///
/// - `results` must be a pointer returned by `sproink_activate()`, or null.
/// - Must not be called more than once on the same pointer.
///   Passing a previously-freed pointer is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_free(results: *mut SproinkResults) {
    if !results.is_null() {
        drop(unsafe { Box::from_raw(results) });
    }
}

// --- Co-Activation Pairs ---

/// Extracts co-activation pairs from results.
///
/// Returns a heap-allocated `SproinkPairs` pointer, or null on failure.
/// Free with [`sproink_pairs_free()`].
///
/// # Safety
///
/// - `results` must be a valid pointer from `sproink_activate()`.
/// - `seed_nodes` must point to an array of at least `num_seeds` elements,
///   or be null when `num_seeds == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_extract_pairs(
    results: *const SproinkResults,
    num_seeds: u32,
    seed_nodes: *const u32,
    activation_threshold: f64,
) -> *mut SproinkPairs {
    if results.is_null() {
        return std::ptr::null_mut();
    }
    if num_seeds > 0 && seed_nodes.is_null() {
        return std::ptr::null_mut();
    }

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let r = unsafe { &*results };

        let seeds: HashSet<NodeId> = if num_seeds > 0 && !seed_nodes.is_null() {
            let nodes = unsafe { std::slice::from_raw_parts(seed_nodes, num_seeds as usize) };
            nodes.iter().map(|&n| NodeId::new(n)).collect()
        } else {
            HashSet::new()
        };

        let config = HebbianConfig::builder()
            .activation_threshold(activation_threshold)
            .build();

        let pairs = match extract_co_activation_pairs(&r.results, &seeds, &config) {
            Ok(p) => p,
            Err(_) => return std::ptr::null_mut(),
        };
        Box::into_raw(Box::new(SproinkPairs { pairs }))
    })) {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

/// Returns the number of co-activation pairs, or 0 if `pairs` is null.
///
/// # Safety
///
/// - `pairs` must be a valid pointer from `sproink_extract_pairs()`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_pairs_len(pairs: *const SproinkPairs) -> u32 {
    if pairs.is_null() {
        return 0;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        u32::try_from(unsafe { &*pairs }.pairs.len()).unwrap_or(u32::MAX)
    }))
    .unwrap_or_default()
}

/// Copies pair node IDs into `out_a` and `out_b`.
///
/// Writes at most `min(buffer_len, sproink_pairs_len(pairs))` elements to
/// each output buffer and returns the count.
///
/// # Safety
///
/// - `pairs` must be a valid pointer from `sproink_extract_pairs()`.
/// - `out_a` and `out_b` must each point to buffers of at least `buffer_len`
///   `u32` elements, or be null (in which case this function is a no-op
///   returning 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_pairs_nodes(
    pairs: *const SproinkPairs,
    out_a: *mut u32,
    out_b: *mut u32,
    buffer_len: u32,
) -> u32 {
    if pairs.is_null() || out_a.is_null() || out_b.is_null() {
        return 0;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let p = unsafe { &*pairs };
        let n = p.pairs.len().min(buffer_len as usize);
        // Write through raw pointers to avoid creating two &mut slices
        // that could overlap if caller passes same pointer for out_a and out_b.
        for (idx, pair) in p.pairs.iter().take(n).enumerate() {
            unsafe {
                out_a.add(idx).write(pair.node_a.get());
                out_b.add(idx).write(pair.node_b.get());
            }
        }
        u32::try_from(n).unwrap_or(u32::MAX)
    }))
    .unwrap_or_default()
}

/// Copies pair activation values into `out_a` and `out_b`.
///
/// Writes at most `min(buffer_len, sproink_pairs_len(pairs))` elements to
/// each output buffer and returns the count.
///
/// # Safety
///
/// - `pairs` must be a valid pointer from `sproink_extract_pairs()`.
/// - `out_a` and `out_b` must each point to buffers of at least `buffer_len`
///   `f64` elements, or be null (in which case this function is a no-op
///   returning 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_pairs_activations(
    pairs: *const SproinkPairs,
    out_a: *mut f64,
    out_b: *mut f64,
    buffer_len: u32,
) -> u32 {
    if pairs.is_null() || out_a.is_null() || out_b.is_null() {
        return 0;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let p = unsafe { &*pairs };
        let n = p.pairs.len().min(buffer_len as usize);
        // Write through raw pointers to avoid creating two &mut slices
        // that could overlap if caller passes same pointer for out_a and out_b.
        for (idx, pair) in p.pairs.iter().take(n).enumerate() {
            unsafe {
                out_a.add(idx).write(pair.activation_a.get());
                out_b.add(idx).write(pair.activation_b.get());
            }
        }
        u32::try_from(n).unwrap_or(u32::MAX)
    }))
    .unwrap_or_default()
}

/// Frees pairs previously returned by [`sproink_extract_pairs()`].
///
/// # Safety
///
/// - `pairs` must be a pointer returned by `sproink_extract_pairs()`, or null.
/// - Must not be called more than once on the same pointer.
///   Passing a previously-freed pointer is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_pairs_free(pairs: *mut SproinkPairs) {
    if !pairs.is_null() {
        drop(unsafe { Box::from_raw(pairs) });
    }
}

// --- Learning ---

/// Computes a single Oja weight update.
///
/// Returns the updated weight, or `min_weight` on internal failure or invalid
/// inputs. Inputs are rejected (and `min_weight` returned) if any of:
///
/// - `current_weight`, `activation_a`, `activation_b`, `learning_rate`,
///   `min_weight`, or `max_weight` is NaN or infinite
/// - `learning_rate < 0.0`
/// - `min_weight < 0.0` or `max_weight > 1.0`
/// - `min_weight > max_weight`
///
/// # Safety
///
/// This function has no pointer parameters and is always safe to call.
/// It is marked `unsafe extern "C"` only for FFI compatibility.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_oja_update(
    current_weight: f64,
    activation_a: f64,
    activation_b: f64,
    learning_rate: f64,
    min_weight: f64,
    max_weight: f64,
) -> f64 {
    // Validate all inputs up front. Any NaN/Inf or out-of-range config is an
    // immediate failure returning `min_weight` (documented error sentinel).
    // Checking `is_finite` on every f64 parameter also rejects NaN, which
    // would otherwise survive `clamp(0.0, 1.0)` untouched and corrupt the
    // Oja computation downstream.
    if !current_weight.is_finite()
        || !activation_a.is_finite()
        || !activation_b.is_finite()
        || !learning_rate.is_finite()
        || !min_weight.is_finite()
        || !max_weight.is_finite()
        || learning_rate < 0.0
        || min_weight < 0.0
        || max_weight > 1.0
        || min_weight > max_weight
    {
        return min_weight;
    }

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let w = EdgeWeight::new_unchecked(current_weight.clamp(0.0, 1.0));
        let a = Activation::new_unchecked(activation_a.clamp(0.0, 1.0));
        let b = Activation::new_unchecked(activation_b.clamp(0.0, 1.0));
        let config = HebbianConfig::builder()
            .learning_rate(learning_rate)
            .min_weight(min_weight)
            .max_weight(max_weight)
            .build();
        OjaLearner.update_weight(w, a, b, &config).get()
    })) {
        Ok(v) => v,
        Err(_) => min_weight,
    }
}
