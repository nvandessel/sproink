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

fn edge_kind_from_u8(v: u8) -> EdgeKind {
    match v {
        0 => EdgeKind::Positive,
        1 => EdgeKind::Conflicts,
        2 => EdgeKind::DirectionalSuppressive,
        // 3 = DirectionalPassive: not accepted at FFI boundary, default to Positive
        4 => EdgeKind::FeatureAffinity,
        _ => EdgeKind::Positive,
    }
}

// --- Graph ---

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
            let w = weights[i].clamp(0.0, 1.0);
            edges.push(EdgeInput {
                source: NodeId(sources[i]),
                target: NodeId(targets[i]),
                weight: EdgeWeight::new_unchecked(w),
                kind: edge_kind_from_u8(kinds[i]),
            });
        }

        let graph = CsrGraph::build(num_nodes, edges);
        Box::into_raw(Box::new(SproinkGraph { inner: graph }))
    })) {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_graph_free(graph: *mut SproinkGraph) {
    if !graph.is_null() {
        drop(unsafe { Box::from_raw(graph) });
    }
}

// --- Activation ---

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
    inhibition_enabled: bool,
    inhibition_strength: f64,
    inhibition_breadth: u32,
) -> *mut SproinkResults {
    if graph.is_null() {
        return std::ptr::null_mut();
    }

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let graph_ref = unsafe { &(*graph).inner };
        let n = num_seeds as usize;

        let seeds: Vec<Seed> = if n > 0 && !seed_nodes.is_null() && !seed_activations.is_null() {
            let nodes = unsafe { std::slice::from_raw_parts(seed_nodes, n) };
            let acts = unsafe { std::slice::from_raw_parts(seed_activations, n) };
            nodes
                .iter()
                .zip(acts.iter())
                .map(|(&node, &act)| Seed {
                    node: NodeId(node),
                    activation: Activation::new_unchecked(act.clamp(0.0, 1.0)),
                })
                .collect()
        } else {
            vec![]
        };

        let inhibition = if inhibition_enabled {
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
        };

        let engine = Engine::new(graph_ref);
        let results = engine.activate(&seeds, &config);

        Box::into_raw(Box::new(SproinkResults { results }))
    })) {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

// --- Results ---

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_len(results: *const SproinkResults) -> u32 {
    if results.is_null() {
        return 0;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { &*results }.results.len() as u32
    }))
    .unwrap_or_default()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_nodes(results: *const SproinkResults, out: *mut u32) {
    if results.is_null() || out.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let r = unsafe { &*results };
        let out = unsafe { std::slice::from_raw_parts_mut(out, r.results.len()) };
        for (slot, result) in out.iter_mut().zip(&r.results) {
            *slot = result.node.get();
        }
    }));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_activations(
    results: *const SproinkResults,
    out: *mut f64,
) {
    if results.is_null() || out.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let r = unsafe { &*results };
        let out = unsafe { std::slice::from_raw_parts_mut(out, r.results.len()) };
        for (slot, result) in out.iter_mut().zip(&r.results) {
            *slot = result.activation.get();
        }
    }));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_distances(results: *const SproinkResults, out: *mut u32) {
    if results.is_null() || out.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let r = unsafe { &*results };
        let out = unsafe { std::slice::from_raw_parts_mut(out, r.results.len()) };
        for (slot, result) in out.iter_mut().zip(&r.results) {
            *slot = result.distance;
        }
    }));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_free(results: *mut SproinkResults) {
    if !results.is_null() {
        drop(unsafe { Box::from_raw(results) });
    }
}

// --- Co-Activation Pairs ---

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

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let r = unsafe { &*results };

        let seeds: HashSet<NodeId> = if num_seeds > 0 && !seed_nodes.is_null() {
            let nodes = unsafe { std::slice::from_raw_parts(seed_nodes, num_seeds as usize) };
            nodes.iter().map(|&n| NodeId(n)).collect()
        } else {
            HashSet::new()
        };

        let config = HebbianConfig::builder()
            .activation_threshold(activation_threshold)
            .build();

        let pairs = extract_co_activation_pairs(&r.results, &seeds, &config);
        Box::into_raw(Box::new(SproinkPairs { pairs }))
    })) {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_pairs_len(pairs: *const SproinkPairs) -> u32 {
    if pairs.is_null() {
        return 0;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { &*pairs }.pairs.len() as u32
    }))
    .unwrap_or_default()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_pairs_nodes(
    pairs: *const SproinkPairs,
    out_a: *mut u32,
    out_b: *mut u32,
) {
    if pairs.is_null() || out_a.is_null() || out_b.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let p = unsafe { &*pairs };
        let out_a = unsafe { std::slice::from_raw_parts_mut(out_a, p.pairs.len()) };
        let out_b = unsafe { std::slice::from_raw_parts_mut(out_b, p.pairs.len()) };
        for ((sa, sb), pair) in out_a.iter_mut().zip(out_b.iter_mut()).zip(&p.pairs) {
            *sa = pair.node_a.get();
            *sb = pair.node_b.get();
        }
    }));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_pairs_activations(
    pairs: *const SproinkPairs,
    out_a: *mut f64,
    out_b: *mut f64,
) {
    if pairs.is_null() || out_a.is_null() || out_b.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let p = unsafe { &*pairs };
        let out_a = unsafe { std::slice::from_raw_parts_mut(out_a, p.pairs.len()) };
        let out_b = unsafe { std::slice::from_raw_parts_mut(out_b, p.pairs.len()) };
        for ((sa, sb), pair) in out_a.iter_mut().zip(out_b.iter_mut()).zip(&p.pairs) {
            *sa = pair.activation_a.get();
            *sb = pair.activation_b.get();
        }
    }));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_pairs_free(pairs: *mut SproinkPairs) {
    if !pairs.is_null() {
        drop(unsafe { Box::from_raw(pairs) });
    }
}

// --- Learning ---

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_oja_update(
    current_weight: f64,
    activation_a: f64,
    activation_b: f64,
    learning_rate: f64,
    min_weight: f64,
    max_weight: f64,
) -> f64 {
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
