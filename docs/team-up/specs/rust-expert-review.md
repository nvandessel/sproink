# Rust Expert Review: Sproink Specification

**Reviewer:** Rust Domain Expert
**Date:** 2026-04-02
**Verdict:** APPROVED with recommendations (no blockers)

---

## 1. Trait Hierarchy Design

### 1.1 `Propagator::activate` should use generics, not `dyn Graph`

The spec has:
```rust
fn activate(&self, graph: &dyn Graph, seeds: &[Seed], config: &PropagationConfig) -> Vec<ActivationResult>;
```

This is a hot-path function called every propagation cycle. Dynamic dispatch here means:
- vtable indirection on every `graph.neighbors()` call (inner loop)
- prevents inlining of `neighbors()`, which is a trivial slice index
- prevents the compiler from seeing through to the CSR layout for auto-vectorization

**Recommendation:** Use a generic parameter with static dispatch:

```rust
pub trait Propagator: Send + Sync {
    fn activate<G: Graph>(
        &self,
        graph: &G,
        seeds: &[Seed],
        config: &PropagationConfig,
    ) -> Vec<ActivationResult>;
}
```

**Caveat:** This makes `Propagator` non-object-safe (generic method). If you need `dyn Propagator` (e.g., for runtime strategy switching), use a concrete method instead:

```rust
// Option A: Make Engine generic over G, not the trait method
pub struct Engine<G: Graph> {
    graph: G,
}

impl<G: Graph> Engine<G> {
    pub fn activate(&self, seeds: &[Seed], config: &PropagationConfig) -> Vec<ActivationResult> {
        // graph is a field, not a parameter — monomorphized once
    }
}
```

```rust
// Option B: Keep trait but accept the object-safety tradeoff
// If you truly need trait-based DI for testing:
pub trait Propagator: Send + Sync {
    fn activate(
        &self,
        graph: &dyn Graph,  // accept vtable cost for testability
        seeds: &[Seed],
        config: &PropagationConfig,
    ) -> Vec<ActivationResult>;
}
```

I'd recommend **Option A** — the `Engine` struct owns the graph reference and is generic over `G`. This gives zero-cost dispatch in production while still allowing mock graphs in tests. The trait-based DI approach from the spec adds indirection without clear benefit since there's only one `Propagator` implementation.

### 1.2 `Inhibitor` and `Squasher` take `&mut [f64]` — type safety gap

The spec defines:
```rust
fn inhibit(&self, activations: &mut [f64], config: &InhibitionConfig);
fn squash(&self, activations: &mut [f64]);
```

But the crate defines an `Activation` newtype. Using raw `f64` here breaks the type safety boundary — a caller could accidentally pass a weight vector.

**Recommendation:** Use `&mut [Activation]` consistently, or accept `&mut [f64]` as an intentional performance/ergonomics choice and document why:

```rust
// Option A: Type-safe (preferred)
pub trait Inhibitor: Send + Sync {
    fn inhibit(&self, activations: &mut [Activation], config: &InhibitionConfig);
}

pub trait Squasher: Send + Sync {
    fn squash(&self, activations: &mut [Activation]);
}
```

```rust
// Option B: Raw f64 for performance — document the decision
// Activation newtype is used at API boundaries (input/output).
// Internally, the engine works on raw f64 slices to avoid
// wrapping/unwrapping overhead in tight loops.
```

If you go with Option B, the `Activation` newtype should have a method to convert a slice in-place:

```rust
impl Activation {
    /// Reinterpret a slice of Activations as raw f64s.
    /// SAFETY: Activation is #[repr(transparent)] over f64.
    #[inline]
    pub fn as_f64_slice(slice: &[Activation]) -> &[f64] {
        // Safe because Activation is repr(transparent)
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f64, slice.len()) }
    }

    #[inline]
    pub fn as_f64_slice_mut(slice: &mut [Activation]) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f64, slice.len()) }
    }
}
```

### 1.3 `Squasher` trait — over-abstraction?

The `Squasher` trait wraps a single sigmoid function. There's only one implementation (`SigmoidSquasher`), and the spec explicitly lists only sigmoid as the squashing function.

**Recommendation:** Start without the trait. Use a plain function:

```rust
/// Apply sigmoid squashing in-place.
#[inline]
pub fn squash_sigmoid(activations: &mut [f64], gain: f64, center: f64) {
    for v in activations.iter_mut() {
        *v = 1.0 / (1.0 + (-gain * (*v - center)).exp());
    }
}
```

If you later need pluggable squashing (ReLU, tanh, etc.), promote to a trait then. YAGNI applies here — the trait adds a vtable or monomorphization cost for zero current benefit.

**If you keep the trait:** At minimum make it generic on the `Engine` so it's statically dispatched:

```rust
pub struct Engine<G: Graph, S: Squasher> {
    graph: G,
    squasher: S,
}
```

---

## 2. Newtype Design

### 2.1 `Deref<Target = f64>` on `EdgeWeight` and `Activation` — problematic

The spec says both implement `Deref<Target = f64>`. The research document explicitly warns against this:

> Don't implement `Deref` to the inner type for domain newtypes — it defeats the purpose by allowing implicit coercion

With `Deref`, this compiles silently:

```rust
let w = EdgeWeight::new(0.5).unwrap();
let a = Activation::new(0.3).unwrap();
let oops: f64 = *w + *a;  // Compiles — is this meaningful?
```

More insidiously, any function taking `&f64` will accept `&EdgeWeight` via auto-deref. The newtype boundary becomes porous.

**Recommendation:** Instead of `Deref`, provide explicit `.value()` or `.get()` accessors and implement arithmetic ops via `derive_more` or manually:

```rust
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Activation(f64);

impl Activation {
    pub fn new(v: f64) -> Result<Self, SproinkError> {
        if v.is_nan() || v.is_infinite() || v < 0.0 || v > 1.0 {
            return Err(SproinkError::InvalidValue { field: "activation", value: v });
        }
        Ok(Self(v))
    }

    /// Access the raw value. Use when you need f64 arithmetic.
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }

    /// Construct without validation — for internal use where bounds are guaranteed.
    /// E.g., after sigmoid which always produces [0, 1].
    #[inline]
    pub(crate) fn new_unchecked(v: f64) -> Self {
        debug_assert!(!v.is_nan() && v >= 0.0 && v <= 1.0);
        Self(v)
    }
}
```

### 2.2 `NodeId(pub u32)` — public inner field

The `pub` inner field means anyone can write `NodeId(999999)` without bounds checking. This is actually fine for a dense index type where the valid range depends on the graph instance, not the type itself. Runtime bounds checking happens at the graph API boundary (`neighbors()` would panic on out-of-bounds).

**Recommendation:** Keep `pub` inner for `NodeId` and `TagId` — these are index types, not constrained-value types. But add `#[repr(transparent)]` for FFI safety:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
```

### 2.3 `derive_more` for arithmetic

**Recommendation:** Use `derive_more` for `From`/`Into`/`Display` on newtypes, but **don't** derive arithmetic ops on `EdgeWeight`/`Activation`. These types have invariants (bounded to [0,1]) that arithmetic ops would silently violate. Arithmetic should happen on raw `f64` values (via `.get()`), with the result re-validated at the boundary.

```toml
derive_more = { version = "1", features = ["from", "into", "display"] }
```

---

## 3. Ownership Model

### 3.1 Engine borrowing the graph — correct

The engine reading from an immutable graph reference is the right pattern. The graph is built once and used immutably — a borrow expresses this perfectly.

**Recommendation:** Make the lifetime explicit if using a reference, or use generics:

```rust
// Option A: Lifetime-annotated borrow
pub struct Engine<'g> {
    graph: &'g CsrGraph,
}

// Option B: Generic (preferred — works with both owned and borrowed)
pub struct Engine<G: Graph> {
    graph: G,  // Can be CsrGraph, &CsrGraph, Arc<CsrGraph>, MockGraph, etc.
}
```

Option B is more flexible — in production, floop can pass an owned `CsrGraph`. In tests, pass a `MockGraph`. No lifetime annotation noise.

### 3.2 Activation vectors: `Vec<f64>` vs `Vec<Activation>`

The hot-loop propagation code manipulates activations hundreds of thousands of times. Using `Vec<Activation>` in the inner loop means wrapping/unwrapping at every arithmetic operation (even with `Deref`, you'd need to re-wrap after computation).

**Recommendation:** Use `Vec<f64>` internally for computation, `Vec<Activation>` at API boundaries:

```rust
impl<G: Graph> Engine<G> {
    pub fn activate(&self, seeds: &[Seed], config: &PropagationConfig) -> Vec<ActivationResult> {
        // Internal: raw f64 for performance
        let mut current = vec![0.0f64; self.graph.num_nodes() as usize];
        let mut next = vec![0.0f64; self.graph.num_nodes() as usize];

        // ... propagation loop operates on raw f64 ...

        // Boundary: wrap into Activation for output
        current.iter().enumerate()
            .filter(|(_, &v)| v > config.min_activation)
            .map(|(i, &v)| ActivationResult {
                node: NodeId(i as u32),
                activation: Activation::new_unchecked(v),
                distance: distances[i],
            })
            .collect()
    }
}
```

---

## 4. Concurrency Patterns

### 4.1 par_iter -> collect updates -> sequential merge

The spec describes: parallel per-node computation producing `(target, delta, distance)` updates, merged sequentially. This is reasonable but there's a more cache-friendly pattern.

**Recommendation:** Use double-buffered parallel write instead of collect-then-merge:

```rust
use rayon::prelude::*;

fn propagation_step<G: Graph>(
    graph: &G,
    current: &[f64],
    next: &mut [f64],
    distances: &mut [u32],
    config: &PropagationConfig,
) {
    // Zero out next buffer
    next.iter_mut().for_each(|v| *v = 0.0);

    // Phase 1: Compute per-SOURCE contributions into thread-local buffers
    // then merge. This avoids write contention on `next[target]`.
    let num_nodes = graph.num_nodes() as usize;

    // Collect updates from all active sources
    let updates: Vec<(usize, f64, u32)> = (0..num_nodes)
        .into_par_iter()
        .filter(|&i| current[i] >= config.min_activation)
        .flat_map_iter(|i| {
            let node = NodeId(i as u32);
            let neighbors = graph.neighbors(node);
            // ... compute per-neighbor contributions ...
            neighbors.iter().map(move |edge| {
                let energy = current[i] * config.spread_factor
                    * edge.weight.get() * config.decay_factor;
                (edge.target.index(), energy, distances[i] + 1)
            })
        })
        .collect();

    // Phase 2: Sequential merge (max-semantics requires this)
    for (target, energy, dist) in updates {
        next[target] = next[target].max(energy);
        distances[target] = distances[target].min(dist);
    }
}
```

The key insight: **max-semantics prevents purely parallel writes** because `max(a, b)` across threads requires synchronization. The collect-then-merge approach from the spec is actually the right call. An alternative is per-thread local buffers merged at the end, which avoids the large intermediate `Vec`:

```rust
use rayon::iter::ParallelIterator;
use std::cell::UnsafeCell;

// Thread-local accumulation buffers — merged after parallel phase
fn propagation_step_threadlocal<G: Graph>(
    graph: &G,
    current: &[f64],
    next: &mut [f64],
    config: &PropagationConfig,
) {
    let num_nodes = graph.num_nodes() as usize;
    next.iter_mut().for_each(|v| *v = 0.0);

    // Use Rayon's fold + reduce for thread-local accumulation
    let merged: Vec<f64> = (0..num_nodes)
        .into_par_iter()
        .filter(|&i| current[i] >= config.min_activation)
        .fold(
            || vec![0.0f64; num_nodes],  // thread-local buffer
            |mut local, i| {
                let node = NodeId(i as u32);
                for edge in graph.neighbors(node) {
                    let energy = current[i] * config.spread_factor
                        * edge.weight.get() * config.decay_factor;
                    // Max-semantics within this thread's view
                    let t = edge.target.index();
                    local[t] = local[t].max(energy);
                }
                local
            },
        )
        .reduce(
            || vec![0.0f64; num_nodes],
            |mut a, b| {
                // Merge thread-local buffers with max-semantics
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x = x.max(*y);
                }
                a
            },
        );

    next.copy_from_slice(&merged);
}
```

**Trade-off:** The `fold + reduce` approach allocates one `Vec<f64>` per Rayon thread (typically 4-16), which is ~400KB per thread for 50k nodes. Acceptable. The intermediate-`Vec` approach from the spec allocates proportional to total edges, which could be larger. Benchmark both.

### 4.2 Send/Sync concerns

All traits already require `Send + Sync`. The concrete types (`CsrGraph`, config structs) are composed of `Vec`, `u32`, `f64` — all `Send + Sync`. No issues here.

**One concern:** If `Engine` holds `dyn Graph` (trait object), you need `Box<dyn Graph + Send + Sync>`. The spec's `Graph: Send + Sync` bound covers this. Good.

---

## 5. FFI Design

### 5.1 Opaque handles via `Box::into_raw` — standard and correct

This is the canonical Rust FFI pattern. The ownership model in section 7.3 is well-documented.

### 5.2 `catch_unwind` at every FFI entry point — yes, mandatory

The spec correctly mandates this. One refinement — use a helper macro to reduce boilerplate:

```rust
/// Wrap an FFI entry point with null-check and panic safety.
macro_rules! ffi_entry {
    ($ptr:expr, $default:expr, $body:expr) => {{
        if $ptr.is_null() {
            return $default;
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(v) => v,
            Err(_) => $default,
        }
    }};
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_results_len(results: *const SproinkResults) -> u32 {
    ffi_entry!(results, 0, {
        let results = unsafe { &*(results as *const Vec<ActivationResult>) };
        results.len() as u32
    })
}
```

### 5.3 `cbindgen` for header generation — yes

The research confirms this is standard practice. Add it as a build dependency with a `build.rs`:

```rust
// build.rs
fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::C)
        .with_include_guard("SPROINK_H")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("sproink.h");
}
```

### 5.4 Missing: `#[repr(C)]` on `EdgeKind`

The spec has `#[repr(u8)]` on `EdgeKind`, which is correct for FFI. But the FFI functions accept `uint8_t` for kind values and map them manually. This is the right approach — don't pass Rust enums across FFI, pass integers and validate on the Rust side.

### 5.5 Double-free safety

The spec mentions "double-free safety: calling `_free` twice doesn't crash (idempotent)" in testing. This is **not achievable** with the `Box::into_raw` / `Box::from_raw` pattern — calling `from_raw` on an already-freed pointer is UB.

**Recommendation:** Document that double-free is **caller's responsibility to avoid**, not a property of the API. The test should verify that calling `_free` once works correctly, and that the documentation clearly states single-free semantics:

```rust
/// Free a graph handle. The handle MUST NOT be used after this call.
/// Passing the same handle twice is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_graph_free(graph: *mut SproinkGraph) {
    if !graph.is_null() {
        drop(unsafe { Box::from_raw(graph as *mut CsrGraph) });
    }
}
```

The null check prevents crash on `free(NULL)` (C convention), but double-free of a non-null pointer cannot be made safe.

---

## 6. Testing Patterns

### 6.1 Mockall compatibility

The traits as specified are mockall-compatible **except** if `Propagator::activate` becomes generic (recommendation 1.1). A generic method makes the trait non-object-safe and incompatible with `#[automock]`.

**Solutions:**

```rust
// Option A: If using generic Engine (recommended), mock the Graph instead
#[cfg(test)]
use mockall::automock;

#[cfg_attr(test, automock)]
pub trait Graph: Send + Sync {
    fn num_nodes(&self) -> u32;
    fn neighbors(&self, node: NodeId) -> &[EdgeData];
}

// Then test Engine directly with MockGraph:
#[test]
fn test_propagation() {
    let mut mock = MockGraph::new();
    mock.expect_num_nodes().returning(|| 3);
    mock.expect_neighbors()
        .with(eq(NodeId(0)))
        .returning(|_| &EDGES_FROM_0);
    let engine = Engine::new(mock);
    let results = engine.activate(&seeds, &config);
    // ...
}
```

```rust
// Option B: If keeping trait-based Propagator with dyn Graph,
// mockall works directly on all traits as specified
#[cfg_attr(test, automock)]
pub trait Propagator: Send + Sync {
    fn activate(&self, graph: &dyn Graph, seeds: &[Seed], config: &PropagationConfig)
        -> Vec<ActivationResult>;
}
```

Note: `MockGraph::neighbors()` returning `&[EdgeData]` requires the mock to own the data. Use `returning` with a static or leaked slice, or use a real small graph instead of mocking.

### 6.2 Proptest strategies — good, one addition

The spec's proptest properties are solid. Add one more:

```rust
proptest! {
    #[test]
    fn propagation_is_idempotent_on_zero_seeds(
        num_nodes in 1u32..100,
    ) {
        // No seeds → no activation → empty results
        let graph = build_random_graph(num_nodes, num_nodes * 2);
        let engine = Engine::new(graph);
        let results = engine.activate(&[], &PropagationConfig::default());
        prop_assert!(results.is_empty());
    }

    #[test]
    fn oja_weight_stays_bounded(
        w in 0.01f64..0.95,
        a_i in 0.0f64..1.0,
        a_j in 0.0f64..1.0,
        iterations in 1usize..1000,
    ) {
        let config = HebbianConfig::default();
        let mut weight = EdgeWeight::new(w).unwrap();
        for _ in 0..iterations {
            weight = oja_update(weight, Activation::new_unchecked(a_i),
                               Activation::new_unchecked(a_j), &config);
            let v = weight.get();
            prop_assert!(v >= config.min_weight && v <= config.max_weight,
                "Weight {} escaped bounds after Oja update", v);
        }
    }
}
```

---

## 7. What's Missing or Wrong

### 7.1 Missing: `#[repr(transparent)]` on newtypes

`NodeId`, `EdgeWeight`, `Activation`, and `TagId` should all have `#[repr(transparent)]` to guarantee layout equivalence with their inner types. This is essential for safe transmutation (e.g., reinterpreting `&[Activation]` as `&[f64]`) and for FFI.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct NodeId(pub u32);
```

### 7.2 Missing: `#[deny(unsafe_op_in_unsafe_fn)]` crate attribute

The research recommends this and it should be in the spec:

```rust
// lib.rs
#![deny(unsafe_op_in_unsafe_fn)]
```

This forces every unsafe operation inside `unsafe fn` to have its own `unsafe` block with a safety comment, improving auditability.

### 7.3 Missing: `thiserror` derive on `SproinkError`

The spec's error type is:
```rust
#[derive(Debug)]
pub enum SproinkError {
    InvalidValue { field: &'static str, value: f64 },
    NodeOutOfBounds { node: NodeId, num_nodes: u32 },
}
```

It should derive `thiserror::Error` with `#[error(...)]` attributes for Display impl:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SproinkError {
    #[error("invalid {field}: {value} (must be finite and in valid range)")]
    InvalidValue { field: &'static str, value: f64 },

    #[error("node {node:?} out of bounds (graph has {num_nodes} nodes)")]
    NodeOutOfBounds { node: NodeId, num_nodes: u32 },
}
```

### 7.4 Missing: minimum work threshold for Rayon

The spec says "sub-millisecond propagation for graphs up to ~50k nodes" but doesn't mention falling back to sequential for small graphs. For graphs under ~1000 nodes, Rayon's thread-pool overhead exceeds the parallelism benefit.

**Recommendation:** Add a config parameter or hardcoded threshold:

```rust
const PARALLEL_THRESHOLD: usize = 1024;

fn propagation_step<G: Graph>(graph: &G, current: &[f64], next: &mut [f64], config: &PropagationConfig) {
    let num_nodes = graph.num_nodes() as usize;
    if num_nodes < PARALLEL_THRESHOLD {
        propagation_step_sequential(graph, current, next, config);
    } else {
        propagation_step_parallel(graph, current, next, config);
    }
}
```

### 7.5 `EdgeKind` value 3 (`DirectionalPassive`) not accepted at FFI but exists in the enum

The spec says DirectionalPassive is "internal only; not accepted at the FFI boundary." This is correct but should be enforced at the type level or at minimum documented in the FFI validation:

```rust
fn edge_kind_from_u8(v: u8) -> EdgeKind {
    match v {
        0 => EdgeKind::Positive,
        1 => EdgeKind::Conflicts,
        2 => EdgeKind::DirectionalSuppressive,
        4 => EdgeKind::FeatureAffinity,
        // 3 (DirectionalPassive) is intentionally rejected — it's created
        // internally during bidirectional edge expansion
        _ => EdgeKind::Positive,  // default for unknown
    }
}
```

### 7.6 Consider `typed-builder` for config structs

`PropagationConfig` has 6 fields, `InhibitionConfig` has 3, `HebbianConfig` has 4, `AffinityConfig` has 2. The larger configs would benefit from `TypedBuilder`:

```rust
use typed_builder::TypedBuilder;

#[derive(Debug, Clone, TypedBuilder)]
pub struct PropagationConfig {
    #[builder(default = 3)]
    pub max_steps: u32,
    #[builder(default = 0.7)]
    pub decay_factor: f64,
    #[builder(default = 0.85)]
    pub spread_factor: f64,
    #[builder(default = 0.01)]
    pub min_activation: f64,
    #[builder(default = 10.0)]
    pub sigmoid_gain: f64,
    #[builder(default = 0.3)]
    pub sigmoid_center: f64,
}
```

---

## Summary

| Area | Assessment | Action |
|------|-----------|--------|
| Trait hierarchy | Good structure, minor dispatch concern | Use generic `Engine<G: Graph>` instead of `dyn Graph` in hot path |
| `Squasher` trait | Over-abstraction | Replace with plain function |
| `Deref` on newtypes | Anti-pattern | Use `.get()` accessor instead |
| `NodeId(pub u32)` | Fine for index type | Keep, add `#[repr(transparent)]` |
| Ownership model | Correct | Use `Engine<G: Graph>` for flexibility |
| Internal activation repr | `Vec<f64>` correct for hot path | Document boundary between internal f64 and API Activation |
| Rayon pattern | Collect-then-merge is correct for max-semantics | Consider fold+reduce; add parallel threshold |
| Send/Sync | Covered | No issues |
| FFI opaque handles | Standard pattern | Add `ffi_entry!` macro, use cbindgen |
| `catch_unwind` | Mandatory, spec is correct | Add helper macro |
| Double-free claim | Incorrect — UB with Box pattern | Document as caller responsibility |
| Mockall compat | Works if traits stay non-generic | Mock Graph instead of Propagator |
| Proptest | Good coverage | Add Oja convergence iteration test |
| Error handling | Missing thiserror | Add derive(Error) |
| Missing repr(transparent) | All newtypes | Add to all four newtypes |

**Overall:** The spec is well-designed. The trait hierarchy is clean, the algorithm is well-specified, and the FFI contract is solid. The recommendations above are refinements, not structural changes. No blockers to proceeding with implementation.
