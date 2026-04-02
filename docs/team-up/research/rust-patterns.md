# Rust Patterns & Libraries Research for Sproink

> Researched 2026-04-02. Targeting Rust edition 2024, stable toolchain.

---

## 1. Trait-Based Dependency Injection

### Current Best Practice

Rust doesn't need DI frameworks. Traits + generics **are** the DI system. The idiomatic approach:

- **Generics (static dispatch)** for hot paths — zero overhead, monomorphized at compile time
- **`dyn Trait` (dynamic dispatch)** only when you need heterogeneous collections or runtime polymorphism (plugin systems, etc.)

For sproink, where the graph engine is performance-critical, **generics are the right default**.

### Pattern: Constructor Injection via Generics

```rust
pub trait ActivationFunction: Send + Sync {
    fn apply(&self, input: f64) -> f64;
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn apply(&self, input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }
}

// Generic — monomorphized, zero-cost
pub struct Engine<A: ActivationFunction> {
    activation: A,
    // ...
}

impl<A: ActivationFunction> Engine<A> {
    pub fn new(activation: A) -> Self {
        Self { activation }
    }
}
```

### When to Use `dyn Trait`

```rust
// Only if you need runtime polymorphism (e.g., user-configurable activation)
pub struct DynEngine {
    activation: Box<dyn ActivationFunction>,
}
```

### C# Comparison

| C# | Rust |
|---|---|
| `interface IFoo` | `trait Foo` |
| Constructor injection via DI container | Pass trait impl as generic param or value |
| `IServiceCollection.AddSingleton<IFoo, FooImpl>()` | `Engine::new(Sigmoid)` — no container needed |

### Pitfalls

- Don't reach for `dyn Trait` by default — it adds vtable overhead and prevents inlining
- Don't build a DI container — Rust's type system makes it unnecessary
- Trait objects require object safety (no `Self` in return position, no generic methods)

---

## 2. Newtype Patterns

### Current Best Practice

Newtypes are zero-cost abstractions (`#[repr(transparent)]`) that add type safety. Essential for domain types like `NodeId`, `EdgeWeight`.

### Recommended Crate: `derive_more` 1.x

```toml
[dependencies]
derive_more = { version = "1", features = ["from", "into", "display", "deref"] }
```

### Pattern

```rust
use derive_more::{From, Into, Display, Deref};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, From, Into, Display)]
#[repr(transparent)]
pub struct NodeId(u32);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, From, Into)]
#[repr(transparent)]
pub struct EdgeWeight(f64);

// For indexed access into slices/vecs:
impl NodeId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
```

### Key Derives to Include

| Need | Derive |
|---|---|
| Debug printing | `Debug` |
| Value semantics | `Clone, Copy` |
| Comparisons | `PartialEq, Eq` (skip `Eq` for f64) |
| HashMap keys | `Hash` (only for integer newtypes) |
| Sorting | `PartialOrd, Ord` |
| Conversions | `From, Into` (via derive_more) |
| Display | `Display` (via derive_more) |

### Pitfalls

- Don't implement `Deref` to the inner type for domain newtypes — it defeats the purpose by allowing implicit coercion
- Don't derive `Eq`/`Hash` for f64-wrapping newtypes (NaN breaks both)
- Do use `#[repr(transparent)]` for FFI compatibility and guaranteed layout

---

## 3. CSR Graph Representation

### Recommendation: Hand-Roll CSR

Petgraph has a `Csr` type, but it's the most restricted graph type in petgraph with a limited API. For sproink's specific use case (static graph, spreading activation with parallel iteration), a hand-rolled CSR is better because:

1. **Full control over memory layout** — cache-friendly iteration
2. **No unnecessary abstractions** — petgraph's generics add compile time without benefit
3. **Rayon integration** — direct parallel iteration over contiguous slices
4. **Simpler FFI** — raw arrays are trivially exposed to C

A 2025 research paper confirmed that custom implementations significantly outperform petgraph for graph traversal workloads.

### Pattern: Minimal CSR

```rust
pub struct CsrGraph {
    /// row_offsets[i] is the index into col_indices where node i's edges start.
    /// Length: num_nodes + 1
    row_offsets: Vec<u32>,
    /// Target node for each edge, packed contiguously.
    /// Length: num_edges
    col_indices: Vec<NodeId>,
    /// Weight for each edge, parallel to col_indices.
    /// Length: num_edges
    weights: Vec<EdgeWeight>,
}

impl CsrGraph {
    /// Edges outgoing from `node` as (target, weight) pairs.
    #[inline]
    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        let start = self.row_offsets[node.index()] as usize;
        let end = self.row_offsets[node.index() + 1] as usize;
        &self.col_indices[start..end]
    }

    #[inline]
    pub fn edge_weights(&self, node: NodeId) -> &[EdgeWeight] {
        let start = self.row_offsets[node.index()] as usize;
        let end = self.row_offsets[node.index() + 1] as usize;
        &self.weights[start..end]
    }

    pub fn num_nodes(&self) -> usize {
        self.row_offsets.len() - 1
    }
}
```

### Cache-Friendly Layout

- Keep `col_indices` and `weights` as separate `Vec`s (struct-of-arrays) for SIMD-friendly traversal
- Alternatively, interleave as `Vec<(NodeId, EdgeWeight)>` (array-of-structs) if you always access both together — benchmark to decide
- Sort edges by target node within each row for predictable access patterns

### Pitfalls

- Don't use `HashMap<NodeId, Vec<Edge>>` — terrible cache behavior for graph traversal
- Don't use petgraph's `Graph` (adjacency list) — O(degree) edge lookup, poor locality
- Do pre-sort edges during construction; CSR is immutable after build

---

## 4. SIMD in Stable Rust

### Current State (2025)

`std::simd` is **still nightly-only** and will remain so "for the foreseeable future." For stable Rust, the practical options are:

| Approach | Stable? | Multiversioning | f64 Support | Notes |
|---|---|---|---|---|
| Auto-vectorization | Yes | N/A | Limited* | Compiler won't optimize f64 math due to precision |
| `wide` crate | Yes | No | Yes | Mature, supports NEON/WASM/x86 |
| `pulp` crate | Yes | Yes | Yes | Powers `faer` math library |
| Raw intrinsics | Yes (since 1.87) | Manual | Yes | Most control, most effort |

*Auto-vectorization won't optimize floating-point operations on stable due to precision concerns.

### Recommendation for Sproink

**Start without SIMD.** The sigmoid function and vector dot products should be written as plain scalar loops first. If benchmarks show these are bottlenecks:

1. Try the `wide` crate for f64x4 operations
2. If you need AVX2/AVX-512 dispatch, use `pulp`

```toml
# Only add when benchmarks justify it
[dependencies]
wide = "0.7"
```

### Pattern: SIMD-Ready Scalar Code

```rust
/// Write scalar code that auto-vectorizes well:
/// - No early returns or branches in the loop
/// - Operate on contiguous slices
/// - Use simple arithmetic
pub fn apply_sigmoid(values: &mut [f64]) {
    for v in values.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

// If benchmarks demand it, explicit SIMD with `wide`:
use wide::f64x4;

pub fn apply_sigmoid_simd(values: &mut [f64]) {
    let chunks = values.chunks_exact_mut(4);
    let remainder = chunks.into_remainder();
    for chunk in values.chunks_exact_mut(4) {
        let v = f64x4::from(chunk);
        let result = f64x4::ONE / (f64x4::ONE + (-v).exp());
        chunk.copy_from_slice(&result.to_array());
    }
    // Handle remainder with scalar
    for v in remainder.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}
```

### Pitfalls

- Don't use nightly-only `std::simd` for a library crate — limits adoption
- Don't prematurely optimize with SIMD — benchmark first
- Auto-vectorization of f64 math is unreliable on stable; don't count on it

---

## 5. Rayon for Graph Algorithms

### Current Best Practice (rayon 1.10)

Rayon's `par_iter()` is ideal for spreading activation because each node's new value can be computed independently from the previous iteration's values (read old, write new).

### Pattern: Parallel Propagation

```rust
use rayon::prelude::*;

pub fn propagate(
    graph: &CsrGraph,
    current: &[f64],    // read-only previous values
    next: &mut [f64],   // write-only new values
) {
    // Each node computes independently — perfect for par_iter
    next.par_iter_mut()
        .enumerate()
        .for_each(|(i, next_val)| {
            let node = NodeId(i as u32);
            let neighbors = graph.neighbors(node);
            let weights = graph.edge_weights(node);

            let mut sum = 0.0;
            for (j, &neighbor) in neighbors.iter().enumerate() {
                sum += current[neighbor.index()] * weights[j].0;
            }
            *next_val = sigmoid(sum);
        });
}
```

### Key Optimization Insights

From a detailed 2024 benchmark study:

- **Don't over-split**: `with_max_len(1)` creates excessive synchronization overhead (2x slower in some cases). Let Rayon's default splitting heuristics work.
- **Double-buffering avoids contention**: Read from `current`, write to `next` — no locks, no atomics, no false sharing.
- **Prefer indexed iteration**: `par_iter_mut().enumerate()` over `par_bridge()` — enables better work splitting.
- **CPU pinning helps at scale**: 20% speedup with 8+ threads via `sched_setaffinity()` (but don't bother for < 4 threads).

### Collecting Results

```rust
// If you need to collect sparse results:
let active_nodes: Vec<NodeId> = (0..graph.num_nodes())
    .into_par_iter()
    .filter(|&i| next[i] > threshold)
    .map(|i| NodeId(i as u32))
    .collect();
```

### Pitfalls

- Don't use shared mutable state (Mutex, AtomicF64) — use double-buffering instead
- Don't use `par_bridge()` for indexed data — it can't split efficiently
- Don't ignore the minimum work threshold — for graphs < ~1000 nodes, sequential is faster

---

## 6. Testing Patterns

### Recommended Crates

| Crate | Version | Purpose |
|---|---|---|
| `mockall` | 0.13 | Auto-generate mocks from traits |
| `proptest` | 1.9 | Property-based / fuzz testing |
| `criterion` | 0.5 | Statistical benchmarking |

```toml
[dev-dependencies]
mockall = "0.13"
proptest = "1.9"
criterion = { version = "0.5", features = ["html_reports"] }
```

### Pattern: Mockall with Traits

```rust
#[cfg(test)]
use mockall::automock;

#[cfg_attr(test, automock)]
pub trait ActivationFunction: Send + Sync {
    fn apply(&self, input: f64) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_calls_activation() {
        let mut mock = MockActivationFunction::new();
        mock.expect_apply()
            .times(1)
            .returning(|x| x * 2.0);

        let engine = Engine::new(mock);
        // ...
    }
}
```

### Pattern: Proptest for Graph Invariants

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn activation_values_bounded(
        input in prop::collection::vec(-10.0f64..10.0, 1..100)
    ) {
        let result = apply_sigmoid_batch(&input);
        for &v in &result {
            prop_assert!(v >= 0.0 && v <= 1.0,
                "Sigmoid output {} out of [0,1] range", v);
        }
    }
}
```

### Pattern: Criterion Benchmarks

```rust
// benches/propagation.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_propagation(c: &mut Criterion) {
    let graph = build_test_graph(10_000, 50_000);
    let current = vec![0.5; graph.num_nodes()];
    let mut next = vec![0.0; graph.num_nodes()];

    c.bench_function("propagate_10k", |b| {
        b.iter(|| propagate(black_box(&graph), &current, &mut next));
    });
}

criterion_group!(benches, bench_propagation);
criterion_main!(benches);
```

### Test Organization

```
src/
  lib.rs          # Integration-level tests in #[cfg(test)] mod
  graph.rs        # Unit tests at bottom: #[cfg(test)] mod tests { ... }
  engine.rs
tests/
  integration.rs  # Cross-module integration tests
benches/
  propagation.rs  # Criterion benchmarks
```

### Pitfalls

- Don't mock everything — only mock at trait boundaries for external dependencies
- Don't benchmark with `#[bench]` (unstable) — use Criterion
- Don't ignore proptest regressions files — commit them to git

---

## 7. Error Handling

### Recommendation: `thiserror` 2.x

```toml
[dependencies]
thiserror = "2"
```

### Pattern: Library Error Type

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SproinkError {
    #[error("node index {0} out of bounds (max: {1})")]
    NodeOutOfBounds(u32, u32),

    #[error("graph has no nodes")]
    EmptyGraph,

    #[error("edge weight {0} is not finite")]
    InvalidWeight(f64),

    #[error("graph construction failed: {0}")]
    BuildError(String),
}

pub type Result<T> = std::result::Result<T, SproinkError>;
```

### When to Use Result vs Panic

| Situation | Use |
|---|---|
| Invalid user input (bad node ID, NaN weight) | `Result<T, SproinkError>` |
| Internal invariant violation (corrupted CSR) | `panic!` / `debug_assert!` |
| FFI null pointer | `assert!(!ptr.is_null())` — caller violated contract |
| Out of memory | Let the allocator panic (default) |

### Error Design for FFI

```rust
/// FFI-safe error codes
#[repr(C)]
pub enum SproinkStatus {
    Ok = 0,
    ErrNullPointer = 1,
    ErrOutOfBounds = 2,
    ErrInvalidWeight = 3,
    ErrPanicked = -1,
}

/// Convert Result to FFI status code
impl From<SproinkError> for SproinkStatus {
    fn from(e: SproinkError) -> Self {
        match e {
            SproinkError::NodeOutOfBounds(..) => SproinkStatus::ErrOutOfBounds,
            SproinkError::InvalidWeight(_) => SproinkStatus::ErrInvalidWeight,
            _ => SproinkStatus::ErrPanicked,
        }
    }
}
```

### Pitfalls

- Don't use `anyhow` in a library crate — callers can't match on error variants
- Don't panic across FFI — it's undefined behavior. Catch with `std::panic::catch_unwind`
- Don't create error variants for impossible states — trust internal invariants

---

## 8. FFI Best Practices

### Key Tools

| Tool | Purpose |
|---|---|
| `cbindgen` | Auto-generate C headers from Rust code |
| `Box::into_raw` / `Box::from_raw` | Transfer ownership across FFI boundary |
| `#[repr(C)]` | Ensure C-compatible memory layout |
| `std::panic::catch_unwind` | Prevent panics from crossing FFI |

### cbindgen Setup

```toml
# cbindgen.toml
language = "C"
include_guard = "SPROINK_H"
no_includes = true

[export]
prefix = "sproink_"

[fn]
prefix = "sproink_"
```

```toml
# Cargo.toml
[build-dependencies]
cbindgen = "0.27"
```

### Pattern: Opaque Pointer

```rust
// The Rust struct — complex, not C-compatible
pub struct Engine<A: ActivationFunction> { /* ... */ }

// Opaque handle for C
pub struct SproinkEngine { _private: () }

// Type-erase for FFI (use a concrete activation function)
type ConcreteEngine = Engine<Sigmoid>;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_engine_new(
    config: *const SproinkConfig,
) -> *mut SproinkEngine {
    let config = unsafe { &*config };
    let engine = Box::new(ConcreteEngine::new(config.into()));
    Box::into_raw(engine) as *mut SproinkEngine
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_engine_free(engine: *mut SproinkEngine) {
    if !engine.is_null() {
        drop(unsafe { Box::from_raw(engine as *mut ConcreteEngine) });
    }
}
```

### Pattern: Panic Safety

```rust
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sproink_propagate(
    engine: *mut SproinkEngine,
    steps: u32,
) -> SproinkStatus {
    if engine.is_null() {
        return SproinkStatus::ErrNullPointer;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &mut *(engine as *mut ConcreteEngine) };
        engine.propagate(steps)
    }));

    match result {
        Ok(Ok(())) => SproinkStatus::Ok,
        Ok(Err(e)) => e.into(),
        Err(_) => SproinkStatus::ErrPanicked,
    }
}
```

### CGO Integration Notes

- Rust `cdylib` produces a `.so`/`.dylib` that Go can link via CGO
- Use `staticlib` for static linking (simpler deployment)
- Generate headers with cbindgen, include them in Go with `// #include "sproink.h"`
- Go's CGO handles the C calling convention — Rust's `extern "C"` is compatible

### Pitfalls

- **Never** let panics cross FFI — always use `catch_unwind`
- **Never** return Rust references across FFI — only raw pointers to heap-allocated data
- **Always** provide a matching `_free` function for every `_new`/`_create`
- **Always** use `#[repr(C)]` on structs passed by value across FFI
- Document ownership: who allocates, who frees

---

## 9. Builder Pattern

### Recommendation: `typed-builder` 0.20+

`typed-builder` is preferred over `derive_builder` because it provides **compile-time enforcement** that all required fields are set. Missing a required field is a compile error, not a runtime panic.

```toml
[dependencies]
typed-builder = "0.20"
```

### Pattern

```rust
use typed_builder::TypedBuilder;

#[derive(Debug, Clone, TypedBuilder)]
pub struct Config {
    #[builder(setter(into))]
    num_nodes: usize,

    #[builder(default = 0.5)]
    decay: f64,

    #[builder(default = 10)]
    max_iterations: u32,

    #[builder(default)]
    inhibition: InhibitionConfig,
}

#[derive(Debug, Clone, Default, TypedBuilder)]
pub struct InhibitionConfig {
    #[builder(default = false)]
    enabled: bool,

    #[builder(default = 0.1)]
    strength: f64,
}

// Usage:
let config = Config::builder()
    .num_nodes(1000)
    .decay(0.3)
    .build();
```

### When to Use Builders vs `new()`

| Fields | Approach |
|---|---|
| 1-3, all required | `fn new(a, b, c) -> Self` |
| 4+, some optional | `TypedBuilder` |
| Complex validation | Hand-rolled builder with `build() -> Result<Self, Error>` |

### Pitfalls

- Don't use builders for simple structs — a constructor is clearer
- Don't use `derive_builder` — it returns `Result` from `build()` which panics at runtime on missing fields; `typed-builder` catches this at compile time
- Don't expose builder in FFI — use a flat C struct or `_set_*` functions instead

---

## 10. Unsafe Minimization

### Core Principle

All `unsafe` in sproink should live in exactly one place: the FFI module. The engine, graph, and activation logic should be 100% safe Rust.

### Pattern: FFI Boundary Layer

```
src/
  lib.rs          # Public safe API
  graph.rs        # 100% safe
  engine.rs       # 100% safe
  activation.rs   # 100% safe
  ffi.rs          # ALL unsafe lives here
```

### Pattern: Safe Wrapper Module

```rust
// src/ffi.rs — the ONLY file with `unsafe`

mod ffi {
    use super::*;

    /// SAFETY: All functions in this module validate:
    /// 1. Pointers are non-null
    /// 2. Panics are caught before crossing FFI
    /// 3. Ownership is explicitly transferred (Box::into_raw / from_raw)

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn sproink_engine_new(/* ... */) -> *mut SproinkEngine {
        // Validate, convert to safe types, call safe code
        let config = unsafe { &*config_ptr };
        let safe_config: Config = config.into();  // Safe conversion

        // All the real logic is in safe Rust:
        let engine = Engine::new(safe_config);

        Box::into_raw(Box::new(engine)) as *mut SproinkEngine
    }
}
```

### Key Techniques

1. **Validate at the boundary, trust internally**: Check null pointers and bounds in FFI functions; internal code uses `NodeId` newtypes and slice indexing
2. **Convert C types to Rust types immediately**: Raw pointers become references, C strings become `&str`, C arrays become slices — all in the FFI function
3. **Use `#[deny(unsafe_op_in_unsafe_fn)]`**: Forces explicit `unsafe` blocks even inside `unsafe fn`, making each operation's safety justification clear
4. **`catch_unwind` at every FFI entry point**: One wrapper function handles all panic safety

### Pattern: Safety Audit Helper

```rust
// At crate root:
#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(unsafe_code)]  // In non-FFI modules

// In ffi.rs only:
#![allow(unsafe_code)]
```

Note: Per-module `#![forbid(unsafe_code)]` isn't directly supported, but you can use `#[cfg_attr]` or clippy lints to approximate this. The key discipline is: if you see `unsafe` outside `ffi.rs`, it's a code review failure.

### Pitfalls

- Don't scatter `unsafe` blocks across the codebase — centralize them
- Don't use `unsafe` for performance unless benchmarks prove it's necessary
- Don't skip `catch_unwind` — a single uncaught panic is UB across FFI
- Don't assume C callers will pass valid data — validate everything at the boundary

---

## Summary: Recommended Dependencies

```toml
[dependencies]
rayon = "1.10"
thiserror = "2"
typed-builder = "0.20"
derive_more = { version = "1", features = ["from", "into", "display"] }
# Only if benchmarks justify SIMD:
# wide = "0.7"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
mockall = "0.13"
proptest = "1.9"

[build-dependencies]
cbindgen = "0.27"
```

---

## Sources

- [Rust Traits and Dependency Injection — jmmv.dev](https://jmmv.dev/2022/04/rust-traits-and-dependency-injection.html)
- [Embrace the Newtype Pattern — Effective Rust](https://www.lurklurk.org/effective-rust/newtype.html)
- [Ultimate Guide to Rust Newtypes — howtocodeit.com](https://www.howtocodeit.com/guides/ultimate-guide-rust-newtypes)
- [Petgraph CSR Documentation](https://docs.rs/petgraph/0.5.0/petgraph/csr/struct.Csr.html)
- [Performance Comparison of Graph Representations (2025)](https://arxiv.org/html/2502.13862v1)
- [The State of SIMD in Rust in 2025 — Shnatsel](https://shnatsel.medium.com/the-state-of-simd-in-rust-in-2025-32c263e5f53d)
- [Using Portable SIMD in Stable Rust — pythonspeed.com](https://pythonspeed.com/articles/simd-stable-rust/)
- [Parallelizing Graph Search with Rayon — tavianator.com](https://tavianator.com/2022/parallel_graph_search.html)
- [Making a Parallel Rust Workload 10x Faster — gendignoux.com](https://gendignoux.com/blog/2024/11/18/rust-rayon-optimized.html)
- [Mockall Crate Guide](https://generalistprogrammer.com/tutorials/mockall-rust-crate-guide)
- [Proptest Crate Guide](https://generalistprogrammer.com/tutorials/proptest-rust-crate-guide)
- [thiserror Crate Guide](https://generalistprogrammer.com/tutorials/thiserror-rust-crate-guide)
- [Exposing FFI from Rust — svartalf.info](https://svartalf.info/posts/2019-03-01-exposing-ffi-from-the-rust-library/)
- [The Rustonomicon — FFI](https://doc.rust-lang.org/nomicon/ffi.html)
- [Builder Pattern — Effective Rust](https://effective-rust.com/builders.html)
- [Builder Pattern — Rust Design Patterns](https://rust-unofficial.github.io/patterns/patterns/creational/builder.html)
- [typed-builder crate](https://crates.io/crates/typed-builder)
- [derive_more crate](https://jeltef.github.io/derive_more/derive_more/)
- [Rust's Hidden Dangers: Unsafe, Embedded, and FFI Risks](https://www.trust-in-soft.com/resources/blogs/rusts-hidden-dangers-unsafe-embedded-and-ffi-risks)
