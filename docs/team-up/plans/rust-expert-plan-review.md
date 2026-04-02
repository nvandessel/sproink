# Rust Expert Review: Implementation Plan

**Reviewer:** Rust Domain Expert
**Date:** 2026-04-02
**Verdict:** APPROVED with minor feedback (no blockers)

---

## 1. Type Signatures and Trait Definitions

**Good:**
- All newtypes have `#[repr(transparent)]` — my review recommendation was adopted
- `Deref` removed in favor of `.get()` — correct
- `Squasher` replaced with free function `squash_sigmoid` — correct
- `Engine<G: Graph>` generic struct — correct, zero-cost dispatch
- `thiserror::Error` derive on `SproinkError` — correct
- `typed-builder` for config structs with 4+ fields — correct

**Minor issues:**

1. **`Inhibitor` trait still takes `&mut [f64]`** (line ~640). This is fine as a pragmatic choice (internal hot path), but document the intent — it's not `&mut [Activation]` by design, not by accident. Add a one-line comment on the trait:

```rust
/// Post-propagation lateral inhibition.
/// Takes raw f64 slices for performance in the hot path.
pub trait Inhibitor: Send + Sync {
    fn inhibit(&self, activations: &mut [f64], config: &InhibitionConfig);
}
```

2. **`ActivationResult` placement decision is good.** Defining in `engine.rs` and importing into `hebbian.rs` is the right call. One-way dependency, matches spec layout.

3. **`EdgeWeight::new_unchecked` visibility.** Plan has `pub(crate)` — but the code shows just `pub`. Should be `pub(crate)` to prevent external callers from bypassing validation. Same for `Activation::new_unchecked`. Confirm in implementation.

---

## 2. Rayon Parallelism (fold+reduce)

**The approach is correct.** The plan correctly identifies that max-semantics prevents purely parallel writes and chooses fold+reduce with thread-local buffers merged via `max`.

**One concern with the plan's description in Task 9:**

The plan says:
> If `num_nodes >= PARALLEL_THRESHOLD` (1024), use `par_iter` with fold+reduce

But the actual fold+reduce code isn't shown in Task 9. The engineer needs to implement:

```rust
// Parallel path
let merged: Vec<f64> = (0..num_nodes)
    .into_par_iter()
    .filter(|&i| current[i] >= config.min_activation)
    .fold(
        || vec![0.0f64; num_nodes],
        |mut local, i| {
            // per-node propagation into thread-local buffer
            local
        },
    )
    .reduce(
        || vec![0.0f64; num_nodes],
        |mut a, b| {
            for (x, y) in a.iter_mut().zip(b.iter()) { *x = x.max(*y); }
            a
        },
    );
```

**Key pitfall the engineer should know:** The fold closure must handle *all* edge kinds including suppression. Suppression uses subtraction, not max. The merge step needs to handle this correctly:
- Positive/FeatureAffinity contributions: merge with `max`
- Conflict/DirectionalSuppressive contributions: these subtract from the target

This means the fold+reduce pattern needs two separate accumulators — one for positive energy (max-merge) and one for suppression deltas (sum-merge) — or the parallel and sequential code paths need to produce identical results. **I'd recommend starting with sequential-only for correctness, then adding the parallel path as an optimization.** The parallel threshold of 1024 nodes means most test graphs will use sequential anyway.

---

## 3. Testing Strategy

**Good:**
- Mockall on `Graph` trait — correct approach since `Engine<G>` is generic
- Proptest for invariant properties — well chosen
- Small real `CsrGraph` instances preferred over mocks in engine tests — pragmatic and correct

**Feedback:**

1. **`test_directional_passive_completely_skipped`** (Task 9): The test comment says "DirectionalPassive should not spread energy or count in any denominator." But per the spec, DirectionalPassive is actually **treated as Positive during propagation** (spec line 69-70: "Reverse side of a DirectionalSuppressive edge. Treated as Positive during propagation."). The test as written expects passive edges to be *completely skipped*, which contradicts the spec.

   **This is a spec-vs-plan inconsistency.** Re-read spec section 4.1: the propagation table shows Positive and DirectionalPassive use the same treatment (same denominator, max-semantics). The plan's Task 9 algorithm description says "DirectionalPassive: skip entirely" — this is wrong per spec.

   **Correct behavior:** DirectionalPassive should be counted in `positive_count` and propagate positive energy, just like Positive edges. The directionality is already handled by the CSR construction — when you store `A->B DirectionalSuppressive`, the reverse `B->A` is stored as `DirectionalPassive`. Node B sees a positive edge back to A. Node A sees a suppressive edge to B. The passive/positive distinction is just how the reverse was generated, not a propagation behavior difference.

   **Action:** Fix the test and the algorithm description in Task 9.

2. **`empty_seeds_returns_empty` test** (Task 9): The comment acknowledges sigmoid(0) ≈ 0.047 > min_activation=0.01, so all nodes appear. This isn't testing "returns empty" — rename to `empty_seeds_no_crash` or `empty_seeds_only_sigmoid_baseline`. The Implementation Notes section at the bottom correctly explains this behavior, but the test name is misleading.

3. **Property test for determinism** (Task 11): Good, but needs to account for Rayon non-determinism. With fold+reduce, the order of float operations can vary between runs when fold boundaries shift. For f64, `max(a, max(b, c))` == `max(max(a, b), c)` (max is associative), so this should be fine. But suppression accumulation order could differ. Worth calling out to the engineer: if determinism tests fail only with Rayon, check float associativity of the merge operation.

---

## 4. Ordering / Dependencies

**The dependency graph is correct.** Each task has what it needs from prior tasks.

**One ordering concern:**

Task 7 (Hebbian) imports `ActivationResult` from `engine.rs` (Task 9). But Task 7 is listed as depending only on Tasks 1-3, while Task 9 depends on Tasks 1-6. This means Task 7 can't compile until at least the `ActivationResult` struct exists in engine.rs.

**Solutions (pick one):**
- Move `ActivationResult` and `Seed` to `types.rs` (Task 3) — simplest, no circular deps
- Add Task 9 as a soft dependency of Task 7 (at least the type definitions)
- Stub `ActivationResult` in engine.rs during Task 4 (graph setup)

The plan already acknowledges this ("Uses `ActivationResult` from engine — define a minimal version here or reference it from a shared location"). I'd go with moving them to `types.rs` since they're pure data types with no logic dependency.

---

## 5. FFI Approach

**Good:**
- `ffi_entry!` macro — correct pattern, reduces boilerplate
- `catch_unwind` at every entry — mandatory, correctly specified
- Opaque handle types with `_private: ()` — standard pattern
- cbindgen in build.rs — correct setup
- `edge_kind_from_u8` rejecting value 3 (DirectionalPassive) — correct per spec

**Feedback:**

1. **`sproink_activate` signature.** 13 parameters is a lot for a C function. It works, but the CGO call site will be unwieldy. Consider whether a config struct passed by pointer would be cleaner for the Go side:

```c
typedef struct {
    uint32_t max_steps;
    double decay_factor;
    double spread_factor;
    double min_activation;
    double sigmoid_gain;
    double sigmoid_center;
    bool inhibition_enabled;
    double inhibition_strength;
    uint32_t inhibition_breadth;
} SproinkConfig;

SproinkResults* sproink_activate(
    const SproinkGraph* graph,
    uint32_t num_seeds,
    const uint32_t* seed_nodes,
    const double* seed_activations,
    const SproinkConfig* config
);
```

This is optional — flat params work and avoid an extra struct definition. But flag it for the Go integration author.

2. **`sproink_extract_pairs` is new** (not in the original spec's FFI signatures). Good addition — the spec only had `sproink_oja_update` for learning. The pairs extraction FFI is needed for floop to get co-activation data back. Make sure the spec is updated to include this.

3. **Miri testing note.** The plan mentions running FFI tests under Miri. Note: Miri doesn't support FFI (`extern "C"` calls) — it will skip or error on those tests. The FFI round-trip tests validate correctness; for UB detection, rely on sanitizers (`cargo +nightly test -Z sanitizer=address`).

---

## 6. Rust-Specific Pitfalls to Warn About

These should be flagged to the implementing engineer:

1. **Suppression in parallel propagation.** As noted in section 2, suppression (subtraction) has different merge semantics than positive energy (max). The parallel path must handle this correctly or produce incorrect results. Start sequential, verify against reference, then parallelize.

2. **`new_unchecked` discipline.** Every call site of `EdgeWeight::new_unchecked` and `Activation::new_unchecked` must be a location where the value is mathematically guaranteed to be in range. After sigmoid: yes (always [0,1]). After subtraction in inhibition: must clamp to 0 first. After Oja update: must clamp first, then construct. The `debug_assert!` in `new_unchecked` will catch violations in test builds.

3. **Float comparison in sorting.** `ActivationResult` sorting by activation descending requires comparing `f64`. Use `f64::total_cmp` (stable since Rust 1.62) for deterministic NaN handling:

```rust
results.sort_by(|a, b| {
    b.activation.get().total_cmp(&a.activation.get())
        .then(a.node.cmp(&b.node))
});
```

4. **`CsrGraph::build` edge deduplication.** If the caller passes duplicate edges (same source, target, kind), the CSR will store both. The spec doesn't mention deduplication. This is probably fine (floop controls the input), but worth a doc comment.

5. **`InhibitionConfig` default for `PropagationConfig`.** With `typed-builder`, `#[builder(default)]` on `Option<InhibitionConfig>` defaults to `None`. The plan's FFI function always constructs an `InhibitionConfig` from the bool/strength/breadth params. Make sure `inhibition_enabled: false` maps to `None`, not `Some(InhibitionConfig { enabled: false, .. })` — though both work, `None` is cleaner.

6. **`Box::into_raw` / `Box::from_raw` type erasure.** The FFI casts `Box<CsrGraph>` to `*mut SproinkGraph` and back. This is fine as long as:
   - The cast is always `CsrGraph` (not some other `impl Graph`) — since `Engine` is generic but FFI is concrete, this is guaranteed
   - The same type is used in `_build` and `_free` — enforce this by having a single concrete type alias

---

## Summary

The plan is well-structured, correctly ordered, and incorporates all recommendations from the spec review. The TDD approach with tests-first is solid. Items needing attention before implementation:

| Issue | Severity | Action |
|-------|----------|--------|
| ~~DirectionalPassive propagation behavior~~ | ~~Medium~~ | **False positive** — spec was revised post-first-review. Approved spec correctly skips DirectionalPassive entirely (matches Go reference `engine.go:182-193`). Plan Task 9 is correct. |
| `ActivationResult` placement for Task 7 dependency | **Low** | Move to `types.rs` or stub early |
| Miri doesn't support FFI | **Low** | Use address sanitizer instead |

Everything else is ready. The engineer can start implementing.
