# Sproink Spec Review

**Reviewer:** spec-reviewer
**Date:** 2026-04-02
**Status:** DONE_WITH_CONCERNS

---

## Summary

The spec is well-structured, thorough, and covers every major algorithm from the Go reference. The math is correct where specified. However, there are two issues that would cause an engineer to build the wrong thing if not clarified, and a handful of smaller gaps.

---

## Critical Issues

### 1. Activation vector initialization is ambiguous — will break parity

**Section 4.1, Steps 1..T** says: *"Each step reads from the previous step's activation vector and writes to a new vector."*

The Go code (`engine.go:272,328`) initializes `newActivation` as a **copy** of the current activation:

```go
newActivation := copyActivation(activation)
```

This means activation **persists** across steps — a seed with activation 0.8 keeps that 0.8 unless something overwrites it. The max-semantics formula `new[j] = max(new[j], base_energy)` then means positive energy can never *decrease* a node's activation within a step.

The spec's phrase "writes to a new vector" reads as a fresh/zeroed vector. If implemented that way:
- Seeds lose their activation after step 1 (nothing spreads energy *back* to them)
- Nodes only have activation from what was spread to them in the current step
- Fundamentally different behavior from Go

**Fix:** Add an explicit initialization step: *"Initialize `new` as a copy of the current activation vector. This preserves activation from prior steps; max-semantics ensures positive energy never decreases a node's value."*

### 2. DirectionalPassive treated as Positive — not how Go works

**Section 2.2** defines `DirectionalPassive` as the reverse side of a `DirectionalSuppressive` edge and says it's *"Treated as Positive during propagation."*

**Section 4.1** confirms: DirectionalPassive uses `positive_count` denominator and max-semantics, same as Positive.

In the Go code (`engine.go:143-160,182-193`), inbound directional edges are **completely skipped** — they don't spread energy and aren't counted in any denominator:

```go
case store.EdgeKindOverrides, store.EdgeKindDeprecatedTo, store.EdgeKindMergedInto:
    if edge.Source == nodeID {
        // suppress outbound only
    }
    // else: nothing happens — no energy spread, not counted
```

The counting loop similarly only counts outbound directional edges in `directionalSuppressiveCount` and does NOT add inbound ones to `positiveCount`.

This means the spec **adds new behavior** not present in Go: energy flowing through the reverse direction of directional edges. This also changes the `positive_count` denominator for all positive edges on nodes that have inbound directional edges.

**Fix:** Either:
- (a) If parity is the goal: Change DirectionalPassive to be skipped during propagation (not counted, no energy spread). It would still exist as a CSR bookkeeping artifact to avoid the source/target check at propagation time, but the propagation table should say "DirectionalPassive: skip (no effect)."
- (b) If this is an intentional improvement: Document it as a deliberate divergence from Go, explain the rationale, and remove the "exact algorithmic parity" claim from the goals.

---

## Minor Issues

### 3. Co-activation pair extraction has no trait or function signature

Section 4.4 describes the co-activation extraction algorithm in detail, but there's no corresponding trait, function signature, or placement in the type system. The `Learner` trait only has `update_weight`. The module layout lists `hebbian.rs` with `CoActivationPair` but no extraction function.

An engineer would have to guess whether this is a standalone function, a method on the engine, or part of a trait.

**Fix:** Add a function signature, e.g.:
```rust
pub fn extract_co_activation_pairs(
    results: &[ActivationResult],
    seed_nodes: &HashSet<NodeId>,
    config: &HebbianConfig,
) -> Vec<CoActivationPair>;
```

### 4. FFI has no function for co-activation pair extraction

Section 7.2 provides `sproink_oja_update` but no FFI function for extracting co-activation pairs. Floop needs these pairs to decide whether to create/strengthen edges. Without an FFI function, floop would have to reimplement the extraction logic in Go, defeating the purpose.

**Fix:** Add FFI functions:
```c
SproinkPairs* sproink_extract_pairs(
    const SproinkResults* results,
    uint32_t num_seeds,
    const uint32_t* seed_nodes,
    double activation_threshold
);
uint32_t sproink_pairs_len(const SproinkPairs* pairs);
void sproink_pairs_nodes(const SproinkPairs* pairs, uint32_t* out_a, uint32_t* out_b);
void sproink_pairs_activations(const SproinkPairs* pairs, double* out_a, double* out_b);
void sproink_pairs_free(SproinkPairs* pairs);
```

### 5. Result sort tie-breaking differs from Go

Section 4.1 Result Collection says: *"Ties broken by NodeId ascending for determinism."*

The Go code (`engine.go:352-354`) sorts by activation descending only — no tie-breaker:
```go
sort.Slice(results, func(i, j int) bool {
    return results[i].Activation > results[j].Activation
})
```

This is an improvement, not a bug, but it's worth noting as a known divergence from strict parity. No fix needed — just acknowledge it.

---

## Verified Correct

These items were specifically checked against the Go reference and are accurate:

- **Four-denominator normalization model** (positive, conflict, directional suppressive, virtual) — matches `engine.go:139-160`
- **Max-semantics for positive energy** (not sum) — matches `engine.go:208-209`
- **DirectionalSuppressive only suppresses outbound** — matches `engine.go:185-186`
- **Inhibition formula**: `beta * (mean_winner - loser)` — matches `inhibition.go:98`
- **Oja's forgetting term**: `A_j^2 * W` — matches `hebbian.go:67`
- **Sigmoid**: `1 / (1 + exp(-10 * (x - 0.3)))` — matches `engine.go:372` + `constants.go:109,113`
- **Co-activation seed-seed exclusion** — matches `hebbian.go:99`
- **Jaccard virtual edge weight**: `J * max_weight` — matches `affinity.go:69`
- **Conflict suppression is symmetric** — matches `engine.go:175-180`
- **NaN/Inf safety in Oja update** — matches `hebbian.go:126-128`
- **Pipeline order**: propagate → inhibit → sigmoid → filter — matches `engine.go:230-243`
- **Inhibition tie-breaking by node ID ascending** — matches `inhibition.go:68-73`
- **Default config values** all match

---

## Scope & Consistency

- Scope is coherent — one crate, one deliverable, clear boundaries with floop
- Types are used consistently throughout
- FFI signatures match the Rust types (opaque handles, flat arrays, correct ownership)
- Non-goals are appropriate (temporal decay, persistence, seed selection all correctly delegated to floop)
- The researcher's findings on Rust patterns align well with the spec's architectural choices (trait-based DI, hand-rolled CSR, Rayon parallelism, cbindgen FFI)

---

## Verdict

**DONE_WITH_CONCERNS** — The spec is solid but issues #1 and #2 must be resolved before planning. An engineer implementing from this spec would build incorrect propagation behavior. Issues #3 and #4 are minor but should ideally be addressed to avoid ambiguity during implementation.
