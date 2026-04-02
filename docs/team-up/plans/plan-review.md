# Implementation Plan Review

**Date:** 2026-04-02
**Reviewer:** plan-reviewer
**Status:** DONE_WITH_CONCERNS

---

## Overall Assessment

The plan is thorough, well-structured, and closely aligned with the spec. Task ordering follows a logical dependency chain, TDD is baked into every task with concrete test code, and the dependency graph is correct. The plan is ready for implementation with the concerns noted below.

---

## Spec Alignment

### Fully Covered

| Spec Section | Plan Coverage |
|---|---|
| 2. Core Types (newtypes, enums) | Task 3 (newtypes), Task 4 (EdgeKind enum) |
| 3. Traits (Graph, Inhibitor, Learner, AffinityGenerator) | Tasks 4, 5, 7, 8 |
| 3.2 Engine (generic struct, not trait) | Task 9 |
| 3.4 Sigmoid (free function, not trait) | Task 6 |
| 4. Algorithms (propagation, inhibition, sigmoid, Oja, Jaccard) | Tasks 5-9 |
| 5. CSR (build, bidirectional, DirectionalPassive reversal) | Task 4 |
| 6. Parallelism (Rayon, PARALLEL_THRESHOLD=1024, thread safety) | Task 9 |
| 7. FFI (all 14 function signatures, co-activation pairs, memory ownership) | Tasks 12-13 |
| 8. Errors (SproinkError, thiserror, FFI error handling) | Task 2, Task 12 |
| 9. Module layout (all files) | Task 1 |
| 10. Testing (unit, integration, property, FFI, benchmarks) | Tasks 2-14 |

### Minor Gaps

1. **Self-loop test missing** — Spec section 10.1 lists "self-loops" as a graph test case. Task 4's tests don't include one. Low risk but should be added.

2. **`Seed` struct** — Spec section 3.2 defines `Seed` but the plan doesn't specify its location until Task 9. Task 7 (`extract_co_activation_pairs`) imports `ActivationResult` from engine but `Seed` is also used there. The plan acknowledges this dependency but doesn't resolve it cleanly (see Concern #1 below).

---

## Task Decomposition

Tasks are generally well-sized. Tasks 2-8 are each focused on a single module and completable in a short session.

**Concern: Task 9 is too large.** It combines:
- `PropagationConfig` with `TypedBuilder` (6+ fields, revised mid-task to add inhibition)
- `Seed` and `ActivationResult` definitions
- The full propagation loop (copy-init, 4-denominator normalization, distance tracking)
- Rayon parallelism with `fold+reduce`
- Post-processing pipeline (inhibition → sigmoid → threshold)
- Result collection and sorting

The plan acknowledges this ("Consider splitting into sub-steps during implementation") but doesn't actually split it. An engineer will be writing 13 tests and a complex algorithm in one sitting. **Recommendation:** At minimum, split into 9a (types + sequential propagation) and 9b (Rayon parallelism + post-processing). This also lets the engineer verify correctness before adding parallelism.

---

## Ordering

The dependency graph is correct. Parallelization opportunities (Tasks 5/6/7/8 and Tasks 10/11/12/14) are correctly identified.

**Concern: Task 7 depends on Task 9's types but is ordered before it.** Task 7 (`extract_co_activation_pairs`) needs `ActivationResult`, which is defined in Task 9. The plan says "define a minimal version here or reference it from a shared location" and the Implementation Notes recommend defining in `engine.rs` and importing in `hebbian.rs`. But Task 7 comes before Task 9, so the engineer must either:
- Stub `ActivationResult` in engine.rs during Task 7 (fragile)
- Move `ActivationResult` to types.rs (deviates from spec module layout)
- Reorder Task 7 after Task 9 (breaks parallelism with Tasks 5/6/8)

**Recommendation:** Define `Seed` and `ActivationResult` as part of Task 3 (newtypes) in `types.rs`, or create a small Task 3b for "shared structs." Re-export from `engine.rs` to match the spec's public API. This unblocks Task 7 without reordering.

---

## Buildability

### Strong Points

- Concrete test code provided for every task — an engineer can copy-paste and start.
- Implementation code is provided (not just descriptions), reducing ambiguity.
- Config structs use `typed-builder` with explicit defaults matching the spec.
- Edge cases (sigmoid on zero, empty seeds, tie-breaking) are called out in Implementation Notes.

### Concerns

3. **FFI opaque handle internals not specified.** Task 12 shows:
   ```rust
   pub struct SproinkGraph { _private: () }
   ```
   But doesn't explain how `CsrGraph` is stored inside or retrieved. The standard pattern is either:
   - Wrapper struct: `pub struct SproinkGraph(CsrGraph)` with `Box::into_raw`/`Box::from_raw`
   - Type-erased cast: `Box::into_raw(Box::new(csr)) as *mut SproinkGraph`

   An engineer unfamiliar with Rust FFI patterns will get stuck here. **Recommendation:** Show the wrapper pattern explicitly, e.g.:
   ```rust
   pub struct SproinkGraph { inner: CsrGraph }
   pub struct SproinkResults { inner: Vec<ActivationResult> }
   pub struct SproinkPairs { inner: Vec<CoActivationPair> }
   ```

4. **`PropagationConfig` shape changes mid-task.** Task 9 first shows `PropagationConfig` without inhibition, then adds `inhibition: Option<InhibitionConfig>` later. An engineer doing TDD will write tests against the first definition. **Recommendation:** Show the final shape upfront.

5. **`empty_seeds_returns_empty` test is misleadingly named.** The test comment explains that sigmoid(0) ≈ 0.047 > min_activation, so the result won't be empty. The test name contradicts its actual assertion. Rename to `empty_seeds_no_panic` or `empty_seeds_sigmoid_baseline`.

---

## TDD Compliance

Every task specifies tests to write first with concrete code. Acceptance criteria are clear and testable. This is well done.

One note: Task 10 (Integration Tests) and Task 11 (Property Tests) show test signatures with `{ ... }` bodies. The engineer will need to fill these in. This is appropriate — integration and property tests require judgment about exact assertions. The spec provides enough detail to derive expected values.

---

## Risk Areas

1. **Rayon `fold+reduce` complexity in Task 9.** The parallel propagation with thread-local buffers merged via max-semantics is the most complex part of the codebase. If the engineer gets this wrong, debugging parallel correctness issues is painful. The plan's recommendation to start with `fold+reduce` is sound, but having sequential-first (verified correct) then parallel (verified equivalent) would be safer. The `parallel_matches_sequential` integration test in Task 10 helps, but it comes after Task 9.

2. **`mockall` and `Graph` trait lifetime.** The plan correctly notes that `MockGraph::neighbors()` returning `&[EdgeData]` is tricky with mockall. The recommendation to prefer real `CsrGraph` in tests is good. The engineer should follow this advice — mocking Graph is more trouble than it's worth for most tests.

3. **cbindgen may not pick up all FFI functions.** The `build.rs` in Task 15 uses cbindgen to auto-generate headers. If FFI functions use complex types or macros, cbindgen may need additional configuration. Low risk given the functions are simple flat-array signatures.

---

## Summary of Concerns

| # | Severity | Issue | Recommendation |
|---|----------|-------|----------------|
| 1 | Medium | Task 7 needs `ActivationResult` from Task 9 (ordering conflict) | Define shared structs in Task 3 or types.rs |
| 2 | Medium | Task 9 too large for one session | Split into 9a (sequential) and 9b (parallel + post-processing) |
| 3 | Medium | FFI opaque handle internals not specified | Show wrapper struct pattern explicitly |
| 4 | Low | `PropagationConfig` shape changes mid-task | Show final shape upfront in Task 9 |
| 5 | Low | `empty_seeds_returns_empty` test name misleading | Rename test |
| 6 | Low | Self-loop graph test missing | Add to Task 4 |

None of these are blocking. An experienced Rust engineer can resolve all of them during implementation. But fixing concerns 1-3 before starting would avoid unnecessary friction.
