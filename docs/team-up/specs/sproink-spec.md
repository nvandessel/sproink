# Sproink Specification

**Date:** 2026-04-02
**Status:** Approved
**Crate:** `sproink` v0.1.0

## 1. Problem Statement & Motivation

Floop's spreading activation engine currently queries a SQLite graph store at each propagation step. This is correct but slow: each of the 3 propagation steps issues O(active_nodes) SQL queries, and the Jaccard affinity computation scans the full tag table per cycle.

Sproink replaces the per-step SQL queries with pure in-memory computation over a CSR (compressed sparse row) graph. Floop builds the CSR once from its SQLite store, passes it to Sproink via C FFI, and gets activation scores back as flat arrays. The graph structure, propagation math, and learning rules are identical to the Go reference (`floop/internal/spreading/`).

### Goals

- Exact algorithmic parity with the Go reference implementation
- Sub-millisecond propagation for graphs up to ~50k nodes
- Cache-friendly memory layout (CSR)
- Data-parallel propagation via Rayon
- C FFI consumed by floop via CGO
- Trait-based architecture enabling isolated unit testing

### Non-Goals

- Graph persistence (floop owns SQLite; Sproink is stateless)
- Seed selection / context matching (floop's responsibility)
- Temporal decay computation (floop computes effective weights before passing to Sproink)
- Online graph mutation (graph is built once, used immutably)

---

## 2. Core Types

### 2.1 Newtypes

```rust
/// Dense graph node identifier. Nodes are numbered [0, num_nodes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// Edge weight in [0.0, 1.0]. Enforced at construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeWeight(f64);

/// Activation energy in [0.0, 1.0].
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Activation(f64);

/// Tag identifier for Jaccard affinity computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TagId(pub u32);
```

All newtypes use `#[repr(transparent)]` for FFI compatibility.

`EdgeWeight` and `Activation` expose:
- `new(f64) -> Result<Self, SproinkError>` — validates range, rejects NaN/Inf
- `new_unchecked(f64) -> Self` — skips validation; for internal use where values are known-good
- `get(self) -> f64` — returns the inner value

No `Deref<Target = f64>`. Explicit `.get()` calls at arithmetic sites prevent accidental implicit coercion.

### 2.2 Enums

```rust
/// Classifies how an edge transmits energy.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// Normal activation spread. Energy propagated via max-semantics.
    Positive = 0,
    /// Symmetric suppression. Both endpoints suppress each other.
    Conflicts = 1,
    /// One-way suppression. Only the source->target direction suppresses.
    /// Models: overrides, deprecated_to, merged_into.
    DirectionalSuppressive = 2,
    /// Reverse side of a DirectionalSuppressive edge. Internal CSR bookkeeping;
    /// skipped during propagation (no energy spread, not counted in any
    /// denominator). Not accepted at the FFI boundary.
    DirectionalPassive = 3,
    /// Virtual edge from tag Jaccard similarity. Uses a separate
    /// normalization denominator to avoid diluting real edges.
    FeatureAffinity = 4,
}
```

---

## 3. Trait Hierarchy

All component interactions go through traits. Concrete types implement traits; consumers depend only on the trait. This enables mock injection for testing (compatible with `mockall`).

### 3.1 Graph

```rust
/// Read-only graph query interface.
pub trait Graph: Send + Sync {
    /// Number of nodes in the graph.
    fn num_nodes(&self) -> u32;

    /// Adjacency list for `node`. Returns a slice of (target, weight, kind)
    /// tuples. The slice is valid for the graph's lifetime.
    fn neighbors(&self, node: NodeId) -> &[EdgeData];
}
```

`EdgeData` is a plain struct:
```rust
pub struct EdgeData {
    pub target: NodeId,
    pub weight: EdgeWeight,
    pub kind: EdgeKind,
}
```

### 3.2 Engine (Generic, not Trait)

The engine is a concrete generic struct parameterized over `Graph`, not a trait. This gives zero-cost static dispatch and monomorphization while still allowing mock graph injection for testing.

```rust
pub struct Engine<G: Graph> {
    graph: G,
}

impl<G: Graph> Engine<G> {
    pub fn new(graph: G) -> Self;

    pub fn activate(
        &self,
        seeds: &[Seed],
        config: &PropagationConfig,
    ) -> Vec<ActivationResult>;
}
```

`PropagationConfig` uses `typed-builder` for ergonomic construction with defaults (6 fields).

**Internal activation representation:** Inside `Engine`, activation is stored as `Vec<f64>` indexed by `NodeId.0`. The `Activation` newtype is used only at API boundaries (seeds in, results out). This avoids per-element validation overhead in the hot loop.

### 3.3 Inhibitor

```rust
/// Post-propagation lateral inhibition.
pub trait Inhibitor: Send + Sync {
    /// Suppress losers in-place. `activations` is indexed by NodeId.
    fn inhibit(&self, activations: &mut [f64], config: &InhibitionConfig);
}
```

### 3.4 Sigmoid (Free Function, not Trait)

Sigmoid squashing is a plain function, not a trait — there's no realistic second implementation to justify the abstraction.

```rust
/// Apply sigmoid squashing element-wise in-place.
pub fn squash_sigmoid(activations: &mut [f64], gain: f64, center: f64);
```

### 3.5 Learner

```rust
/// Hebbian weight update computation.
pub trait Learner: Send + Sync {
    /// Compute updated weight using Oja's rule.
    fn update_weight(
        &self,
        current_weight: EdgeWeight,
        activation_a: Activation,
        activation_b: Activation,
        config: &HebbianConfig,
    ) -> EdgeWeight;
}
```

### 3.6 AffinityGenerator

```rust
/// Generates virtual edges from tag similarity.
pub trait AffinityGenerator: Send + Sync {
    /// Compute affinity edges for all node pairs above the Jaccard threshold.
    fn generate(
        &self,
        node_tags: &[Vec<TagId>],
        config: &AffinityConfig,
    ) -> Vec<EdgeInput>;
}
```

---

## 4. Algorithm Specifications

### 4.1 Propagation Pipeline

The engine runs a fixed pipeline: **seed -> propagate -> inhibit -> sigmoid -> filter**.

#### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps` | `u32` | 3 | Propagation iterations (T) |
| `decay_factor` | `f64` | 0.7 | Energy retention per hop (delta) |
| `spread_factor` | `f64` | 0.85 | Edge transmission efficiency (S) |
| `min_activation` | `f64` | 0.01 | Pruning threshold (epsilon) |
| `sigmoid_gain` | `f64` | 10.0 | Sigmoid steepness |
| `sigmoid_center` | `f64` | 0.3 | Sigmoid inflection point |

#### Step 0: Seed Initialization

```
for each seed in seeds:
    activation[seed.node] = seed.activation
    distance[seed.node] = 0
```

#### Steps 1..T: Synchronous Propagation

Each step reads from the previous step's activation vector and writes to a new vector. No within-step feedback.

Initialize `new` as a **copy** of the current activation vector. This preserves activation from prior steps; max-semantics ensures positive energy never decreases a node's value.

For each active node `i` where `activation[i] >= min_activation`:

1. **Count edges by category** (three independent denominators; DirectionalPassive is excluded from all counts):
   - `positive_count`: Positive edges only
   - `conflict_count`: Conflicts edges
   - `dir_suppress_count`: DirectionalSuppressive edges (outbound only)
   - `virtual_count`: FeatureAffinity edges

2. **For each neighbor `j` via edge `(i, j, w, kind)`**:

   ```
   base_energy = activation[i] * spread_factor * w * decay_factor
   ```

   | Edge Kind | Denominator | Effect on target |
   |-----------|-------------|-----------------|
   | Positive | positive_count | `new[j] = max(new[j], base_energy / positive_count)` |
   | FeatureAffinity | virtual_count | `new[j] = max(new[j], base_energy / virtual_count)` |
   | Conflicts | conflict_count | `new[j] = max(0, new[j] - base_energy / conflict_count)` |
   | DirectionalSuppressive | dir_suppress_count | `new[j] = max(0, new[j] - base_energy / dir_suppress_count)` |
   | DirectionalPassive | — | Skip (no effect) |

3. **Distance tracking**: If energy was spread (positive delta > 0 or suppression applied):
   ```
   candidate = distance[i] + 1
   if candidate < distance[j]:
       distance[j] = candidate
   ```

**Critical invariants:**
- Positive energy uses **max-semantics** (not sum) to prevent runaway accumulation
- Suppression is clamped to floor of 0
- Each edge category uses its own denominator to prevent cross-category dilution
- DirectionalSuppressive only suppresses in the source->target direction; the reverse direction is stored as DirectionalPassive and **skipped entirely** during propagation (no energy spread, not counted in any denominator). This matches the Go reference where inbound directional edges are completely ignored.
- Updates are synchronous: step N reads from step N-1's values only

#### Post-Processing

Applied once after all propagation steps complete, in order:

1. **Lateral inhibition** (if enabled)
2. **Sigmoid squashing**: `activation[i] = 1 / (1 + exp(-gain * (activation[i] - center)))`
3. **Threshold filter**: zero out nodes with `activation[i] < min_activation`

#### Result Collection

Collect all nodes with `activation > 0` into `Vec<ActivationResult>`, sorted by activation descending. Ties broken by `NodeId` ascending for determinism. (This is an intentional improvement over the Go reference, which does not specify tie-breaking for results.)

```rust
pub struct ActivationResult {
    pub node: NodeId,
    pub activation: Activation,
    pub distance: u32,
}
```

### 4.2 Lateral Inhibition

Top-M lateral inhibition: the M strongest nodes suppress all others.

#### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strength` | `f64` | 0.15 | Suppression intensity (beta) |
| `breadth` | `usize` | 7 | Number of winner nodes (M) |
| `enabled` | `bool` | true | Toggle |

#### Algorithm

1. Collect all nodes with `activation > 0`
2. If count <= `breadth`: return unchanged (all are winners)
3. Sort by activation descending; tie-break by `NodeId` ascending
4. Winners = top M nodes
5. `mean_winner = sum(winner activations) / M`
6. For each loser (rank > M):
   ```
   suppression = strength * (mean_winner - loser.activation)
   loser.activation = max(0, loser.activation - suppression)
   ```

### 4.3 Sigmoid

```
sigmoid(x) = 1 / (1 + exp(-gain * (x - center)))
```

With defaults `gain=10.0`, `center=0.3`:
- Values near 0.0 are squashed toward `sigmoid(0) = 0.047`
- Values at 0.3 map to 0.5
- Values near 1.0 are squashed toward `sigmoid(1) = 0.999`

### 4.4 Hebbian Learning (Oja's Rule)

Weight updates for co-activated node pairs.

#### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | `f64` | 0.05 | Update magnitude (eta) |
| `min_weight` | `f64` | 0.01 | Weight floor |
| `max_weight` | `f64` | 0.95 | Weight ceiling |
| `activation_threshold` | `f64` | 0.15 | Min activation for co-activation pairing |

`HebbianConfig` uses `typed-builder` for ergonomic construction with defaults (4 fields).

#### Oja Update Formula

```
dW = eta * (A_i * A_j - A_j^2 * W)
new_weight = clamp(W + dW, min_weight, max_weight)
```

Where:
- `A_i * A_j` = Hebbian co-activation term (strengthens edges between co-active nodes)
- `A_j^2 * W` = forgetting/stabilization term (prevents unbounded growth)
- Result is NaN/Inf-safe: falls back to `min_weight`

#### Co-Activation Pair Extraction

After propagation completes:

1. Filter nodes with `activation >= activation_threshold`
2. Generate all unique pairs `(a, b)` where `a < b` (canonical ordering)
3. Exclude pairs where **both** members are seed nodes (seed-seed pairs reflect context coincidence, not behavioral affinity)
4. Return `Vec<CoActivationPair>`:
   ```rust
   pub struct CoActivationPair {
       pub node_a: NodeId,
       pub node_b: NodeId,
       pub activation_a: Activation,
       pub activation_b: Activation,
   }
   ```

#### Extraction API

```rust
pub fn extract_co_activation_pairs(
    results: &[ActivationResult],
    seed_nodes: &HashSet<NodeId>,
    config: &HebbianConfig,
) -> Vec<CoActivationPair>;
```

Floop uses these pairs to decide whether to create/strengthen edges in its graph store. Sproink does not persist anything.

### 4.5 Feature Affinity (Jaccard Virtual Edges)

Generates virtual edges between nodes with similar tag sets.

#### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_weight` | `f64` | 0.4 | Maximum virtual edge weight |
| `min_jaccard` | `f64` | 0.3 | Minimum Jaccard similarity threshold |

#### Jaccard Similarity

```
J(A, B) = |A intersection B| / |A union B|
```

Input tag slices must be sorted. Implementation uses two-pointer merge (O(n+m)). Both-empty and either-empty cases return 0.0.

#### Virtual Edge Generation

For each node pair `(i, j)` where `i < j`:
1. Skip if either node has an empty tag set
2. Compute `J = jaccard(tags[i], tags[j])`
3. If `J >= min_jaccard`: emit `EdgeInput { source: i, target: j, weight: J * max_weight, kind: FeatureAffinity }`

Virtual edges are included in the CSR graph at build time. They use a **separate normalization denominator** (`virtual_count`) during propagation to prevent diluting real edges.

---

## 5. CSR Graph Construction

### 5.1 Builder Input

```rust
pub struct EdgeInput {
    pub source: NodeId,
    pub target: NodeId,
    pub weight: EdgeWeight,
    pub kind: EdgeKind,
}
```

### 5.2 Construction Algorithm

The CSR graph stores edges **bidirectionally**: each `EdgeInput(u, v, w, kind)` produces two stored edges.

1. For each input edge `(u, v, w, kind)`:
   - Store `(u -> v, w, kind)` at node `u`
   - Store `(v -> u, w, reverse_kind)` at node `v`
   - Where `reverse_kind`:
     - `DirectionalSuppressive` -> `DirectionalPassive` (reverse direction is passive)
     - All other kinds -> same kind (symmetric)

2. Build prefix-sum `row_offsets` array of length `num_nodes + 1`
3. Pack edges contiguously, grouped by source node

### 5.3 Internal Layout

```rust
pub struct CsrGraph {
    num_nodes: u32,
    row_offsets: Vec<u32>,  // length: num_nodes + 1
    edges: Vec<EdgeData>,   // length: 2 * num_input_edges
}
```

### 5.4 Query API

```rust
impl Graph for CsrGraph {
    fn num_nodes(&self) -> u32;
    fn neighbors(&self, node: NodeId) -> &[EdgeData];  // O(1) via row_offsets
}
```

`neighbors()` returns `&edges[row_offsets[node]..row_offsets[node+1]]`.

---

## 6. Parallelism

### 6.1 Rayon for Per-Node Propagation

Each propagation step computes per-node contributions in parallel via `rayon::par_iter`. Each parallel task produces a list of `(target, delta, distance)` updates. Updates are merged sequentially to maintain correctness (max-semantics for positive, subtraction for suppressive).

**Parallel threshold:** `const PARALLEL_THRESHOLD: u32 = 1024`. Graphs with fewer than 1024 nodes use sequential iteration to avoid Rayon overhead.

### 6.2 Thread Safety

- `CsrGraph` is immutable after construction: `Send + Sync` by nature
- Activation vectors are per-step owned: no shared mutable state during parallel computation
- All traits require `Send + Sync` bounds
- No `Mutex`, `RwLock`, or atomics needed in the hot path

### 6.3 SIMD

SIMD optimization is **optional** and can be applied to:
- Sigmoid computation over activation vectors
- Jaccard similarity inner loop

The compiler's auto-vectorization should handle sigmoid. Explicit SIMD is deferred unless benchmarks show a bottleneck.

---

## 7. C FFI Contract

### 7.1 Design Principles

- **Opaque handles**: Rust types are hidden behind `*mut SproinkGraph` / `*mut SproinkResults` pointers
- **Flat arrays**: All data crosses the boundary as `*const T` / `*mut T` + length. No structs cross FFI.
- **Caller-allocated output buffers**: Results are copied into caller-provided arrays
- **Explicit lifetime**: `_build` returns an owned handle; `_free` deallocates it

### 7.2 Function Signatures

```c
// --- Graph ---

// Build a CSR graph from flat arrays. Returns owned handle.
// kind encoding: 0=Positive, 1=Conflicts, 2=DirectionalSuppressive, 4=FeatureAffinity
SproinkGraph* sproink_graph_build(
    uint32_t num_nodes,
    uint32_t num_edges,
    const uint32_t* sources,
    const uint32_t* targets,
    const double*   weights,
    const uint8_t*  kinds
);

// Free a graph handle.
void sproink_graph_free(SproinkGraph* graph);

// --- Activation ---

// Run spreading activation. Returns owned results handle.
SproinkResults* sproink_activate(
    const SproinkGraph* graph,
    uint32_t num_seeds,
    const uint32_t* seed_nodes,
    const double*   seed_activations,
    uint32_t max_steps,
    double   decay_factor,
    double   spread_factor,
    double   min_activation,
    double   sigmoid_gain,
    double   sigmoid_center,
    bool     inhibition_enabled,
    double   inhibition_strength,
    uint32_t inhibition_breadth
);

// --- Results ---

uint32_t sproink_results_len(const SproinkResults* results);
void sproink_results_nodes(const SproinkResults* results, uint32_t* out);
void sproink_results_activations(const SproinkResults* results, double* out);
void sproink_results_distances(const SproinkResults* results, uint32_t* out);
void sproink_results_free(SproinkResults* results);

// --- Co-Activation Pairs ---

// Extract co-activation pairs from results. Returns owned handle.
SproinkPairs* sproink_extract_pairs(
    const SproinkResults* results,
    uint32_t num_seeds,
    const uint32_t* seed_nodes,
    double   activation_threshold
);

uint32_t sproink_pairs_len(const SproinkPairs* pairs);
void sproink_pairs_nodes(const SproinkPairs* pairs, uint32_t* out_a, uint32_t* out_b);
void sproink_pairs_activations(const SproinkPairs* pairs, double* out_a, double* out_b);
void sproink_pairs_free(SproinkPairs* pairs);

// --- Learning ---

// Pure function. No handle needed.
double sproink_oja_update(
    double current_weight,
    double activation_a,
    double activation_b,
    double learning_rate,
    double min_weight,
    double max_weight
);
```

### 7.3 Memory Ownership Rules

| Function | Ownership |
|----------|-----------|
| `sproink_graph_build` | Returns owned `Box<SproinkGraph>`. Caller must call `_free`. |
| `sproink_graph_free` | Takes ownership back, deallocates. |
| `sproink_activate` | Borrows graph immutably. Returns owned `Box<SproinkResults>`. |
| `sproink_results_*` | Borrow results immutably. Copy into caller buffer. |
| `sproink_results_free` | Takes ownership back, deallocates. |
| `sproink_extract_pairs` | Borrows results immutably. Returns owned `Box<SproinkPairs>`. |
| `sproink_pairs_*` | Borrow pairs immutably. Copy into caller buffers. |
| `sproink_pairs_free` | Takes ownership back, deallocates. |
| `sproink_oja_update` | Stateless. No allocation. |

### 7.4 Safety

All FFI functions are `unsafe` on the Rust side. Each function:
- Checks for null pointers and returns null / 0 / `min_weight` on null input
- Validates `kind` values, defaulting to `Positive` for unknown values
- Never panics across the FFI boundary (catch_unwind at boundary)

---

## 8. Error Handling Strategy

### 8.1 Internal (Rust-side)

- **Newtype constructors** return `Result<T, SproinkError>` for invalid values (NaN, out of range)
- **Graph building** is infallible given valid inputs (no I/O, no allocation failure in practice)
- **Propagation** is infallible on a valid graph
- No `panic!` in library code. Use `debug_assert!` for invariant checks in dev builds.

### 8.2 FFI Boundary

- Null pointer inputs: return null handle / zero-length results / default values
- Invalid enum values: map to safe defaults (`Positive`)
- Rust panics: caught by `catch_unwind` wrapper, return null
- No error codes or error strings in v1. Floop validates inputs before calling.

### 8.3 Error Type

```rust
#[derive(Debug, thiserror::Error)]
pub enum SproinkError {
    #[error("invalid {field}: {value} (expected finite value in valid range)")]
    InvalidValue { field: &'static str, value: f64 },

    #[error("node {node:?} out of bounds (graph has {num_nodes} nodes)")]
    NodeOutOfBounds { node: NodeId, num_nodes: u32 },
}
```

---

## 9. Module Layout

```
src/
  lib.rs          -- #![deny(unsafe_op_in_unsafe_fn)], public API re-exports, crate docs
  types.rs        -- NodeId, EdgeWeight, Activation, TagId newtypes (#[repr(transparent)])
  error.rs        -- SproinkError (thiserror)
  graph.rs        -- EdgeKind, EdgeData, EdgeInput, CsrGraph, Graph trait
  engine.rs       -- PropagationConfig (typed-builder), Seed, ActivationResult, Engine<G: Graph>
  inhibition.rs   -- InhibitionConfig, Inhibitor trait, TopMInhibitor impl
  squash.rs       -- squash_sigmoid() free function
  hebbian.rs      -- HebbianConfig (typed-builder), CoActivationPair, Learner trait, OjaLearner impl
  affinity.rs     -- AffinityConfig, AffinityGenerator trait, JaccardAffinity impl
  ffi.rs          -- extern "C" functions, opaque handle types (SproinkGraph, SproinkResults, SproinkPairs)
```

---

## 10. Testing Strategy

### 10.1 Unit Tests (per module)

Each module tests its own logic via trait mocks where dependencies exist.

| Module | Key test cases |
|--------|---------------|
| `types` | Newtype construction rejects NaN/Inf/out-of-range; `.get()` returns inner value; `new_unchecked` bypasses validation |
| `graph` | CSR build correctness; bidirectional storage; DirectionalSuppressive reversal; empty graph; self-loops |
| `engine` | Single seed propagation; multi-seed; decay attenuation over distance; conflict suppression; directional suppression (forward suppresses, reverse is skipped); max-semantics (no accumulation); activation preserved across steps (copy-init); min_activation pruning; deterministic tie-breaking (intentional improvement over Go reference which has no tie-breaker) |
| `inhibition` | Fewer nodes than breadth (no-op); winner/loser partition; suppression math; disabled config |
| `squash` | Sigmoid at center = 0.5; well-above -> ~1; well-below -> ~0; gain sensitivity |
| `hebbian` | Co-activation strengthens; high-weight stabilizes (forgetting dominates); clamp to bounds; NaN safety; seed-seed pair exclusion |
| `affinity` | Identical tags -> J=1.0; disjoint -> J=0.0; partial overlap; empty sets; threshold filtering; weight = J * max_weight |

### 10.2 Integration Tests

Test the full pipeline end-to-end with known small graphs:

1. **3-node chain**: Verify propagation reaches node 2 with expected activation after 3 steps
2. **Conflict triangle**: Verify mutual suppression prevents all-active state
3. **Directional override**: Verify deprecated node suppresses target but not reverse
4. **Affinity integration**: Build graph with affinity edges, verify they spread activation but don't dilute real edges
5. **Inhibition + sigmoid**: Verify losers are suppressed and sigmoid shapes output distribution

### 10.3 Property-Based Tests

Use `proptest` or `quickcheck`:

- **Activation bounds**: For any graph and seeds, all output activations are in [0.0, 1.0]
- **Determinism**: Same inputs always produce same outputs
- **Monotonic decay**: Activation at distance d+1 <= activation at distance d (for positive-only graphs)
- **Oja convergence**: Repeated updates converge weight to a stable value (not NaN/Inf)
- **Jaccard bounds**: Output always in [0.0, 1.0]; J(A,A) = 1.0; J(A,B) = J(B,A)

### 10.4 FFI Tests

- Round-trip: build graph via FFI, activate via FFI, extract results via FFI, compare to pure-Rust results
- Null-pointer safety: all functions handle null gracefully
- Single-free semantics: document that calling `_free` twice is undefined behavior (caller's responsibility to free exactly once)

### 10.5 Benchmarks (Criterion)

| Benchmark | Graph size | Purpose |
|-----------|-----------|---------|
| `chain_100` | 100 nodes, linear | Baseline latency |
| `random_1k_5e` | 1,000 nodes, 5 edges/node | Typical workload |
| `random_10k_10e` | 10,000 nodes, 10 edges/node | Stress test / cache behavior |

---

## 11. Scope

### In Scope

- CSR graph construction from flat edge arrays
- 3-step spreading activation with four edge kinds
- Four-denominator energy normalization
- Synchronous discrete-time updates with max-semantics
- Top-M lateral inhibition
- Configurable sigmoid squashing
- Oja's rule weight update (pure computation; no persistence)
- Co-activation pair extraction
- Jaccard affinity edge generation
- C FFI (cdylib + staticlib)
- Rayon-based parallelism
- Comprehensive test suite

### Out of Scope

- Graph persistence / SQLite integration (floop's domain)
- Seed selection / context matching (floop's domain)
- Temporal decay computation (floop pre-computes effective weights)
- Online graph mutation (graph is immutable after build)
- Step-by-step snapshot collection (floop can reconstruct from intermediate calls if needed)
- Async runtime (pure synchronous computation)
- Serde serialization of internal types
- Logging / tracing (pure library; caller instruments)
- WASM target
