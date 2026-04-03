# sproink

[![CI](https://github.com/nvandessel/sproink/actions/workflows/ci.yml/badge.svg)](https://github.com/nvandessel/sproink/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nvandessel/sproink/branch/main/graph/badge.svg)](https://codecov.io/gh/nvandessel/sproink)
[![Crates.io](https://img.shields.io/crates/v/sproink.svg)](https://crates.io/crates/sproink)
[![docs.rs](https://docs.rs/sproink/badge.svg)](https://docs.rs/sproink)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**In-memory spreading activation engine with CSR graph and C FFI.**

Sproink models associative memory as energy spreading through a weighted graph. Seed nodes are activated, energy propagates along edges with decay, and the result is a ranked set of associated nodes -- useful for recommendation, semantic search, and knowledge retrieval.

## Features

- **CSR graph** -- compressed sparse row storage for cache-friendly traversal
- **3-step propagation** -- spread, squash (sigmoid), inhibit per iteration
- **Lateral inhibition** -- top-M winner-take-all competition suppresses noise
- **Oja's rule** -- Hebbian weight updates with normalization for online learning
- **Jaccard affinity** -- tag-based edge generation from shared attributes
- **Rayon parallelism** -- parallel activation spreading across nodes
- **C FFI** -- full C API with opaque handles (`sproink.h`), usable from any language

## Quick Start

### Rust

```rust
use sproink::*;

let graph = CsrGraph::build(3, vec![
    EdgeInput { source: NodeId(0), target: NodeId(1), weight: EdgeWeight::new(0.8).unwrap(), kind: EdgeKind::Positive },
    EdgeInput { source: NodeId(1), target: NodeId(2), weight: EdgeWeight::new(0.6).unwrap(), kind: EdgeKind::Positive },
]);

let engine = Engine::new(graph);
let config = PropagationConfig::builder().build(); // defaults: 3 steps, decay=0.7, spread=0.85
let seeds = vec![Seed { node: NodeId(0), activation: Activation::new(1.0).unwrap() }];

let results = engine.activate(&seeds, &config);
for r in &results {
    println!("node={:?} activation={:.3} distance={}", r.node, r.activation.get(), r.distance);
}
```

### C FFI

```c
#include "sproink.h"

uint32_t sources[] = {0, 1};
uint32_t targets[] = {1, 2};
double weights[] = {0.8, 0.6};
uint8_t kinds[] = {0, 0};

SproinkGraph *g = sproink_graph_build(3, 2, sources, targets, weights, kinds);
SproinkResults *r = sproink_activate(g, 1, (uint32_t[]){0}, (double[]){1.0},
    3, 0.85, 1.0, 0.001, 5.0, 0.5, false, 0.15, 5);

uint32_t len = sproink_results_len(r);
// ... read results ...

sproink_results_free(r);
sproink_graph_free(g);
```

Link against `libsproink.so` / `libsproink.dylib` / `sproink.dll` (cdylib) or `libsproink.a` (staticlib).

## Architecture

| Module | Purpose |
|---|---|
| `graph` | CSR graph storage, edge types, construction |
| `engine` | Propagation loop: spread, squash, inhibit |
| `squash` | Sigmoid activation squashing |
| `inhibition` | Top-M lateral inhibition |
| `hebbian` | Oja's rule for Hebbian weight learning |
| `affinity` | Jaccard similarity for edge generation |
| `types` | Core newtypes (`NodeId`, `Activation`, `EdgeWeight`, etc.) |
| `error` | Error types |
| `ffi` | C-compatible API surface (`sproink.h`) |

## Algorithm

1. **Seed** -- initial nodes receive activation energy
2. **Spread** -- energy flows along outgoing edges, scaled by weight and decay
3. **Squash** -- sigmoid function normalizes activations
4. **Inhibit** -- top-M competition suppresses low activations (optional)
5. **Repeat** steps 2-4 for `max_steps` iterations
6. **Learn** -- Oja's rule updates edge weights from co-activation pairs (optional)

## Development

```bash
make ci        # fmt-check, lint, test, build
make test      # cargo test
make bench     # criterion benchmarks
make coverage  # generate lcov coverage
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development workflow.

## License

[Apache 2.0](LICENSE) -- Copyright 2026 Nic van Dessel
