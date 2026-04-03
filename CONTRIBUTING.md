# Contributing to Sproink

Thank you for your interest in contributing! This document covers the development workflow and standards.

## Prerequisites

- **Rust 1.83+** (2024 edition)
- **cargo**, **clippy**, **rustfmt** (included with rustup)
- **cargo-llvm-cov** (optional, for coverage: `cargo install cargo-llvm-cov`)

## Workflow

1. **Open an issue** describing the change you'd like to make
2. **Fork** the repository and create a feature branch
3. **Write tests first** (TDD: red, green, refactor)
4. **Implement** the minimum to pass
5. **Run the quality gate**: `make ci`
6. **Open a PR** against `main`

## Code Standards

- **Formatting**: `cargo fmt` -- enforced in CI
- **Linting**: `cargo clippy -- -D warnings` -- all warnings are errors
- **No `unwrap()` in library code** -- use `Result` or `expect()` with a descriptive message
- **`unsafe` only in `ffi.rs`** -- the FFI module is the sole location for unsafe code
- **No TODO comments** -- file an issue instead

## Testing

| Type | Location | Run |
|---|---|---|
| Unit tests | `src/*.rs` (inline `#[cfg(test)]`) | `cargo test` |
| Integration tests | `tests/` | `cargo test` |
| Property tests | `tests/property_tests.rs` (proptest) | `cargo test` |
| FFI tests | `tests/ffi_tests.rs` | `cargo test` |
| Benchmarks | `benches/` (criterion) | `cargo bench` |

## Makefile Targets

```bash
make test          # cargo test
make lint          # cargo clippy -- -D warnings
make fmt           # cargo fmt
make fmt-check     # cargo fmt --all --check
make build         # cargo build --release
make ci            # fmt-check + lint + test + build
make bench         # cargo bench
make coverage      # generate lcov coverage report
make clean         # cargo clean
```

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add weighted decay scheduling
fix: handle empty seed list without panic
test: add property tests for sigmoid squashing
docs: update FFI usage examples
refactor: extract propagation step into helper
```

## Pull Requests

- Keep PRs focused -- one logical change per PR
- Fill out the PR template
- All CI checks must pass
- Expect at least one review before merge

## Questions?

Open a [discussion](https://github.com/nvandessel/sproink/discussions) or comment on the relevant issue.
