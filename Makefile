.PHONY: test lint fmt fmt-check build ci bench coverage clean

test:
	cargo test --all-features

lint:
	cargo clippy --all-targets --all-features -- -D warnings

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all --check

build:
	cargo build --release

bench:
	cargo bench

coverage:
	cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info

clean:
	cargo clean
	rm -f lcov.info

ci: fmt-check lint test build
