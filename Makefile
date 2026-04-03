.PHONY: test lint fmt fmt-check build ci bench coverage clean

test:
	cargo test

lint:
	cargo clippy -- -D warnings

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all --check

build:
	cargo build --release

bench:
	cargo bench

coverage:
	cargo llvm-cov --lcov --output-path lcov.info

clean:
	cargo clean
	rm -f lcov.info

ci: fmt-check lint test build
