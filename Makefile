SHELL := /usr/bin/env bash

.PHONY: fmt fmt-check clippy test readme-links bench-smoke release-check

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	bash ./scripts/run_tests.sh

readme-links:
	python3 ./scripts/check_local_links.py README.md deSnark/README.md THIRD_PARTY.md .github/CONTRIBUTING.md .github/SECURITY.md .github/CODE_OF_CONDUCT.md

bench-smoke:
	bash ./scripts/smoke_bench.sh

release-check: fmt-check clippy test readme-links bench-smoke
