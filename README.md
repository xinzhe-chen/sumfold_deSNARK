# sumfold_deSNARK

Public research artifact for a distributed SNARK prover that folds `M` instances with SumFold on top of a HyperPlonk-derived backend.

This repository is for reproducible experiments and code inspection. It is not an audited production cryptography library, and it does not promise a stable public API.

## What This Repo Contains

There are two top-level systems in this repository:

- `deSnark/`: the main artifact in this repo. This is the implementation of the distributed SumFold-based prover.
- `HyperPianist/`: a bundled comparison baseline kept in-tree so both systems can be benchmarked from one checkout.

If you only remember one thing: the main project is `deSnark`; `HyperPianist` is here only for comparison.

## Start Here

If your goal is:

- verify the repo is healthy: run `make release-check`
- run the main benchmark (`deSnark`): run `./scripts/run_interactive_bench.sh`
- run the comparison baseline (`HyperPianist`): run `./HyperPianist/scripts/run_interactive_bench.sh`
- understand the main implementation: read [deSnark/README.md](deSnark/README.md)

## Terminology

The benchmark scripts and CSV files use these parameters:

- `nv`: `log2(N)`, where `N` is the number of constraints per instance
- `M`: number of instances folded together
- `K`: number of parties / sub-provers

So when you see `nv=20, M=8, K=4`, it means:

- each instance has `2^20` constraints
- 8 instances are folded together
- the work is distributed across 4 parties

## Quick Setup

The workspace is pinned to `nightly-2026-02-22` via [rust-toolchain.toml](rust-toolchain.toml).

```bash
rustup toolchain install nightly-2026-02-22 --component rustfmt --component clippy
make release-check
```

`make release-check` runs:

- formatting check
- clippy with `-D warnings`
- workspace tests
- local Markdown link checks
- minimal smoke benchmarks for both `deSnark` and `HyperPianist`

## The Main Workflow

### 1. Validate The Repo

```bash
make release-check
```

Or run individual checks:

```bash
make fmt-check
make clippy
make test
make readme-links
make bench-smoke
```

### 2. Run The Main Benchmark

```bash
./scripts/run_interactive_bench.sh
```

This script:

- asks for `nv` range, `K`, `M`, and repetition count
- builds `deSnark/examples/dist_bench`
- generates a localhost host file and config automatically
- launches the distributed benchmark locally
- writes CSV output to `target/bench_logs/`

### 3. Run The Comparison Baseline

```bash
./HyperPianist/scripts/run_interactive_bench.sh
```

This script does the same kind of localhost benchmark setup for the bundled baseline and writes CSV output to `HyperPianist/target/bench_logs/`.

## How The Two Benchmark Outputs Relate

`deSnark` emits a CSV with 16 columns:

```text
nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb,d_commit_ms,sumfold_ms,sumcheck_ms,fold_ms,multi_open_ms
```

`HyperPianist` emits a CSV with 11 columns:

```text
nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb
```

The first 11 columns are intentionally aligned so the two systems can be compared directly.

The extra 5 `deSnark` columns are additional phase timings:

- `d_commit_ms`
- `sumfold_ms`
- `sumcheck_ms`
- `fold_ms`
- `multi_open_ms`

## Repository Map

- [deSnark/README.md](deSnark/README.md): main implementation, binaries, config files, manual execution
- [HyperPianist/README.md](HyperPianist/README.md): scope of the bundled comparison baseline
- [HyperPianist/bench_results/README.md](HyperPianist/bench_results/README.md): policy for curated benchmark artifacts
- [THIRD_PARTY.md](THIRD_PARTY.md): provenance and license notes for vendored code and the baseline snapshot
- [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md): contribution expectations
- [.github/SECURITY.md](.github/SECURITY.md): vulnerability reporting and scope

## Important Boundaries

- This is a research artifact, not a production deployment target.
- `deSnark/` is the main code path this repository is presenting.
- `HyperPianist/` is preserved as a comparison baseline, not as the primary deliverable.
- Historical benchmark CSVs are not kept in the public branch unless they come with enough provenance to reproduce them.

## License And Citation

First-party code in this repository is released under the MIT license in [LICENSE](LICENSE).

Bundled vendored code and baseline snapshots retain their own upstream licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).

If you cite or reference this repository, use [CITATION.cff](CITATION.cff).
