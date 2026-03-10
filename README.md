# sumfold_deSNARK

Public research artifact for a distributed SNARK prover that folds `M` instances with SumFold on top of a HyperPlonk-derived backend.

This repository is for reproducible experiments and code inspection. It is not an audited production cryptography library, and it does not promise a stable public API.

## What This Repo Contains

The main artifact is `deSnark/`, supported by the workspace crates `hyperplonk/`, `subroutines/`, `arithmetic/`, `transcript/`, `deNetwork/`, and `util/`.

If your goal is to understand or run the project, start from `deSnark`.

## Start Here

If your goal is:

- verify the repo is healthy: run `make release-check`
- run the main benchmark: run `./scripts/run_interactive_bench.sh`
- understand the implementation: read [deSnark/README.md](deSnark/README.md)

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
- a minimal `deSnark` benchmark smoke test

## Main Workflow

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

## Benchmark Output

`deSnark` emits a CSV with 16 columns:

```text
nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb,d_commit_ms,sumfold_ms,sumcheck_ms,fold_ms,multi_open_ms
```

The last 5 columns are additional phase timings:

- `d_commit_ms`
- `sumfold_ms`
- `sumcheck_ms`
- `fold_ms`
- `multi_open_ms`

## Repository Map

- [deSnark/README.md](deSnark/README.md): main implementation, binaries, config files, manual execution
- [THIRD_PARTY.md](THIRD_PARTY.md): provenance and license notes for vendored code
- [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md): contribution expectations
- [.github/SECURITY.md](.github/SECURITY.md): vulnerability reporting and scope

## Important Boundaries

- This is a research artifact, not a production deployment target.
- `deSnark/` is the main code path this repository is presenting.
- Historical benchmark CSVs are not kept in the public branch unless they come with enough provenance to reproduce them.

## License And Citation

First-party code in this repository is released under the MIT license in [LICENSE](LICENSE).

Vendored code under `third_party/` retains its own upstream licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).

If you cite or reference this repository, use [CITATION.cff](CITATION.cff).
