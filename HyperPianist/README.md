# HyperPianist Baseline Snapshot

This directory keeps a bundled comparison baseline for the public `sumfold_deSNARK` artifact. It is preserved in-tree so that benchmark reproduction can compare the main `deSnark` path against a single-checkout baseline.

This code is not the primary product of this repository. Maintenance here is limited to keeping comparison builds and scripts reproducible. It should not be interpreted as an audited or production-ready component of the main artifact.

## Scope In This Repository

- used for side-by-side benchmark comparisons
- kept under its original directory layout to avoid breaking benchmark scripts
- documented separately from the main first-party crates

See the repository [README](../README.md) for the main artifact overview and [THIRD_PARTY.md](../THIRD_PARTY.md) for provenance notes.

## Build

This workspace also requires the pinned nightly toolchain.

```bash
cargo build --manifest-path HyperPianist/Cargo.toml --release
```

For the benchmark example used in the public artifact:

```bash
cargo build --manifest-path HyperPianist/Cargo.toml --example hyperpianist-bench --release
```

## Tests

Some tests are ordinary Rust tests; some distributed checks live under legacy `dTests/` paths.

```bash
cargo test --manifest-path HyperPianist/Cargo.toml --release
```

## Benchmark Entry Point

For the comparison path used by this repository:

```bash
./HyperPianist/scripts/run_interactive_bench.sh
```

That wrapper:

- prompts for `nv`, `k`, and repetition settings
- builds `hyperpianist-bench`
- runs a localhost benchmark topology
- writes CSV files under `HyperPianist/target/bench_logs/`

The CSV schema exposed by the wrapper is:

```text
nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb
```

`M` is always `1` for this baseline.
