# deSnark

Detailed notes for the main distributed proving prototype used in the root-level `sumfold_deSNARK` research artifact. Start with the repository [README](../README.md) for setup, policy, and public release context.

## Purpose

`deSnark` is the main first-party crate in this repository. It packages the distributed proving flow that combines:

- circuit generation and preprocessing
- SumFold-based instance folding
- HyperPlonk-style sumcheck and gate verification
- distributed polynomial commitment opening

## Architecture

```text
Master (party 0)          Workers (party 1..K-1)
      |                         |
      |-- network bootstrap ----|
      |-- SRS setup/cache ------|
      |-- preprocess -----------|
      |-- SumFold --------------|
      |-- unified sumcheck -----|
      |-- distributed PCS ------|
      `-- proof assembly        |
```

Workers bind first. The master connects last.

## Example Binaries

| Binary | Purpose |
| --- | --- |
| `dist_bench` | Distributed benchmark runner over an `nv` range, with warmup and repeated timed reps |
| `dist_prove_demo` | Interactive demo with logging |
| `lightweight_bench` | Single-process phase breakdown for local profiling |

## Configuration

TOML examples are under `deSnark/examples/`.

| File | Purpose |
| --- | --- |
| `bench_small.toml` | Sample benchmark config for `nv = 10..14`, `M = 8`, `K = 4` |
| `bench_large.toml` | Sample benchmark config for `nv = 22..26`, `M = 8`, `K = 4` |
| `demo_config.toml` | Demo config for `M = 4`, `N = 1024`, `K = 4` |
| `hosts_4.txt` | Localhost host list for four parties |

Important config fields:

| Parameter | Meaning |
| --- | --- |
| `log_num_instances` | `log2(M)`, number of instances folded together |
| `log_num_constraints` | `log2(N)`, constraints per instance |
| `log_num_parties` | `log2(K)`, number of sub-provers |
| `gate_type` | Gate family, currently `"vanilla"` in the public examples |
| `srs_path` | Optional SRS cache path |

## Manual Execution

```bash
cargo build --example dist_bench -p deSnark --release

for i in 1 2 3; do
  RAYON_NUM_THREADS=2 ./target/release/examples/dist_bench \
    --party "$i" --nv-min 10 --nv-max 14 --reps 5 deSnark/examples/bench_small.toml &
done

sleep 2

RAYON_NUM_THREADS=2 ./target/release/examples/dist_bench \
  --party 0 --nv-min 10 --nv-max 14 --reps 5 deSnark/examples/bench_small.toml
```

For an interactive wrapper, use:

```bash
./scripts/run_interactive_bench.sh
```

## Raw CSV Output

The `dist_bench` example emits:

```text
nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb,d_commit_ms,sumfold_ms,sumcheck_ms,fold_ms,multi_open_ms
```

The first 11 columns match the comparison schema used by the bundled `HyperPianist` baseline. The last 5 columns expose per-phase timings for the public artifact.

## Timing Scope

| Metric | Scope |
| --- | --- |
| `setup_ms` | `preprocess()` plus PCS trim, excluding circuit generation and SRS load |
| `prover_ms` | Proving-phase average per instance |
| `verifier_ms` | Verification-phase average per instance |
| `comm_sent` / `comm_recv` | Proving-phase network traffic average per instance |
| `avg_cpu_pct` | CPU time divided by proving-phase wall time |
| `d_commit_ms` | Distributed commitment average per instance |
| `sumfold_ms` | SumFold phase average per instance |
| `sumcheck_ms` | Unified sumcheck average per instance |
| `fold_ms` | Commitment folding average per instance |
| `multi_open_ms` | PCS opening average per instance |

## Standalone Verification

The public verification path is the `verify` API exported by the crate root. It verifies from the proof and verifying key, without requiring prover state or the proving key.

## Tests

```bash
cargo test -p deSnark --release
```

## Security Note

This crate is part of a research artifact. It is useful for reproducible experiments and inspection, but it is not presented as an audited production implementation.
