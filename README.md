# SumFold_deSNARK and HyperPianist Benchmarks Comparison

## Prerequisites

- **Rust nightly** (pinned via `rust-toolchain.toml`)
- If your PATH resolves to a standalone stable Rust first, use `rustup run nightly-2026-02-22 ...` as shown below.

## Interactive Benchmark for SumFold_deSNARK

An interactive script that prompts for all parameters, auto-generates configs, builds, and runs the benchmark.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `nMIN` | Min n, where total constraints per instance N = 2^n |
| `nMAX` | Max n (benchmark sweeps nMIN..nMAX) |
| `k` | Number of Sub_Provers (comma-separated, each a power of 2) |
| `M` | Number of instances (must be a power of 2) |
| `reps` | Repetitions per nv for averaging (default 5) |

### Linux / macOS

```bash
./scripts/run_interactive_bench.sh
```

### Windows (PowerShell)

```powershell
.\scripts\run_interactive_bench.ps1
```

The script will:
1. Validate inputs (k, M must be powers of 2; nMIN <= nMAX; nMIN > log2(k) for k > 1)
2. Compute `RAYON_NUM_THREADS` per prover: `floor((total_cores − 2) / max_k)`, fixed across all k values
3. Auto-generate a hosts file and TOML config for k parties on localhost
4. Build the `dist_bench` binary in release mode
5. For each k, spawn k distributed processes (1 master + k−1 workers)
6. A warmup run is performed before timed repetitions
7. Output CSV results to `target/bench_logs/`

CSV columns: `nv, M, K, setup_ms, prover_ms, verifier_ms, proof_bytes, comm_sent, comm_recv, avg_cpu_pct, peak_rss_mb`

## Interactive Benchmark for HyperPianist

An interactive script that prompts for parameters, auto-generates the hosts file, builds, and runs benchmarks across a range of nv values.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `nMIN` | Min n, where num_vars = n (constraints = 2^n) |
| `nMAX` | Max n (benchmark sweeps nMIN..nMAX) |
| `k` | Number of Sub_Provers (comma-separated, each a power of 2) |
| `reps` | Repetitions per nv for averaging (default 5) |

### Linux / macOS

```bash
./HyperPianist/scripts/run_interactive_bench.sh
```

### Windows (PowerShell)

```powershell
.\HyperPianist\scripts\run_interactive_bench.ps1
```

The script will:
1. Validate inputs (k must be power of 2; nMIN <= nMAX; nMIN > log2(k) for k > 1)
2. Compute `RAYON_NUM_THREADS` per prover: `floor((total_cores − 2) / max_k)`, fixed across all k values
3. Auto-generate a hosts file for k parties on localhost
4. Build the `hyperpianist-bench` binary in release mode
5. For each nv in [nMIN..nMAX], spawn k processes and run `reps` repetitions
6. Each binary invocation performs an internal warmup d_prove + 20× timed proving
7. Parse output and save CSV results to `HyperPianist/target/bench_logs/`

CSV columns: `nv, M, K, setup_ms, prover_ms, verifier_ms, proof_bytes, comm_sent, comm_recv, avg_cpu_pct, peak_rss_mb`

## Measurement Alignment

Both benchmarks output the same CSV columns for direct comparison.
The following semantics are unified:

| Metric | sumfold_deSNARK | HyperPianist |
|--------|-----------------|---------------|
| **M** (CSV) | Number of instances batched via SumFold | Always 1 (HP can't batch) |
| **prover_ms** | Per-instance average (total ÷ M ÷ reps) | Per-call average (total ÷ internal_reps ÷ shell_reps) |
| **verifier_ms** | Per-instance average | Per-call average |
| **setup_ms** | `preprocess()` + PCS trim (excludes SRS load & circuit gen) | `d_preprocess()` (excludes SRS load & circuit gen) |
| **comm_sent/recv** | Proving phase only (per instance per rep average, ÷ M ÷ reps) | Proving phase only (per internal_rep average) |
| **avg_cpu_pct** | CPU ÷ wall of proving phase | CPU ÷ wall of timed proving loop |
| **peak_rss_mb** | Max RSS during a single rep | Max RSS during the entire binary invocation |

> **Workload note:** Both systems use similar `MockCircuit` construction (first 1/4
> constraints with random witnesses, rest with copies; gate function is vanilla plonk).
> sumfold uses identity permutation; HyperPianist uses a non-trivial cross-party
> permutation. This reflects the architectural difference — sumfold bypasses
> permutation arguments via SumFold.

## License

MIT
