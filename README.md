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
| `k` | Number of Sub_Provers (must be power of 2) |
| `M` | Number of instances (must be power of 2) |

### Linux / macOS

```bash
./scripts/run_interactive_bench.sh
```

### Windows (PowerShell)

```powershell
.\scripts\run_interactive_bench.ps1
```

The script will:
1. Validate inputs (k, M must be powers of 2; nMIN <= nMAX; nMIN >= log2(k))
2. Auto-generate a hosts file and TOML config for k parties on localhost
3. Build the `dist_bench` binary in release mode
4. Spawn k distributed processes (1 master + k-1 workers)
5. Output CSV results to `target/bench_logs/`

CSV columns: `nv, M, K, setup_ms, prover_ms, verifier_ms, proof_bytes, comm_sent, comm_recv`

## Interactive Benchmark for HyperPianist

An interactive script that prompts for parameters, auto-generates the hosts file, builds, and runs benchmarks across a range of nv values.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `nMIN` | Min n, where num_vars = n (constraints = 2^n) |
| `nMAX` | Max n (benchmark sweeps nMIN..nMAX) |
| `k` | Number of Sub_Provers (must be power of 2) |
| `M` | Number of instances (used as repetitions per nv) |

### Linux / macOS

```bash
./HyperPianist/scripts/run_interactive_bench.sh
```

### Windows (PowerShell)

```powershell
.\HyperPianist\scripts\run_interactive_bench.ps1
```

The script will:
1. Validate inputs (k must be power of 2; nMIN <= nMAX; nMIN > log2(k))
2. Auto-generate a hosts file for k parties on localhost
3. Build the `hyperpianist-bench` binary in release mode
4. For each nv in [nMIN..nMAX], spawn k processes and run M repetitions
5. Parse output and save CSV results to `HyperPianist/target/bench_logs/`

CSV columns: `nv, M, K, setup_ms, prover_ms, verifier_ms, proof_bytes, comm_sent, comm_recv`

> **Comparability note:** Both benchmarks output the same CSV columns for direct
> comparison. sumfold_deSNARK proves M instances in one batched call;
> HyperPianist proves 1 instance per call, so the script runs M times and
> **sums** all metrics (not averages). This way both CSVs report the total cost
> for M instances — batched vs sequential — making the comparison direct.
> Timings are converted from HyperPianist's native microseconds to milliseconds.
> Communication stats sum setup + proving phases (verification is local).

## License

MIT
