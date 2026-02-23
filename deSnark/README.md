# deSnark — Distributed SNARK Protocol

Distributed proving protocol that splits a HyperPlonk SNARK across multiple parties using SumFold.

## Architecture

```
Master (party 0)          Workers (parties 1..K-1)
      │                         │
      ├── Network sync ─────────┤
      ├── SRS setup ────────────┤
      ├── Circuit gen + preprocess ┤
      ├── SumFold ──────────────┤
      ├── Distributed PCS ──────┤
      └── Proof assembly        │
```

**Star topology**: Workers bind and listen first; master connects to all workers.

## Example Binaries

| Binary | Description |
|--------|-------------|
| `dist_bench` | Benchmark: sweeps nv range, outputs CSV metrics |
| `dist_prove_demo` | Interactive demo with tracing logs |

## Configuration

TOML configs in `deSnark/examples/`:

| File | Purpose |
|------|---------|
| `bench_small.toml` | Benchmark nv=10..14, M=8, K=4 |
| `bench_large.toml` | Benchmark nv=22..26, M=8, K=4 |
| `demo_config.toml` | Demo: M=4, N=1024, K=4 |
| `hosts_4.txt` | 4 localhost ports (12350–12353) |

### Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `log_num_instances` | ν | log₂(M), number of circuit instances |
| `log_num_constraints` | μ | log₂(N), constraints per instance |
| `log_num_parties` | κ | log₂(K), number of sub-provers |
| `gate_type` | — | Gate type (`"vanilla"`) |
| `srs_path` | — | SRS cache file (auto-generated) |

## Manual Execution

```bash
# Build
cargo build --example dist_bench -p deSnark --release

# Start workers first (they bind and listen)
for i in 1 2 3; do
    ./target/release/examples/dist_bench \
        --party $i --nv-min 10 --nv-max 14 deSnark/examples/bench_small.toml &
done
sleep 2

# Start master last (connects to workers, outputs CSV to stdout)
./target/release/examples/dist_bench \
    --party 0 --nv-min 10 --nv-max 14 deSnark/examples/bench_small.toml
```

Or use the wrapper: `./scripts/run_bench.sh small`

## Tests

```bash
cargo test -p deSnark --release
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `parallel` | Rayon parallelism (default on) |
| `print-trace` | Timing traces via `ark-std` |
