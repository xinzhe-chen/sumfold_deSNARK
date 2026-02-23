# SumFold-deSNARK

## Prerequisites

- **Rust nightly** (auto-selected via `rust-toolchain.toml`)

## Quick Start

```bash
# Build
cargo build --release

# Small benchmark (nv = 10..14, M=8, K=4)
./scripts/run_bench.sh small          # Linux / macOS
.\scripts\run_bench.ps1 small         # Windows (PowerShell)

# Large benchmark (nv = 22..26, M=8, K=4)
./scripts/run_bench.sh large          # Linux / macOS
.\scripts\run_bench.ps1 large         # Windows (PowerShell)
```

Output is a CSV table printed to stdout and saved to `target/bench_logs/`.

SRS files are generated automatically on first run and cached for reuse.

## License

MIT
