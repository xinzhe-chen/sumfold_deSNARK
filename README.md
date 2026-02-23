# SumFold-deSNARK

## Prerequisites

- **Rust nightly** (pinned via `rust-toolchain.toml`)
- If your PATH resolves to a standalone stable Rust first, use `rustup run nightly-2026-02-22 ...` as shown below.

## Quick Start

### Linux / macOS

```bash
# Build
rustup run nightly-2026-02-22 cargo build --release

# Small benchmark (nv = 10..14, M=8, K=4)
rustup run nightly-2026-02-22 bash ./scripts/run_bench.sh small

# Large benchmark (nv = 22..26, M=8, K=4)
rustup run nightly-2026-02-22 bash ./scripts/run_bench.sh large
```

### Windows (PowerShell)

```powershell
# Build
rustup run nightly-2026-02-22 cargo build --release

# Small benchmark (nv = 10..14, M=8, K=4)
.\scripts\run_bench.ps1 small

# Large benchmark (nv = 22..26, M=8, K=4)
.\scripts\run_bench.ps1 large
```

Output is a CSV table printed to stdout and saved to `target/bench_logs/`.

SRS files are generated automatically on first run and cached for reuse.

## License

MIT
