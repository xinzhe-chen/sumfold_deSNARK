#!/bin/bash
# Distributed Benchmark Runner — 4 parties on localhost
#
# Usage:
#   ./scripts/run_bench.sh small          # nv = 10..14
#   ./scripts/run_bench.sh large          # nv = 22..26
#   ./scripts/run_bench.sh small 3        # nv = 10..14, 3 repetitions
#   ./scripts/run_bench.sh --nv-min 12 --nv-max 16 deSnark/examples/bench_small.toml
#
# Output: CSV on stdout (master), per-party logs in target/bench_logs/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT/target/release/examples/dist_bench"
LOG_DIR="$ROOT/target/bench_logs"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

die() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# ─── Build if needed ──────────────────────────────────────────────
build_if_needed() {
    if [[ ! -f "$BIN" ]]; then
        echo -e "${YELLOW}Building dist_bench (release)...${NC}" >&2
        cd "$ROOT"
        cargo build --example dist_bench -p deSnark --release
        echo -e "${GREEN}Build complete.${NC}" >&2
    fi
}

# ─── Kill leftover processes on benchmark ports ───────────────────
cleanup_ports() {
    for port in 12350 12351 12352 12353; do
        lsof -ti:$port 2>/dev/null | xargs kill -9 2>/dev/null || true
    done
    pkill -f "dist_bench" 2>/dev/null || true
    sleep 1
}

# ─── Main ─────────────────────────────────────────────────────────
build_if_needed

# Parse profile shortcuts or raw args
REPS=1
case "${1:-}" in
    small)
        CONFIG="$ROOT/deSnark/examples/bench_small.toml"
        NV_MIN=10; NV_MAX=14
        REPS="${2:-1}"
        ;;
    large)
        CONFIG="$ROOT/deSnark/examples/bench_large.toml"
        NV_MIN=22; NV_MAX=26
        REPS="${2:-1}"
        ;;
    --nv-min)
        # Pass-through raw args: --nv-min X --nv-max Y [--reps R] config.toml
        NV_MIN="$2"; shift 2
        [[ "$1" == "--nv-max" ]] || die "Expected --nv-max after --nv-min"
        NV_MAX="$2"; shift 2
        if [[ "${1:-}" == "--reps" ]]; then REPS="$2"; shift 2; fi
        CONFIG="${1:?Missing config.toml path}"
        ;;
    *)
        echo "Usage:"
        echo "  $0 small [reps]              # nv=10..14, M=8, K=4"
        echo "  $0 large [reps]              # nv=22..26, M=8, K=4"
        echo "  $0 --nv-min N --nv-max N [--reps R] config.toml"
        exit 0
        ;;
esac

[[ -f "$CONFIG" ]] || die "Config not found: $CONFIG"

cleanup_ports
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo -e "${YELLOW}Benchmark: nv=${NV_MIN}..${NV_MAX}, reps=${REPS}${NC}" >&2
echo -e "${YELLOW}Config: ${CONFIG}${NC}" >&2

# Start workers (parties 1-3) in background
WORKER_PIDS=()
for i in 1 2 3; do
    "$BIN" --party $i --nv-min $NV_MIN --nv-max $NV_MAX --reps $REPS "$CONFIG" \
        > "$LOG_DIR/p${i}.log" 2>&1 &
    WORKER_PIDS+=($!)
done
sleep 2

# Run master (party 0) — filter CSV from stdout (ark-std print-trace pollutes stdout)
CSV_FILE="$LOG_DIR/bench_nv${NV_MIN}_${NV_MAX}_${TIMESTAMP}.csv"
RAW_FILE="$LOG_DIR/p0_raw.log"
echo -e "${GREEN}Running master...${NC}" >&2

"$BIN" --party 0 --nv-min $NV_MIN --nv-max $NV_MAX --reps $REPS "$CONFIG" \
    2> "$LOG_DIR/p0.log" > "$RAW_FILE"

# Extract only CSV lines (header + data rows) from the raw output
CSV_HEADER="nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb"
echo "$CSV_HEADER" | tee "$CSV_FILE"
grep '^[0-9]\+,' "$RAW_FILE" | tee -a "$CSV_FILE"

# Wait for workers
for pid in "${WORKER_PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done

echo "" >&2
echo -e "${GREEN}Done. Results saved to: ${CSV_FILE}${NC}" >&2
