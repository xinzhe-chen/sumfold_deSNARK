#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Interactive Distributed Benchmark Runner — HyperPianist
#
# Prompts the user for benchmark parameters, generates the required
# config files, builds the binary, and runs the distributed benchmark.
#
# Parameters:
#   nMIN / nMAX  — range of n where nv = 2^n (num_vars)
#   k            — Number of Sub_Provers (must be a power of 2)
#   M            — Number of instances (used as repetitions per nv)
#
# Output: CSV file saved to HyperPianist/target/bench_logs/
#         Format matches sumfold_deSNARK for direct comparison:
#         nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv
# ═══════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$HP_ROOT/target/release/examples/hyperpianist-bench"
LOG_DIR="$HP_ROOT/target/bench_logs"
TMP_DIR="$HP_ROOT/target/bench_tmp"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

die() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# ─── Helper functions ────────────────────────────────────────────────

is_power_of_2() {
    local n=$1
    (( n > 0 && (n & (n - 1)) == 0 ))
}

log2() {
    local n=$1
    local log=0
    while (( n > 1 )); do
        (( n >>= 1 ))
        (( log++ ))
    done
    echo $log
}

# Parse HyperPianist master output into a CSV line.
# Output format: setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv
parse_hp_output() {
    local raw_file="$1"
    awk '
    /key extraction.*us$/ { setup_us = $(NF-1) }
    /bytes_sent:/ {
        for (i=1; i<=NF; i++) {
            if ($i == "bytes_sent:") { gsub(/,/, "", $(i+1)); sent += $(i+1)+0 }
            if ($i == "bytes_recv:") { recv += $(i+1)+0 }
        }
    }
    /^proving for.*us$/ { prover_us = $(NF-1) }
    /compressed:/ && !/uncompressed/ { proof_bytes = $(NF-1) }
    /^verifying for.*us$/ { verifier_us = $(NF-1) }
    END {
        setup_ms  = (setup_us+0)    / 1000.0
        prover_ms = (prover_us+0)   / 1000.0
        verify_ms = (verifier_us+0) / 1000.0
        printf "%.3f,%.3f,%.3f,%d,%d,%d\n", setup_ms, prover_ms, verify_ms, proof_bytes+0, sent+0, recv+0
    }
    ' "$raw_file"
}

# ─── Prompt for parameters ──────────────────────────────────────────

echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  HyperPianist — Interactive Benchmark Runner${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo ""

read -p "  nMIN (min n, where nv = 2^n): " NV_MIN
read -p "  nMAX (max n, where nv = 2^n): " NV_MAX
read -p "  k    (Number of Sub_Provers, power of 2): " K
read -p "  M    (Number of instances / repetitions): " M
echo ""

# ─── Validate inputs ────────────────────────────────────────────────

[[ "$NV_MIN" =~ ^[0-9]+$ ]] || die "nMIN must be a positive integer"
[[ "$NV_MAX" =~ ^[0-9]+$ ]] || die "nMAX must be a positive integer"
[[ "$K" =~ ^[0-9]+$ ]]      || die "k must be a positive integer"
[[ "$M" =~ ^[0-9]+$ ]]      || die "M must be a positive integer"

(( NV_MIN <= NV_MAX )) || die "nMIN ($NV_MIN) must be <= nMAX ($NV_MAX)"
is_power_of_2 "$K"     || die "k ($K) must be a power of 2"
(( M >= 1 ))           || die "M must be >= 1"

LOG_K=$(log2 "$K")

# nv must be large enough for the number of parties
(( NV_MIN > LOG_K )) || die "nMIN ($NV_MIN) must be > log2(k) = $LOG_K (each party needs at least 2 constraints)"

echo -e "${GREEN}Parameters:${NC}"
echo -e "  nv range      : ${NV_MIN} .. ${NV_MAX}"
echo -e "  k (Sub_Provers): ${K}  (log2 = ${LOG_K})"
echo -e "  M (instances)  : ${M}"
echo ""

# ─── Generate hosts file ────────────────────────────────────────────

mkdir -p "$TMP_DIR"
HOSTS_FILE="$TMP_DIR/hosts_${K}.txt"
BASE_PORT=8000

: > "$HOSTS_FILE"
for (( i = 0; i < K; i++ )); do
    echo "127.0.0.1:$(( BASE_PORT + i ))" >> "$HOSTS_FILE"
done
echo -e "${GREEN}Generated hosts file:${NC} $HOSTS_FILE  (${K} parties on ports ${BASE_PORT}..$(( BASE_PORT + K - 1 )))"
echo ""

# ─── Resolve nightly toolchain ────────────────────────────────────────
# A standalone stable rustc may shadow the rustup shim on Windows.
# Prepend the nightly bin directory so cargo picks up the correct compiler.

NIGHTLY_RUSTC="$(rustup which rustc --toolchain nightly-2026-02-22 2>/dev/null || true)"
if [[ -n "$NIGHTLY_RUSTC" ]]; then
    NIGHTLY_BIN="$(dirname "$NIGHTLY_RUSTC")"
    export PATH="$NIGHTLY_BIN:$PATH"
    export RUSTC="$NIGHTLY_RUSTC"
    echo -e "${GREEN}Using nightly rustc:${NC} $($NIGHTLY_RUSTC --version)"
else
    echo -e "${YELLOW}Warning: could not resolve nightly-2026-02-22 toolchain, using default${NC}"
fi

# ─── Build ───────────────────────────────────────────────────────────

echo -e "${YELLOW}Building hyperpianist-bench (release)...${NC}"
cd "$HP_ROOT"
cargo build --example hyperpianist-bench --release
echo -e "${GREEN}Build complete.${NC}"
echo ""

# ─── Kill leftover processes ─────────────────────────────────────────

cleanup() {
    for (( p = 0; p < K; p++ )); do
        local port=$(( BASE_PORT + p ))
        lsof -ti:$port 2>/dev/null | xargs kill -9 2>/dev/null || true
    done
    pkill -f "hyperpianist-bench" 2>/dev/null || true
    sleep 1
}

# ─── Setup log directory ─────────────────────────────────────────────

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$LOG_DIR/bench_nv${NV_MIN}_${NV_MAX}_k${K}_M${M}_${TIMESTAMP}.csv"

echo -e "${YELLOW}Starting benchmark: nv=${NV_MIN}..${NV_MAX}, K=${K}, M=${M} reps${NC}"
echo ""

# Write CSV header (same columns as sumfold_deSNARK for direct comparison)
CSV_HEADER="nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv"
echo "$CSV_HEADER" | tee "$CSV_FILE"

# ─── CWD into target/bench_tmp so SRS cache files stay inside target/ ─
cd "$TMP_DIR"

# ─── Run benchmarks ──────────────────────────────────────────────────

for (( nv = NV_MIN; nv <= NV_MAX; nv++ )); do
    echo -e "${CYAN}──────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  nv = ${nv}  (constraints = 2^${nv} = $(( 1 << nv )))${NC}"
    echo -e "${CYAN}──────────────────────────────────────────────────${NC}"

    # Accumulators — sum across M runs (not averaged) for direct comparison
    # with sumfold_deSNARK which reports total cost for M instances in one batch.
    #
    # Setup is counted ONCE (first rep only): SRS generation and key extraction
    # are one-time costs for both systems.
    # Prover/verifier/comm/proof are summed: M sequential proofs vs 1 batched.
    total_setup_ms=0
    total_prover_ms=0
    total_verifier_ms=0
    total_proof_bytes=0
    total_comm_sent=0
    total_comm_recv=0

    for (( rep = 1; rep <= M; rep++ )); do
        echo -e "  ${YELLOW}Repetition ${rep}/${M}...${NC}"

        # Clean up any leftover processes
        cleanup

        # Start workers (parties 1..K-1) in background
        WORKER_PIDS=()
        for (( i = 1; i < K; i++ )); do
            "$BIN" $i "$HOSTS_FILE" $nv \
                > "$LOG_DIR/hp_p${i}_nv${nv}_rep${rep}.log" 2>&1 &
            WORKER_PIDS+=($!)
        done
        sleep 2

        # Run master (party 0)
        MASTER_RAW="$LOG_DIR/hp_p0_nv${nv}_rep${rep}_raw.log"
        "$BIN" 0 "$HOSTS_FILE" $nv \
            2> "$LOG_DIR/hp_p0_nv${nv}_rep${rep}.log" > "$MASTER_RAW"

        # Parse this repetition's output
        if [[ -s "$MASTER_RAW" ]]; then
            PARSED=$(parse_hp_output "$MASTER_RAW")
            IFS=',' read -r s_ms p_ms v_ms p_bytes c_sent c_recv <<< "$PARSED"

            echo -e "  ${GREEN}setup=${s_ms}ms  prove=${p_ms}ms  verify=${v_ms}ms  proof=${p_bytes}B  sent=${c_sent}B  recv=${c_recv}B${NC}"

            # Accumulate (use awk for float addition)
            # Setup is one-time; only count from first repetition
            if (( rep == 1 )); then
                total_setup_ms="$s_ms"
            fi
            total_prover_ms=$(awk "BEGIN{printf \"%.3f\", $total_prover_ms + $p_ms}")
            total_verifier_ms=$(awk "BEGIN{printf \"%.3f\", $total_verifier_ms + $v_ms}")
            total_proof_bytes=$(( total_proof_bytes + p_bytes ))
            total_comm_sent=$(( total_comm_sent + c_sent ))
            total_comm_recv=$(( total_comm_recv + c_recv ))
        else
            echo -e "  ${RED}No output from master${NC}"
        fi

        # Wait for workers
        for pid in "${WORKER_PIDS[@]}"; do
            wait $pid 2>/dev/null || true
        done
    done

    # Output summed CSV line: total cost for M instances (sequential),
    # directly comparable with sumfold_deSNARK's M-instance batch cost
    CSV_LINE="${nv},${M},${K},${total_setup_ms},${total_prover_ms},${total_verifier_ms},${total_proof_bytes},${total_comm_sent},${total_comm_recv}"
    echo "$CSV_LINE" | tee -a "$CSV_FILE"
done

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Benchmark complete!${NC}"
echo -e "${GREEN}  Results: ${CSV_FILE}${NC}"
echo -e "${GREEN}  Logs:    ${LOG_DIR}/hp_p*.log${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
