#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Interactive Distributed Benchmark Runner — sumfold_deSNARK
#
# Prompts the user for benchmark parameters, generates the required
# config files, builds the binary, and runs the distributed benchmark.
#
# Parameters:
#   nMIN / nMAX  — range of n where nv = 2^n (log_num_constraints)
#   k            — Number of Sub_Provers (comma-separated, each a power of 2)
#   M            — Number of instances  (must be a power of 2)
#   reps         — Repetitions per nv for averaging
#
# Thread control:
#   RAYON_NUM_THREADS is set per prover to floor(total_cores / max_K),
#   so that every prover gets the SAME number of threads regardless of K.
#   This ensures a fair comparison across different K values.
#
# Output: CSV file saved to target/bench_logs/
# ═══════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT/target/release/examples/dist_bench"
LOG_DIR="$ROOT/target/bench_logs"
TMP_DIR="$ROOT/target/bench_tmp"

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

# Detect total physical/logical CPU cores
detect_cores() {
    if [[ "$(uname)" == "Darwin" ]]; then
        sysctl -n hw.ncpu
    elif command -v nproc &>/dev/null; then
        nproc
    elif [[ -f /proc/cpuinfo ]]; then
        grep -c '^processor' /proc/cpuinfo
    else
        echo 1
    fi
}

# ─── Prompt for parameters ──────────────────────────────────────────

echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  sumfold_deSNARK — Interactive Benchmark Runner${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo ""

TOTAL_CORES=$(detect_cores)
echo -e "  Detected ${GREEN}${TOTAL_CORES}${NC} CPU cores on this machine."
echo ""

read -p "  nMIN (min n, where nv = 2^n): " NV_MIN
read -p "  nMAX (max n, where nv = 2^n): " NV_MAX
read -p "  k    (Sub_Provers, comma-separated, each power of 2, e.g. 1,2,4,8): " K_INPUT
read -p "  M    (Number of instances, power of 2):   " M
read -p "  reps (Repetitions per nv for averaging, default 5): " REPS
REPS=${REPS:-5}
echo ""

# ─── Validate common inputs ─────────────────────────────────────────

[[ "$NV_MIN" =~ ^[0-9]+$ ]] || die "nMIN must be a positive integer"
[[ "$NV_MAX" =~ ^[0-9]+$ ]] || die "nMAX must be a positive integer"
[[ "$M" =~ ^[0-9]+$ ]]      || die "M must be a positive integer"
[[ "$REPS" =~ ^[0-9]+$ ]]   || die "reps must be a positive integer"

(( NV_MIN <= NV_MAX )) || die "nMIN ($NV_MIN) must be <= nMAX ($NV_MAX)"
is_power_of_2 "$M"     || die "M ($M) must be a power of 2"
(( REPS >= 1 ))        || die "reps must be >= 1"

LOG_M=$(log2 "$M")

# Parse comma-separated k values
IFS=',' read -ra K_VALUES <<< "$K_INPUT"
[[ ${#K_VALUES[@]} -gt 0 ]] || die "Must provide at least one k value"

for K in "${K_VALUES[@]}"; do
    K=$(echo "$K" | tr -d ' ')
    [[ "$K" =~ ^[0-9]+$ ]] || die "k value '$K' must be a positive integer"
    is_power_of_2 "$K"     || die "k ($K) must be a power of 2"
done

# ─── Compute fixed threads per prover (based on max k) ────────────────
MAX_K=0
for K in "${K_VALUES[@]}"; do
    K=$(echo "$K" | tr -d ' ')
    (( K > MAX_K )) && MAX_K=$K
done
USABLE_CORES=$(( TOTAL_CORES - 2 ))
(( USABLE_CORES >= 1 )) || USABLE_CORES=1
THREADS_PER_PROVER=$(( USABLE_CORES / MAX_K ))
(( THREADS_PER_PROVER >= 1 )) || die "max k=$MAX_K exceeds usable cores ($USABLE_CORES = $TOTAL_CORES - 2). Cannot assign >= 1 thread per prover."

echo -e "${GREEN}Parameters:${NC}"
echo -e "  nv range      : ${NV_MIN} .. ${NV_MAX}"
echo -e "  k values      : ${K_VALUES[*]}"
echo -e "  M (instances) : ${M}  (log2 = ${LOG_M})"
echo -e "  reps          : ${REPS}"
echo -e "  Total cores   : ${TOTAL_CORES}  (reserving 2 for OS → ${USABLE_CORES} usable)"
echo -e "  Threads/prover: ${THREADS_PER_PROVER}  (fixed by max k=${MAX_K}: ${MAX_K} x ${THREADS_PER_PROVER} = $(( MAX_K * THREADS_PER_PROVER )) <= ${USABLE_CORES} usable cores)"
echo ""

# ─── Resolve nightly toolchain ────────────────────────────────────────

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

echo -e "${YELLOW}Building dist_bench (release)...${NC}"
cd "$ROOT"
cargo build --example dist_bench -p deSnark --release
echo -e "${GREEN}Build complete.${NC}"
echo ""

# ─── Iterate over each k value ───────────────────────────────────────

for K in "${K_VALUES[@]}"; do
    K=$(echo "$K" | tr -d ' ')
    LOG_K=$(log2 "$K")

    # Validate nMIN > log2(K) so effective_nv > 0 (consistent with HyperPianist)
    if (( K > 1 )); then
        (( NV_MIN > LOG_K )) || { echo -e "${RED}Error: nMIN ($NV_MIN) must be > log2(k=$K) = $LOG_K for K>1, skipping k=$K${NC}"; continue; }
    fi

    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Running with k=${K} (log2=${LOG_K})${NC}"
    echo -e "${CYAN}  Threads per prover: ${THREADS_PER_PROVER} (fixed, ${K} provers x ${THREADS_PER_PROVER} threads = $(( K * THREADS_PER_PROVER )) used / ${TOTAL_CORES} cores)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"

    # ─── Generate hosts file ─────────────────────────────────────────
    mkdir -p "$TMP_DIR"
    HOSTS_FILE="$TMP_DIR/hosts_${K}.txt"
    BASE_PORT=12350

    : > "$HOSTS_FILE"
    for (( i = 0; i < K; i++ )); do
        echo "127.0.0.1:$(( BASE_PORT + i ))" >> "$HOSTS_FILE"
    done
    echo -e "${GREEN}Generated hosts file:${NC} $HOSTS_FILE  (${K} parties on ports ${BASE_PORT}..$(( BASE_PORT + K - 1 )))"

    # ─── Generate TOML config ────────────────────────────────────────
    CONFIG_FILE="$TMP_DIR/bench_config.toml"
    cat > "$CONFIG_FILE" <<EOF
# Auto-generated benchmark config
# nv range: ${NV_MIN}..${NV_MAX}, M=${M}, K=${K}

[config]
log_num_instances = ${LOG_M}
log_num_constraints = ${NV_MIN}
gate_type = "vanilla"
log_num_parties = ${LOG_K}
srs_path = "srs_interactive_nv${NV_MAX}.params"

[network]
hosts_file = "${HOSTS_FILE}"
EOF
    echo -e "${GREEN}Generated TOML config:${NC} $CONFIG_FILE"

    # ─── Kill leftover processes ─────────────────────────────────────
    for (( p = 0; p < K; p++ )); do
        port=$(( BASE_PORT + p ))
        lsof -ti:$port 2>/dev/null | xargs kill -9 2>/dev/null || true
    done
    pkill -f "dist_bench" 2>/dev/null || true
    sleep 1

    # ─── Setup log directory ─────────────────────────────────────────
    mkdir -p "$LOG_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CSV_FILE="$LOG_DIR/bench_nv${NV_MIN}_${NV_MAX}_k${K}_M${M}_${TIMESTAMP}.csv"
    RAW_FILE="$LOG_DIR/p0_raw.log"

    echo -e "${YELLOW}Starting benchmark: nv=${NV_MIN}..${NV_MAX}, K=${K}, M=${M}, threads/prover=${THREADS_PER_PROVER}${NC}"
    echo ""

    # ─── CWD into target/bench_tmp so SRS cache files stay inside target/ ─
    cd "$TMP_DIR"

    # ─── Start workers (parties 1..K-1) in background ────────────────
    WORKER_PIDS=()
    for (( i = 1; i < K; i++ )); do
        RAYON_NUM_THREADS=$THREADS_PER_PROVER \
        "$BIN" --party $i --nv-min $NV_MIN --nv-max $NV_MAX --reps $REPS "$CONFIG_FILE" \
            > "$LOG_DIR/p${i}.log" 2>&1 &
        WORKER_PIDS+=($!)
    done

    # Give workers time to bind their ports (skip for k=1)
    if (( K > 1 )); then
        sleep 2
    fi

    # ─── Run master (party 0) ────────────────────────────────────────
    echo -e "${GREEN}Running master (party 0) with RAYON_NUM_THREADS=${THREADS_PER_PROVER}...${NC}"
    RAYON_NUM_THREADS=$THREADS_PER_PROVER \
    "$BIN" --party 0 --nv-min $NV_MIN --nv-max $NV_MAX --reps $REPS "$CONFIG_FILE" \
        2> "$LOG_DIR/p0.log" > "$RAW_FILE"

    # ─── Extract CSV ─────────────────────────────────────────────────
    CSV_HEADER="nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb,d_commit_ms,sumfold_ms,sumcheck_ms,fold_ms,multi_open_ms"
    echo "$CSV_HEADER" | tee "$CSV_FILE"
    grep '^[0-9]\+,' "$RAW_FILE" | tee -a "$CSV_FILE"

    # ─── Wait for workers ────────────────────────────────────────────
    for pid in "${WORKER_PIDS[@]}"; do
        wait $pid 2>/dev/null || true
    done

    echo ""
    echo -e "${GREEN}  Results for k=${K}: ${CSV_FILE}${NC}"
    echo ""

    # Return to ROOT for next iteration
    cd "$ROOT"
done

echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All benchmarks complete!${NC}"
echo -e "${GREEN}  Logs:    ${LOG_DIR}/p*.log${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
