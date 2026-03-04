#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Interactive Distributed Benchmark Runner — HyperPianist
#
# Prompts the user for benchmark parameters, generates the required
# config files, builds the binary, and runs the distributed benchmark.
#
# Parameters:
#   nMIN / nMAX  — range of n where nv = 2^n (num_vars)
#   k            — Number of Sub_Provers (comma-separated, each a power of 2)
#   reps         — Repetitions per nv for averaging (HP always proves 1 instance)
#
# Thread control:
#   RAYON_NUM_THREADS is set per prover to floor(total_cores / max_K),
#   so that every prover gets the SAME number of threads regardless of K.
#   This ensures a fair comparison across different K values.
#
# Output: CSV file saved to HyperPianist/target/bench_logs/
#         Format matches sumfold_deSNARK for direct comparison:
#         nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb
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

# Parse the CSV line from HyperPianist binary stdout.
# Binary outputs: setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,cpu_ms,wall_ms,peak_rss_mb
# This function extracts the first CSV data line (skipping any debug output).
parse_hp_csv() {
    local raw_file="$1"
    grep '^[0-9]' "$raw_file" | head -1
}

# ─── Prompt for parameters ──────────────────────────────────────────

echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  HyperPianist — Interactive Benchmark Runner${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo ""

TOTAL_CORES=$(detect_cores)
echo -e "  Detected ${GREEN}${TOTAL_CORES}${NC} CPU cores on this machine."
echo ""

read -p "  nMIN (min n, where nv = 2^n): " NV_MIN
read -p "  nMAX (max n, where nv = 2^n): " NV_MAX
read -p "  k    (Sub_Provers, comma-separated, each power of 2, e.g. 1,2,4,8): " K_INPUT
read -p "  reps (Repetitions per nv for averaging, default 5): " REPS
REPS=${REPS:-5}
echo ""

# ─── Validate common inputs ─────────────────────────────────────────

[[ "$NV_MIN" =~ ^[0-9]+$ ]] || die "nMIN must be a positive integer"
[[ "$NV_MAX" =~ ^[0-9]+$ ]] || die "nMAX must be a positive integer"
[[ "$REPS" =~ ^[0-9]+$ ]]   || die "reps must be a positive integer"

(( NV_MIN <= NV_MAX )) || die "nMIN ($NV_MIN) must be <= nMAX ($NV_MAX)"
(( REPS >= 1 ))        || die "reps must be >= 1"

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
echo -e "  M (instances) : 1  (HyperPianist always proves 1 instance)"
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

echo -e "${YELLOW}Building hyperpianist-bench (release)...${NC}"
cd "$HP_ROOT"
cargo build --example hyperpianist-bench --release
echo -e "${GREEN}Build complete.${NC}"
echo ""

# ─── Iterate over each k value ───────────────────────────────────────

for K in "${K_VALUES[@]}"; do
    K=$(echo "$K" | tr -d ' ')
    LOG_K=$(log2 "$K")

    # nv must be large enough for the number of parties
    if (( K > 1 )); then
        (( NV_MIN > LOG_K )) || { echo -e "${RED}Error: nMIN ($NV_MIN) must be > log2(k=$K) = $LOG_K, skipping k=$K${NC}"; continue; }
    fi

    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Running with k=${K} (log2=${LOG_K})${NC}"
    echo -e "${CYAN}  Threads per prover: ${THREADS_PER_PROVER} (fixed, ${K} provers x ${THREADS_PER_PROVER} threads = $(( K * THREADS_PER_PROVER )) used / ${TOTAL_CORES} cores)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"

    # ─── Generate hosts file ─────────────────────────────────────────
    mkdir -p "$TMP_DIR"
    HOSTS_FILE="$TMP_DIR/hosts_${K}.txt"
    BASE_PORT=8000

    : > "$HOSTS_FILE"
    for (( i = 0; i < K; i++ )); do
        echo "127.0.0.1:$(( BASE_PORT + i ))" >> "$HOSTS_FILE"
    done
    echo -e "${GREEN}Generated hosts file:${NC} $HOSTS_FILE  (${K} parties on ports ${BASE_PORT}..$(( BASE_PORT + K - 1 )))"
    echo ""

    # ─── Kill leftover processes ─────────────────────────────────────
    cleanup() {
        for (( p = 0; p < K; p++ )); do
            local port=$(( BASE_PORT + p ))
            lsof -ti:$port 2>/dev/null | xargs kill -9 2>/dev/null || true
        done
        pkill -f "hyperpianist-bench" 2>/dev/null || true
        sleep 1
    }

    # ─── Setup log directory ─────────────────────────────────────────
    mkdir -p "$LOG_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CSV_FILE="$LOG_DIR/bench_nv${NV_MIN}_${NV_MAX}_k${K}_reps${REPS}_${TIMESTAMP}.csv"

    echo -e "${YELLOW}Starting benchmark: nv=${NV_MIN}..${NV_MAX}, K=${K}, M=1, reps=${REPS}, threads/prover=${THREADS_PER_PROVER}${NC}"
    echo ""

    # Write CSV header (same columns as sumfold_deSNARK for direct comparison)
    CSV_HEADER="nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb"
    echo "$CSV_HEADER" | tee "$CSV_FILE"

    # ─── CWD into target/bench_tmp so SRS cache files stay inside target/ ─
    cd "$TMP_DIR"

    # ─── Run benchmarks ──────────────────────────────────────────────

    for (( nv = NV_MIN; nv <= NV_MAX; nv++ )); do
        echo -e "${CYAN}──────────────────────────────────────────────────${NC}"
        echo -e "${CYAN}  nv = ${nv}  (constraints = 2^${nv} = $(( 1 << nv )))${NC}"
        echo -e "${CYAN}──────────────────────────────────────────────────${NC}"

        # Accumulators — sum across M runs (not averaged) for direct comparison
        total_setup_ms=0
        total_prover_ms=0
        total_verifier_ms=0
        total_proof_bytes=0
        total_comm_sent=0
        total_comm_recv=0
        total_cpu_ms=0
        total_wall_ms=0
        max_peak_rss_mb=0
        successful_reps=0

        for (( rep = 1; rep <= REPS; rep++ )); do
            echo -e "  ${YELLOW}Repetition ${rep}/${REPS}...${NC}"

            # Clean up any leftover processes
            cleanup

            # Start workers (parties 1..K-1) in background
            WORKER_PIDS=()
            for (( i = 1; i < K; i++ )); do
                RAYON_NUM_THREADS=$THREADS_PER_PROVER \
                "$BIN" $i "$HOSTS_FILE" $nv \
                    > "$LOG_DIR/hp_p${i}_nv${nv}_rep${rep}.log" 2>&1 &
                WORKER_PIDS+=($!)
            done

            # Give workers time to bind (skip for k=1)
            if (( K > 1 )); then
                sleep 2
            fi

            # Run master (party 0) — binary outputs CSV on stdout, debug on stderr
            MASTER_RAW="$LOG_DIR/hp_p0_nv${nv}_rep${rep}_raw.log"
            MASTER_STDERR="$LOG_DIR/hp_p0_nv${nv}_rep${rep}.log"
            RAYON_NUM_THREADS=$THREADS_PER_PROVER \
                "$BIN" 0 "$HOSTS_FILE" $nv \
                > "$MASTER_RAW" 2> "$MASTER_STDERR"

            # Parse this repetition's CSV output from binary
            # Format: setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,cpu_ms,wall_ms,peak_rss_mb
            if [[ -s "$MASTER_RAW" ]]; then
                PARSED=$(parse_hp_csv "$MASTER_RAW")
                IFS=',' read -r s_ms p_ms v_ms p_bytes c_sent c_recv cpu_ms wall_ms rss_mb <<< "$PARSED"

                avg_cpu=$(awk "BEGIN{printf \"%.0f\", ($wall_ms > 0) ? $cpu_ms / $wall_ms * 100 : 0}")
                echo -e "  ${GREEN}setup=${s_ms}ms  prove=${p_ms}ms  verify=${v_ms}ms  proof=${p_bytes}B  sent=${c_sent}B  recv=${c_recv}B  cpu=${avg_cpu}%  rss=${rss_mb}MB${NC}"

                # Accumulate for averaging (prover/verifier/comm are averaged over M;
                # setup taken from rep 1; peak_rss is max)
                if (( rep == 1 )); then
                    total_setup_ms="$s_ms"
                fi
                total_prover_ms=$(awk "BEGIN{printf \"%.3f\", $total_prover_ms + $p_ms}")
                total_verifier_ms=$(awk "BEGIN{printf \"%.3f\", $total_verifier_ms + $v_ms}")
                total_proof_bytes=$(( total_proof_bytes + p_bytes ))
                total_comm_sent=$(( total_comm_sent + c_sent ))
                total_comm_recv=$(( total_comm_recv + c_recv ))
                total_cpu_ms=$(awk "BEGIN{printf \"%.3f\", $total_cpu_ms + $cpu_ms}")
                total_wall_ms=$(awk "BEGIN{printf \"%.3f\", $total_wall_ms + $wall_ms}")
                cur_rss=$(awk "BEGIN{print ($rss_mb > $max_peak_rss_mb) ? $rss_mb : $max_peak_rss_mb}")
                max_peak_rss_mb="$cur_rss"
                successful_reps=$(( successful_reps + 1 ))
            else
                echo -e "  ${RED}No output from master${NC}"
            fi

            # Wait for workers
            for pid in "${WORKER_PIDS[@]}"; do
                wait $pid 2>/dev/null || true
            done
        done

        # Average over successful reps (M=1 always for HyperPianist)
        R=${successful_reps:-1}
        avg_prover_ms=$(awk "BEGIN{printf \"%.3f\", $total_prover_ms / $R}")
        avg_verifier_ms=$(awk "BEGIN{printf \"%.3f\", $total_verifier_ms / $R}")
        avg_proof_bytes=$(( total_proof_bytes / R ))
        avg_comm_sent=$(( total_comm_sent / R ))
        avg_comm_recv=$(( total_comm_recv / R ))
        avg_cpu_pct=$(awk "BEGIN{printf \"%.1f\", ($total_wall_ms > 0) ? $total_cpu_ms / $total_wall_ms * 100 : 0}")
        CSV_LINE="${nv},1,${K},${total_setup_ms},${avg_prover_ms},${avg_verifier_ms},${avg_proof_bytes},${avg_comm_sent},${avg_comm_recv},${avg_cpu_pct},${max_peak_rss_mb}"
        echo "$CSV_LINE" | tee -a "$CSV_FILE"
    done

    echo ""
    echo -e "${GREEN}  Results for k=${K}: ${CSV_FILE}${NC}"
    echo ""

    # Return to HP_ROOT for next iteration
    cd "$HP_ROOT"
done

echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All benchmarks complete!${NC}"
echo -e "${GREEN}  Logs:    ${LOG_DIR}/hp_p*.log${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
