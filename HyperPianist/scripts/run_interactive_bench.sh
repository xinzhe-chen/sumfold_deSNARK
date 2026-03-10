#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Interactive Distributed Benchmark Runner — HyperPianist
#
# Single entry point for both vanilla and custom gate benchmarks.
#
# Parameters:
#   nMIN / nMAX  — range of n where nv = 2^n (num_vars)
#   k            — Number of Sub_Provers (comma-separated, each a power of 2)
#   reps         — Repetitions per nv for averaging
#   gate_preset  — vanilla, jellyfish_turbo, super_long_selector, mock
#
# Thread control:
#   RAYON_NUM_THREADS is set per prover to floor(total_cores / max_K),
#   so that every prover gets the SAME number of threads regardless of K.
#
# Output: CSV file saved to HyperPianist/target/bench_logs/
# ═══════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$HP_ROOT/target/bench_logs"
TMP_DIR="$HP_ROOT/target/bench_tmp"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

die() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

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
    echo "$log"
}

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

parse_hp_csv() {
    local raw_file="$1"
    grep '^[0-9]' "$raw_file" | head -1
}

is_valid_gate_preset() {
    case "$1" in
        vanilla|jellyfish_turbo|super_long_selector|mock) return 0 ;;
        *) return 1 ;;
    esac
}

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

echo -e "${CYAN}  Gate presets: vanilla, jellyfish_turbo, super_long_selector, mock${NC}"
read -p "  gate_preset (default: vanilla): " GATE_PRESET
GATE_PRESET=${GATE_PRESET:-vanilla}

MOCK_DEGREE=""
MOCK_NUM_WITNESS=""
if [[ "$GATE_PRESET" == "mock" ]]; then
    read -p "  mock degree (default 3): " MOCK_DEGREE
    MOCK_DEGREE=${MOCK_DEGREE:-3}
    read -p "  mock num_witness (default 4): " MOCK_NUM_WITNESS
    MOCK_NUM_WITNESS=${MOCK_NUM_WITNESS:-4}
fi
echo ""

[[ "$NV_MIN" =~ ^[0-9]+$ ]] || die "nMIN must be a positive integer"
[[ "$NV_MAX" =~ ^[0-9]+$ ]] || die "nMAX must be a positive integer"
[[ "$REPS" =~ ^[0-9]+$ ]] || die "reps must be a positive integer"
is_valid_gate_preset "$GATE_PRESET" || die "gate_preset must be one of: vanilla, jellyfish_turbo, super_long_selector, mock"

(( NV_MIN <= NV_MAX )) || die "nMIN ($NV_MIN) must be <= nMAX ($NV_MAX)"
(( REPS >= 1 )) || die "reps must be >= 1"

if [[ "$GATE_PRESET" == "mock" ]]; then
    [[ "$MOCK_DEGREE" =~ ^[0-9]+$ ]] || die "mock degree must be a positive integer"
    [[ "$MOCK_NUM_WITNESS" =~ ^[0-9]+$ ]] || die "mock num_witness must be a positive integer"
    (( MOCK_DEGREE >= 1 )) || die "mock degree must be >= 1"
    (( MOCK_NUM_WITNESS >= 1 )) || die "mock num_witness must be >= 1"
fi

IFS=',' read -ra K_VALUES <<< "$K_INPUT"
[[ ${#K_VALUES[@]} -gt 0 ]] || die "Must provide at least one k value"

for K in "${K_VALUES[@]}"; do
    K=$(echo "$K" | tr -d ' ')
    [[ "$K" =~ ^[0-9]+$ ]] || die "k value '$K' must be a positive integer"
    is_power_of_2 "$K" || die "k ($K) must be a power of 2"
done

MAX_K=0
for K in "${K_VALUES[@]}"; do
    K=$(echo "$K" | tr -d ' ')
    (( K > MAX_K )) && MAX_K=$K
done
USABLE_CORES=$(( TOTAL_CORES - 2 ))
(( USABLE_CORES >= 1 )) || USABLE_CORES=1
THREADS_PER_PROVER=$(( USABLE_CORES / MAX_K ))
(( THREADS_PER_PROVER >= 1 )) || die "max k=$MAX_K exceeds usable cores ($USABLE_CORES = $TOTAL_CORES - 2). Cannot assign >= 1 thread per prover."

USE_CUSTOM_GATE_BENCH=0
if [[ "$GATE_PRESET" != "vanilla" ]]; then
    USE_CUSTOM_GATE_BENCH=1
fi

if (( USE_CUSTOM_GATE_BENCH )); then
    BIN_NAME="hyperpianist-bench-custom-gate"
    CSV_HEADER="nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb,gate_name"
    LOG_PREFIX="hp_cg"
    if [[ "$GATE_PRESET" == "mock" ]]; then
        GATE_FILE_TAG="mock_w${MOCK_NUM_WITNESS}_d${MOCK_DEGREE}"
    else
        GATE_FILE_TAG="$GATE_PRESET"
    fi
    CSV_PREFIX="bench_cg_${GATE_FILE_TAG}"
else
    BIN_NAME="hyperpianist-bench"
    CSV_HEADER="nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb"
    LOG_PREFIX="hp"
    CSV_PREFIX="bench"
fi
BIN="$HP_ROOT/target/release/examples/$BIN_NAME"

echo -e "${GREEN}Parameters:${NC}"
echo -e "  nv range      : ${NV_MIN} .. ${NV_MAX}"
echo -e "  k values      : ${K_VALUES[*]}"
echo -e "  M (instances) : 1  (HyperPianist always proves 1 instance)"
echo -e "  reps          : ${REPS}"
echo -e "  gate preset   : ${GATE_PRESET}"
echo -e "  benchmark bin : ${BIN_NAME}"
echo -e "  Total cores   : ${TOTAL_CORES}  (reserving 2 for OS -> ${USABLE_CORES} usable)"
echo -e "  Threads/prover: ${THREADS_PER_PROVER}  (fixed by max k=${MAX_K}: ${MAX_K} x ${THREADS_PER_PROVER} = $(( MAX_K * THREADS_PER_PROVER )) <= ${USABLE_CORES} usable cores)"
echo ""

NIGHTLY_RUSTC="$(rustup which rustc --toolchain nightly-2026-02-22 2>/dev/null || true)"
if [[ -n "$NIGHTLY_RUSTC" ]]; then
    NIGHTLY_BIN="$(dirname "$NIGHTLY_RUSTC")"
    export PATH="$NIGHTLY_BIN:$PATH"
    export RUSTC="$NIGHTLY_RUSTC"
    echo -e "${GREEN}Using nightly rustc:${NC} $($NIGHTLY_RUSTC --version)"
else
    echo -e "${YELLOW}Warning: could not resolve nightly-2026-02-22 toolchain, using default${NC}"
fi

echo -e "${YELLOW}Building ${BIN_NAME} (release)...${NC}"
cd "$HP_ROOT"
cargo build --example "$BIN_NAME" --release
echo -e "${GREEN}Build complete.${NC}"
echo ""

for K in "${K_VALUES[@]}"; do
    K=$(echo "$K" | tr -d ' ')
    LOG_K=$(log2 "$K")

    if (( K > 1 )); then
        (( NV_MIN > LOG_K )) || {
            echo -e "${RED}Error: nMIN ($NV_MIN) must be > log2(k=$K) = $LOG_K, skipping k=$K${NC}"
            continue
        }
    fi

    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Running with k=${K} (log2=${LOG_K}), gate=${GATE_PRESET}${NC}"
    echo -e "${CYAN}  Threads per prover: ${THREADS_PER_PROVER} (fixed, ${K} provers x ${THREADS_PER_PROVER} threads = $(( K * THREADS_PER_PROVER )) used / ${TOTAL_CORES} cores)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"

    mkdir -p "$TMP_DIR"
    HOSTS_FILE="$TMP_DIR/hosts_${K}.txt"
    BASE_PORT=8000

    : > "$HOSTS_FILE"
    for (( i = 0; i < K; i++ )); do
        echo "127.0.0.1:$(( BASE_PORT + i ))" >> "$HOSTS_FILE"
    done
    echo -e "${GREEN}Generated hosts file:${NC} $HOSTS_FILE  (${K} parties on ports ${BASE_PORT}..$(( BASE_PORT + K - 1 )))"
    echo ""

    cleanup() {
        for (( p = 0; p < K; p++ )); do
            local port=$(( BASE_PORT + p ))
            lsof -ti:$port 2>/dev/null | xargs kill -9 2>/dev/null || true
        done
        pkill -f "hyperpianist-bench-custom-gate" 2>/dev/null || true
        pkill -f "hyperpianist-bench" 2>/dev/null || true
        sleep 1
    }

    mkdir -p "$LOG_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CSV_FILE="$LOG_DIR/${CSV_PREFIX}_nv${NV_MIN}_${NV_MAX}_k${K}_reps${REPS}_${TIMESTAMP}.csv"

    echo -e "${YELLOW}Starting benchmark: nv=${NV_MIN}..${NV_MAX}, K=${K}, M=1, reps=${REPS}, gate=${GATE_PRESET}, threads/prover=${THREADS_PER_PROVER}${NC}"
    echo ""

    echo "$CSV_HEADER" | tee "$CSV_FILE"

    cd "$TMP_DIR"

    for (( nv = NV_MIN; nv <= NV_MAX; nv++ )); do
        echo -e "${CYAN}──────────────────────────────────────────────────${NC}"
        echo -e "${CYAN}  nv = ${nv}  (constraints = 2^${nv} = $(( 1 << nv )))${NC}"
        echo -e "${CYAN}──────────────────────────────────────────────────${NC}"

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
        gate_name_csv="$GATE_PRESET"

        for (( rep = 1; rep <= REPS; rep++ )); do
            echo -e "  ${YELLOW}Repetition ${rep}/${REPS}...${NC}"

            cleanup

            WORKER_PIDS=()
            for (( i = 1; i < K; i++ )); do
                worker_cmd=( "$BIN" "$i" "$HOSTS_FILE" "$nv" )
                if (( USE_CUSTOM_GATE_BENCH )); then
                    worker_cmd+=( "$GATE_PRESET" )
                    if [[ "$GATE_PRESET" == "mock" ]]; then
                        worker_cmd+=( --mock-degree "$MOCK_DEGREE" --mock-num-witness "$MOCK_NUM_WITNESS" )
                    fi
                fi

                RAYON_NUM_THREADS=$THREADS_PER_PROVER \
                    "${worker_cmd[@]}" > "$LOG_DIR/${LOG_PREFIX}_p${i}_nv${nv}_rep${rep}.log" 2>&1 &
                WORKER_PIDS+=($!)
            done

            if (( K > 1 )); then
                sleep 2
            fi

            MASTER_RAW="$LOG_DIR/${LOG_PREFIX}_p0_nv${nv}_rep${rep}_raw.log"
            MASTER_STDERR="$LOG_DIR/${LOG_PREFIX}_p0_nv${nv}_rep${rep}.log"

            master_cmd=( "$BIN" 0 "$HOSTS_FILE" "$nv" )
            if (( USE_CUSTOM_GATE_BENCH )); then
                master_cmd+=( "$GATE_PRESET" )
                if [[ "$GATE_PRESET" == "mock" ]]; then
                    master_cmd+=( --mock-degree "$MOCK_DEGREE" --mock-num-witness "$MOCK_NUM_WITNESS" )
                fi
            fi

            RAYON_NUM_THREADS=$THREADS_PER_PROVER \
                "${master_cmd[@]}" > "$MASTER_RAW" 2> "$MASTER_STDERR"

            if [[ -s "$MASTER_RAW" ]]; then
                PARSED=$(parse_hp_csv "$MASTER_RAW")
                if (( USE_CUSTOM_GATE_BENCH )); then
                    IFS=',' read -r s_ms p_ms v_ms p_bytes c_sent c_recv cpu_ms wall_ms rss_mb g_name <<< "$PARSED"
                    gate_name_csv="$g_name"
                else
                    IFS=',' read -r s_ms p_ms v_ms p_bytes c_sent c_recv cpu_ms wall_ms rss_mb <<< "$PARSED"
                fi

                avg_cpu=$(awk "BEGIN{printf \"%.0f\", ($wall_ms > 0) ? $cpu_ms / $wall_ms * 100 : 0}")
                echo -e "  ${GREEN}setup=${s_ms}ms  prove=${p_ms}ms  verify=${v_ms}ms  proof=${p_bytes}B  sent=${c_sent}B  recv=${c_recv}B  cpu=${avg_cpu}%  rss=${rss_mb}MB${NC}"

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

            for pid in "${WORKER_PIDS[@]}"; do
                wait "$pid" 2>/dev/null || true
            done
        done

        if (( successful_reps == 0 )); then
            echo -e "${RED}No successful repetitions for nv=${nv}; skipping CSV row${NC}"
            continue
        fi

        R=$successful_reps
        avg_prover_ms=$(awk "BEGIN{printf \"%.3f\", $total_prover_ms / $R}")
        avg_verifier_ms=$(awk "BEGIN{printf \"%.3f\", $total_verifier_ms / $R}")
        avg_proof_bytes=$(( total_proof_bytes / R ))
        avg_comm_sent=$(( total_comm_sent / R ))
        avg_comm_recv=$(( total_comm_recv / R ))
        avg_cpu_pct=$(awk "BEGIN{printf \"%.1f\", ($total_wall_ms > 0) ? $total_cpu_ms / $total_wall_ms * 100 : 0}")

        if (( USE_CUSTOM_GATE_BENCH )); then
            CSV_LINE="${nv},1,${K},${total_setup_ms},${avg_prover_ms},${avg_verifier_ms},${avg_proof_bytes},${avg_comm_sent},${avg_comm_recv},${avg_cpu_pct},${max_peak_rss_mb},${gate_name_csv}"
        else
            CSV_LINE="${nv},1,${K},${total_setup_ms},${avg_prover_ms},${avg_verifier_ms},${avg_proof_bytes},${avg_comm_sent},${avg_comm_recv},${avg_cpu_pct},${max_peak_rss_mb}"
        fi
        echo "$CSV_LINE" | tee -a "$CSV_FILE"
    done

    echo ""
    echo -e "${GREEN}  Results for k=${K}: ${CSV_FILE}${NC}"
    echo ""

    cd "$HP_ROOT"
done

echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All benchmarks complete!${NC}"
echo -e "${GREEN}  Logs: ${LOG_DIR}${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
