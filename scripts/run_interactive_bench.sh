#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Interactive Distributed Benchmark Runner — sumfold_deSNARK
#
# Single entry point for both vanilla and custom gate benchmarks.
#
# Parameters:
#   nMIN / nMAX  — range of n where nv = 2^n (log_num_constraints)
#   k            — Number of Sub_Provers (comma-separated, each a power of 2)
#   M            — Number of instances (must be a power of 2)
#   reps         — Repetitions per nv for averaging
#   gate_preset  — vanilla, jellyfish_turbo, super_long_selector, mock
#
# Thread control:
#   RAYON_NUM_THREADS is set per prover to floor(total_cores / max_K),
#   so that every prover gets the SAME number of threads regardless of K.
#
# Output: CSV file saved to target/bench_logs/
# ═══════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT/target/bench_logs"
TMP_DIR="$ROOT/target/bench_tmp"

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

is_valid_gate_preset() {
    case "$1" in
        vanilla|jellyfish_turbo|super_long_selector|mock) return 0 ;;
        *) return 1 ;;
    esac
}

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
read -p "  M    (Number of instances, power of 2): " M
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
[[ "$M" =~ ^[0-9]+$ ]] || die "M must be a positive integer"
[[ "$REPS" =~ ^[0-9]+$ ]] || die "reps must be a positive integer"
is_valid_gate_preset "$GATE_PRESET" || die "gate_preset must be one of: vanilla, jellyfish_turbo, super_long_selector, mock"

(( NV_MIN <= NV_MAX )) || die "nMIN ($NV_MIN) must be <= nMAX ($NV_MAX)"
is_power_of_2 "$M" || die "M ($M) must be a power of 2"
(( REPS >= 1 )) || die "reps must be >= 1"

if [[ "$GATE_PRESET" == "mock" ]]; then
    [[ "$MOCK_DEGREE" =~ ^[0-9]+$ ]] || die "mock degree must be a positive integer"
    [[ "$MOCK_NUM_WITNESS" =~ ^[0-9]+$ ]] || die "mock num_witness must be a positive integer"
    (( MOCK_DEGREE >= 1 )) || die "mock degree must be >= 1"
    (( MOCK_NUM_WITNESS >= 1 )) || die "mock num_witness must be >= 1"
fi

LOG_M=$(log2 "$M")

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
    BIN_NAME="dist_bench_custom_gate"
    BIN="$ROOT/target/release/examples/$BIN_NAME"
    SRS_PATH="srs_custom_gate_nv${NV_MAX}.params"
    PARTY_LOG_PREFIX="cg_p"
    if [[ "$GATE_PRESET" == "mock" ]]; then
        GATE_FILE_TAG="mock_w${MOCK_NUM_WITNESS}_d${MOCK_DEGREE}"
    else
        GATE_FILE_TAG="$GATE_PRESET"
    fi
    CSV_PREFIX="bench_cg_${GATE_FILE_TAG}"
else
    BIN_NAME="dist_bench"
    BIN="$ROOT/target/release/examples/$BIN_NAME"
    SRS_PATH="srs_interactive_nv${NV_MAX}.params"
    PARTY_LOG_PREFIX="p"
    CSV_PREFIX="bench"
fi

echo -e "${GREEN}Parameters:${NC}"
echo -e "  nv range      : ${NV_MIN} .. ${NV_MAX}"
echo -e "  k values      : ${K_VALUES[*]}"
echo -e "  M (instances) : ${M}  (log2 = ${LOG_M})"
echo -e "  reps          : ${REPS}"
echo -e "  gate preset   : ${GATE_PRESET}"
echo -e "  benchmark bin : ${BIN_NAME}"
echo -e "  Total cores   : ${TOTAL_CORES}  (reserving 2 for OS -> ${USABLE_CORES} usable)"
echo -e "  Threads/prover: ${THREADS_PER_PROVER}  (fixed by max k=${MAX_K}: ${MAX_K} x ${THREADS_PER_PROVER} = $(( MAX_K * THREADS_PER_PROVER )) <= ${USABLE_CORES} usable cores)"
if (( M > 1 )) && [[ "$GATE_PRESET" == "super_long_selector" || "$GATE_PRESET" == "mock" ]]; then
    echo -e "${YELLOW}  Warning       : ${GATE_PRESET} may fail for M > 1 because shared-selector mode requires a solvable output term.${NC}"
fi
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
cd "$ROOT"
cargo build --example "$BIN_NAME" -p deSnark --release
echo -e "${GREEN}Build complete.${NC}"
echo ""

for K in "${K_VALUES[@]}"; do
    K=$(echo "$K" | tr -d ' ')
    LOG_K=$(log2 "$K")

    if (( K > 1 )); then
        (( NV_MIN > LOG_K )) || {
            echo -e "${RED}Error: nMIN ($NV_MIN) must be > log2(k=$K) = $LOG_K for K>1, skipping k=$K${NC}"
            continue
        }
    fi

    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Running with k=${K} (log2=${LOG_K}), gate=${GATE_PRESET}${NC}"
    echo -e "${CYAN}  Threads per prover: ${THREADS_PER_PROVER} (fixed, ${K} provers x ${THREADS_PER_PROVER} threads = $(( K * THREADS_PER_PROVER )) used / ${TOTAL_CORES} cores)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"

    mkdir -p "$TMP_DIR"
    HOSTS_FILE="$TMP_DIR/hosts_${K}.txt"
    BASE_PORT=12350

    : > "$HOSTS_FILE"
    for (( i = 0; i < K; i++ )); do
        echo "127.0.0.1:$(( BASE_PORT + i ))" >> "$HOSTS_FILE"
    done
    echo -e "${GREEN}Generated hosts file:${NC} $HOSTS_FILE  (${K} parties on ports ${BASE_PORT}..$(( BASE_PORT + K - 1 )))"

    CONFIG_FILE="$TMP_DIR/bench_config.toml"
    if [[ "$GATE_PRESET" == "mock" ]]; then
        cat > "$CONFIG_FILE" <<EOF
# Auto-generated benchmark config
# nv range: ${NV_MIN}..${NV_MAX}, M=${M}, K=${K}, gate=${GATE_PRESET}

[config]
log_num_instances = ${LOG_M}
log_num_constraints = ${NV_MIN}
log_num_parties = ${LOG_K}
srs_path = "${SRS_PATH}"

[config.gate_type.mock]
num_witness = ${MOCK_NUM_WITNESS}
degree = ${MOCK_DEGREE}

[network]
hosts_file = "${HOSTS_FILE}"
EOF
    else
        cat > "$CONFIG_FILE" <<EOF
# Auto-generated benchmark config
# nv range: ${NV_MIN}..${NV_MAX}, M=${M}, K=${K}, gate=${GATE_PRESET}

[config]
log_num_instances = ${LOG_M}
log_num_constraints = ${NV_MIN}
gate_type = "${GATE_PRESET}"
log_num_parties = ${LOG_K}
srs_path = "${SRS_PATH}"

[network]
hosts_file = "${HOSTS_FILE}"
EOF
    fi
    echo -e "${GREEN}Generated TOML config:${NC} $CONFIG_FILE"

    for (( p = 0; p < K; p++ )); do
        port=$(( BASE_PORT + p ))
        lsof -ti:$port 2>/dev/null | xargs kill -9 2>/dev/null || true
    done
    pkill -f "dist_bench_custom_gate" 2>/dev/null || true
    pkill -f "dist_bench" 2>/dev/null || true
    sleep 1

    mkdir -p "$LOG_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CSV_FILE="$LOG_DIR/${CSV_PREFIX}_nv${NV_MIN}_${NV_MAX}_k${K}_M${M}_${TIMESTAMP}.csv"

    echo -e "${YELLOW}Starting benchmark: nv=${NV_MIN}..${NV_MAX}, K=${K}, M=${M}, gate=${GATE_PRESET}, threads/prover=${THREADS_PER_PROVER}${NC}"
    echo ""

    cd "$TMP_DIR"

    WORKER_PIDS=()
    for (( i = 1; i < K; i++ )); do
        worker_cmd=(
            "$BIN"
            --party "$i"
            --nv-min "$NV_MIN"
            --nv-max "$NV_MAX"
            --reps "$REPS"
        )
        if (( USE_CUSTOM_GATE_BENCH )); then
            worker_cmd+=( --gate-preset "$GATE_PRESET" )
            if [[ "$GATE_PRESET" == "mock" ]]; then
                worker_cmd+=( --mock-degree "$MOCK_DEGREE" --mock-num-witness "$MOCK_NUM_WITNESS" )
            fi
        fi
        worker_cmd+=( "$CONFIG_FILE" )

        RAYON_NUM_THREADS=$THREADS_PER_PROVER \
            "${worker_cmd[@]}" > "$LOG_DIR/${PARTY_LOG_PREFIX}${i}.log" 2>&1 &
        WORKER_PIDS+=($!)
    done

    if (( K > 1 )); then
        sleep 2
    fi

    echo -e "${GREEN}Running master (party 0) with RAYON_NUM_THREADS=${THREADS_PER_PROVER}...${NC}"
    master_cmd=(
        "$BIN"
        --party 0
        --nv-min "$NV_MIN"
        --nv-max "$NV_MAX"
        --reps "$REPS"
    )
    if (( USE_CUSTOM_GATE_BENCH )); then
        master_cmd+=( --gate-preset "$GATE_PRESET" )
        if [[ "$GATE_PRESET" == "mock" ]]; then
            master_cmd+=( --mock-degree "$MOCK_DEGREE" --mock-num-witness "$MOCK_NUM_WITNESS" )
        fi
    fi
    master_cmd+=( "$CONFIG_FILE" )

    RAYON_NUM_THREADS=$THREADS_PER_PROVER \
        "${master_cmd[@]}" 2> "$LOG_DIR/${PARTY_LOG_PREFIX}0.log" | tee "$CSV_FILE"

    for pid in "${WORKER_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    echo ""
    echo -e "${GREEN}  Results for k=${K}: ${CSV_FILE}${NC}"
    echo ""

    cd "$ROOT"
done

echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All benchmarks complete!${NC}"
echo -e "${GREEN}  Logs: ${LOG_DIR}${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
