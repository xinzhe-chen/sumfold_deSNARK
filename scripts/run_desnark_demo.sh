#!/bin/bash
# deSnark Distributed Prove Demo — 4 nodes
#
# Usage:
#   ./scripts/run_desnark_demo.sh [command]
#     build     — build the example binary
#     run       — run 4 nodes, print summary (default)
#     tmux      — run 4 nodes in tmux 2x2 grid (sync-scrollable)
#     logs      — show logs from previous run
#     clean     — kill processes and remove logs

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOSTS_FILE="$PROJECT_ROOT/deSnark/examples/hosts_4.txt"
CONFIG_FILE="${DESNARK_CONFIG:-$PROJECT_ROOT/deSnark/examples/demo_config.toml}"
BINARY="$PROJECT_ROOT/target/release/examples/dist_prove_demo"
LOG_DIR="$PROJECT_ROOT/target/desnark_demo_logs"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}   deSnark Distributed Prove Demo (4 nodes)${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════${NC}"
}

do_build() {
    echo -e "${YELLOW}Building dist_prove_demo (release)...${NC}"
    cd "$PROJECT_ROOT"
    cargo build --example dist_prove_demo -p deSnark --release
    echo -e "${GREEN}Build complete: $BINARY${NC}"
}

kill_existing() {
    echo -e "${YELLOW}Cleaning up existing processes...${NC}"
    for port in 12350 12351 12352 12353; do
        lsof -ti:$port 2>/dev/null | xargs kill -9 2>/dev/null || true
    done
    pkill -f "dist_prove_demo" 2>/dev/null || true
    sleep 1
}

setup_logs() {
    mkdir -p "$LOG_DIR"
    for i in 0 1 2 3; do
        > "$LOG_DIR/party$i.log"
    done
}

check_binary() {
    if [[ ! -f "$BINARY" ]]; then
        echo -e "${RED}Binary not found. Run '$0 build' first.${NC}"
        exit 1
    fi
}

start_nodes() {
    local LOG_LEVEL="${RUST_LOG:-info}"
    # Workers first (parties 1-3)
    for i in 1 2 3; do
        echo -e "  ${GREEN}→${NC} Party $i (worker)"
        RUST_LOG="$LOG_LEVEL" "$BINARY" --party $i "$CONFIG_FILE" > "$LOG_DIR/party$i.log" 2>&1 &
        eval "PID$i=$!"
    done
    sleep 2

    # Master last (party 0)
    echo -e "  ${GREEN}→${NC} Party 0 (master)"
    RUST_LOG="$LOG_LEVEL" "$BINARY" --party 0 "$CONFIG_FILE" > "$LOG_DIR/party0.log" 2>&1 &
    PID0=$!
}

wait_all() {
    echo -e "${YELLOW}Waiting for completion...${NC}"
    wait $PID0 2>/dev/null || true
    sleep 2
    kill $PID1 $PID2 $PID3 2>/dev/null || true
}

show_summary() {
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}   Results${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════${NC}"
    for i in 0 1 2 3; do
        echo ""
        if [[ $i -eq 0 ]]; then
            echo -e "${GREEN}─── Party $i (Master) ───${NC}"
        else
            echo -e "${BLUE}─── Party $i (Worker) ───${NC}"
        fi
        if [[ -s "$LOG_DIR/party$i.log" ]]; then
            cat "$LOG_DIR/party$i.log"
        else
            echo "(empty log)"
        fi
    done
    echo ""
    echo -e "Logs: ${YELLOW}$LOG_DIR/${NC}"
}

do_run() {
    check_binary
    kill_existing
    setup_logs
    echo -e "${BLUE}Starting 4 nodes...${NC}"
    start_nodes
    wait_all
    show_summary
}

do_tmux() {
    check_binary
    if ! command -v tmux &>/dev/null; then
        echo -e "${RED}tmux not found. Install with: brew install tmux${NC}"
        exit 1
    fi

    kill_existing
    setup_logs

    local SESSION="desnark"
    tmux kill-session -t "$SESSION" 2>/dev/null || true

    echo -e "${BLUE}Starting 4 nodes...${NC}"
    start_nodes

    # Create tmux session with 4 panes
    # Use less -R +F: follows file like tail -f, Ctrl-C to pause & scroll, F to resume
    # -R preserves ANSI colors; +F starts in follow mode
    local COLS=$(tput cols)
    local ROWS=$(tput lines)
    tmux new-session -d -s "$SESSION" -x "$COLS" -y "$ROWS" \
        "less -R +F $LOG_DIR/party0.log"
    tmux split-window -t "$SESSION" -h \
        "less -R +F $LOG_DIR/party1.log"
    tmux split-window -t "$SESSION:0.0" -v \
        "less -R +F $LOG_DIR/party2.log"
    tmux split-window -t "$SESSION:0.1" -v \
        "less -R +F $LOG_DIR/party3.log"

    # Equal 2x2 grid
    tmux select-layout -t "$SESSION" tiled

    # Pane titles
    tmux select-pane -t "$SESSION:0.0" -T "Party 0 (Master)"
    tmux select-pane -t "$SESSION:0.1" -T "Party 1 (Worker)"
    tmux select-pane -t "$SESSION:0.2" -T "Party 2 (Worker)"
    tmux select-pane -t "$SESSION:0.3" -T "Party 3 (Worker)"
    tmux set-option -t "$SESSION" pane-border-status top
    tmux set-option -t "$SESSION" pane-border-format " #{pane_title} "

    # Enable synchronize-panes: keystrokes go to ALL panes (including less)
    tmux set-window-option -t "$SESSION" synchronize-panes on

    echo -e "${GREEN}tmux session '${SESSION}' created (2x2 equal grid)${NC}"
    echo ""
    echo -e "${YELLOW}Keybindings (synced across all 4 panes):${NC}"
    echo -e "  ${CYAN}Ctrl-C${NC}     Pause live follow → enter scroll mode"
    echo -e "  ${CYAN}↑/↓${NC}        Scroll up/down (all panes sync)"
    echo -e "  ${CYAN}PgUp/PgDn${NC}  Scroll page up/down"
    echo -e "  ${CYAN}g / G${NC}      Jump to top / bottom"
    echo -e "  ${CYAN}F${NC}          Resume live follow mode"
    echo -e "  ${CYAN}q${NC}          Quit less (closes pane)"
    echo -e "  ${CYAN}Ctrl-b d${NC}   Detach tmux (nodes keep running)"
    echo ""

    tmux attach-session -t "$SESSION"

    # Cleanup after detach/exit
    kill_existing
    tmux kill-session -t "$SESSION" 2>/dev/null || true
}

do_logs() {
    for i in 0 1 2 3; do
        echo -e "${BLUE}─── Party $i ───${NC}"
        [[ -f "$LOG_DIR/party$i.log" ]] && cat "$LOG_DIR/party$i.log" || echo "(no log)"
        echo ""
    done
}

do_clean() {
    kill_existing
    rm -rf "$LOG_DIR"
    echo -e "${GREEN}Cleaned${NC}"
}

print_header

case "${1:-run}" in
    build)     do_build ;;
    run)       do_run ;;
    tmux)      do_tmux ;;
    logs)      do_logs ;;
    clean)     do_clean ;;
    -h|--help) echo "Usage: $0 {build|run|tmux|logs|clean}" ;;
    *)         echo -e "${RED}Unknown: $1${NC}"; exit 1 ;;
esac
