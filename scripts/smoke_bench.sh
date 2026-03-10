#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUMFOLD_TMP="$ROOT/target/smoke/sumfold"
HP_ROOT="$ROOT/HyperPianist"
HP_TMP="$HP_ROOT/target/smoke"

SUMFOLD_HEADER="nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb,d_commit_ms,sumfold_ms,sumcheck_ms,fold_ms,multi_open_ms"
SUMFOLD_CG_HEADER="nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb,d_commit_ms,sumfold_ms,sumcheck_ms,fold_ms,multi_open_ms,gate_name,gate_degree,num_witnesses,num_selectors"
HP_HEADER="nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb"

run_sumfold_smoke() {
    mkdir -p "$SUMFOLD_TMP"

    local hosts_file="$SUMFOLD_TMP/hosts_1.txt"
    local config_file="$SUMFOLD_TMP/config.toml"
    printf '127.0.0.1:12350\n' > "$hosts_file"
    cat > "$config_file" <<EOF
[config]
log_num_instances = 0
log_num_constraints = 4
gate_type = "vanilla"
log_num_parties = 0
srs_path = "$SUMFOLD_TMP/srs_smoke.params"

[network]
hosts_file = "$hosts_file"
EOF

    local output header row
    output="$(
        cd "$ROOT" && \
        RAYON_NUM_THREADS=1 cargo run --quiet --release --example dist_bench -p deSnark -- \
            --party 0 --nv-min 4 --nv-max 4 --reps 1 "$config_file"
    )"
    header="$(printf '%s\n' "$output" | grep '^nv,' | head -n 1)"
    row="$(printf '%s\n' "$output" | grep '^[0-9]' | head -n 1)"

    [[ "$header" == "$SUMFOLD_HEADER" ]] || {
        echo "sumfold smoke header mismatch" >&2
        echo "expected: $SUMFOLD_HEADER" >&2
        echo "actual:   $header" >&2
        return 1
    }
    [[ -n "$row" ]] || {
        echo "sumfold smoke did not emit a data row" >&2
        return 1
    }
}

run_hyperpianist_smoke() {
    mkdir -p "$HP_TMP"

    local hosts_file="$HP_TMP/hosts_1.txt"
    local stdout_file="$HP_TMP/hyperpianist.stdout"
    local stderr_file="$HP_TMP/hyperpianist.stderr"
    printf '127.0.0.1:18000\n' > "$hosts_file"

    (
        cd "$HP_TMP"
        RAYON_NUM_THREADS=1 cargo run --quiet --manifest-path "$HP_ROOT/Cargo.toml" --release --example hyperpianist-bench -- \
            0 "$hosts_file" 4 > "$stdout_file" 2> "$stderr_file"
    )

    local metrics row
    metrics="$(grep '^[0-9]' "$stdout_file" | head -n 1)"
    [[ -n "$metrics" ]] || {
        echo "hyperpianist smoke did not emit metric output" >&2
        cat "$stderr_file" >&2
        return 1
    }
    IFS=',' read -r setup_ms prover_ms verifier_ms proof_bytes comm_sent comm_recv cpu_ms wall_ms peak_rss_mb <<< "$metrics"
    local avg_cpu
    avg_cpu="$(python3 - "$cpu_ms" "$wall_ms" <<'PY'
import sys

cpu_ms = float(sys.argv[1])
wall_ms = float(sys.argv[2])
value = 0.0 if wall_ms == 0 else cpu_ms / wall_ms * 100.0
print(f"{value:.1f}")
PY
)"
    row="4,1,1,$setup_ms,$prover_ms,$verifier_ms,$proof_bytes,$comm_sent,$comm_recv,$avg_cpu,$peak_rss_mb"

    python3 - "$HP_HEADER" "$row" <<'PY'
import csv
import io
import sys

header = sys.argv[1]
row = sys.argv[2]
field_count = len(next(csv.reader([header])))
row_count = len(next(csv.reader([row])))
if field_count != row_count:
    raise SystemExit(f"hyperpianist smoke column mismatch: expected {field_count}, got {row_count}")
PY
}

run_sumfold_custom_gate_smoke() {
    mkdir -p "$SUMFOLD_TMP"

    local hosts_file="$SUMFOLD_TMP/hosts_1.txt"
    local config_file="$SUMFOLD_TMP/config_cg.toml"
    printf '127.0.0.1:12351\n' > "$hosts_file"
    cat > "$config_file" <<EOF
[config]
log_num_instances = 0
log_num_constraints = 4
gate_type = "vanilla"
log_num_parties = 0
srs_path = "$SUMFOLD_TMP/srs_smoke_cg.params"

[network]
hosts_file = "$hosts_file"
EOF

    local output header row
    output="$(
        cd "$ROOT" && \
        RAYON_NUM_THREADS=1 cargo run --quiet --release --example dist_bench_custom_gate -p deSnark -- \
            --party 0 --nv-min 4 --nv-max 4 --reps 1 --gate-preset jellyfish_turbo "$config_file"
    )"
    header="$(printf '%s\n' "$output" | grep '^nv,' | head -n 1)"
    row="$(printf '%s\n' "$output" | grep '^[0-9]' | head -n 1)"

    [[ "$header" == "$SUMFOLD_CG_HEADER" ]] || {
        echo "sumfold custom-gate smoke header mismatch" >&2
        echo "expected: $SUMFOLD_CG_HEADER" >&2
        echo "actual:   $header" >&2
        return 1
    }
    [[ -n "$row" ]] || {
        echo "sumfold custom-gate smoke did not emit a data row" >&2
        return 1
    }
}

run_hyperpianist_custom_gate_smoke() {
    mkdir -p "$HP_TMP"

    local hosts_file="$HP_TMP/hosts_1_cg.txt"
    local stdout_file="$HP_TMP/hyperpianist_cg.stdout"
    local stderr_file="$HP_TMP/hyperpianist_cg.stderr"
    printf '127.0.0.1:18001\n' > "$hosts_file"

    (
        cd "$HP_TMP"
        RAYON_NUM_THREADS=1 cargo run --quiet --manifest-path "$HP_ROOT/Cargo.toml" --release --example hyperpianist-bench-custom-gate -- \
            0 "$hosts_file" 4 jellyfish_turbo > "$stdout_file" 2> "$stderr_file"
    )

    local metrics
    metrics="$(grep '^[0-9]' "$stdout_file" | head -n 1)"
    [[ -n "$metrics" ]] || {
        echo "hyperpianist custom-gate smoke did not emit metric output" >&2
        cat "$stderr_file" >&2
        return 1
    }
    # Verify output has the expected number of fields (9 base + 1 gate_name = 10)
    local field_count
    field_count="$(echo "$metrics" | awk -F',' '{print NF}')"
    [[ "$field_count" -eq 10 ]] || {
        echo "hyperpianist custom-gate smoke: expected 10 fields, got $field_count" >&2
        return 1
    }
}

run_sumfold_smoke
run_hyperpianist_smoke
run_sumfold_custom_gate_smoke
run_hyperpianist_custom_gate_smoke
echo "benchmark smoke checks passed"
