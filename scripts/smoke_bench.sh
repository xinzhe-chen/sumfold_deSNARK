#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUMFOLD_TMP="$ROOT/target/smoke/sumfold"
SUMFOLD_HEADER="nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb,d_commit_ms,sumfold_ms,sumcheck_ms,fold_ms,multi_open_ms"

mkdir -p "$SUMFOLD_TMP"

hosts_file="$SUMFOLD_TMP/hosts_1.txt"
config_file="$SUMFOLD_TMP/config.toml"
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
    exit 1
}

[[ -n "$row" ]] || {
    echo "sumfold smoke did not emit a data row" >&2
    exit 1
}

echo "benchmark smoke checks passed"
