//! SumFold stage-level profiling benchmark.
//!
//! Diagnoses why distributed SumFold speedup is low (1.71x with K=4 parties)
//! by measuring:
//!   A) Single-node sum_fold_v3 (M=16, num_vars=20)  — the baseline
//!   B) Same sum_fold_v3       (M=16, num_vars=18)  — per-party compute in current d_sumfold
//!   C) Partitioned sum_fold_v3(M=4,  num_vars=18)  — if each party only folded M/K instances
//!
//! Also profiles the costly sub-stages: split, merge, compose, fix_variables.
//!
//! Run:
//!   cargo bench -p deSnark --bench sumfold_profile
//!
//! Or with restricted threads (matching production):
//!   RAYON_NUM_THREADS=2 cargo bench -p deSnark --bench sumfold_profile

use std::time::{Duration, Instant};

use ark_bls12_381::Fr;
use ark_ff::{PrimeField, Zero};
use ark_poly::DenseMultilinearExtension;
use ark_std::{log2, test_rng};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use arithmetic::{
    build_eq_x_r, eq_poly::EqPolynomial, fix_variables, VPAuxInfo, VirtualPolynomial,
};
use subroutines::poly_iop::{
    prelude::{PolyIOP, SumCheck},
    sum_check::{stage2_compute_sum_t, stage3_merge_split_mles, stage4_compose_poly},
};

#[allow(unused_imports)]
use ark_poly::MultilinearExtension;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Generate m random VirtualPolynomials with structure similar to HyperPlonk vanilla gate.
fn gen_polys(nv: usize, m: usize) -> (Vec<VirtualPolynomial<Fr>>, Vec<Fr>) {
    let mut rng = test_rng();
    let num_multiplicands = 3;
    let num_products = 2;

    let (template, _) = VirtualPolynomial::<Fr>::rand(
        nv,
        (num_multiplicands, num_multiplicands + 1),
        num_products,
        &mut rng,
    )
    .unwrap();

    let t = template.flattened_ml_extensions.len();
    let mut polys = Vec::with_capacity(m);
    let mut sums = Vec::with_capacity(m);

    for _ in 0..m {
        let mut new_mles: Vec<Arc<DenseMultilinearExtension<Fr>>> = Vec::with_capacity(t);
        let mut lut = HashMap::new();
        for _ in 0..t {
            let mle = Arc::new(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng));
            let ptr = Arc::as_ptr(&mle);
            lut.insert(ptr, new_mles.len());
            new_mles.push(mle);
        }
        let poly = VirtualPolynomial {
            aux_info: template.aux_info.clone(),
            products: template.products.clone(),
            flattened_ml_extensions: new_mles,
            raw_pointers_lookup_table: lut,
        };
        sums.push(Fr::zero());
        polys.push(poly);
    }
    (polys, sums)
}

fn print_header(title: &str) {
    println!();
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  {:<64} ║", title);
    println!("╠════════════════════════════════════════════════════════════════════╣");
}

fn print_footer() {
    println!("╚════════════════════════════════════════════════════════════════════╝");
}

fn print_row(label: &str, avg: Duration) {
    println!("║  {:<46} {:>14.3?}  ║", label, avg);
}

fn print_row_pct(label: &str, avg: Duration, total: Duration) {
    let pct = avg.as_secs_f64() / total.as_secs_f64() * 100.0;
    println!("║  {:<40} {:>10.3?} ({:>5.1}%)  ║", label, avg, pct);
}

fn print_sep() {
    println!("╟────────────────────────────────────────────────────────────────────╢");
}

/// Run f repeatedly, return median duration.
fn bench_median<F: FnMut() -> Duration>(mut f: F, iters: usize) -> Duration {
    let mut times: Vec<Duration> = (0..iters).map(|_| f()).collect();
    times.sort();
    times[iters / 2]
}

// ═══════════════════════════════════════════════════════════════════════════
// Bench 1: End-to-end sum_fold_v3 at three configurations
// ═══════════════════════════════════════════════════════════════════════════

fn bench_sumfold_e2e() {
    let threads = rayon::current_num_threads();
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "  SumFold E2E — {} threads (set RAYON_NUM_THREADS)",
        threads
    );
    println!("═══════════════════════════════════════════════════════════════════════");

    let configs: Vec<(&str, usize, usize, usize)> = vec![
        ("A: Single (M=16,nv=20)", 16, 20, 2),
        ("B: CurrDist (M=16,nv=18)", 16, 18, 3),
        ("C: Partition (M=4,nv=18)", 4, 18, 3),
    ];

    let mut results = Vec::new();

    for (label, m, nv, iters) in &configs {
        print_header(label);

        eprintln!("[{}] gen_polys (M={}, nv={}) ...", label, m, nv);
        let (polys, sums) = gen_polys(*nv, *m);
        let t = polys[0].flattened_ml_extensions.len();
        let length = log2(*m) as usize;
        let new_nv = length + nv;

        println!(
            "║  t={} MLEs, length={}, compose_nv={} (2^{} evals) ║",
            t, length, new_nv, new_nv
        );
        print_sep();

        // sum_fold_v3
        let dur_v3 = bench_median(
            || {
                let ps: Vec<_> = polys.iter().map(|p| p.deep_copy()).collect();
                let ss = sums.clone();
                let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                let start = Instant::now();
                let _ =
                    <PolyIOP<Fr> as SumCheck<Fr>>::sum_fold_v3(ps, ss, &mut transcript).unwrap();
                start.elapsed()
            },
            *iters,
        );

        print_row("sum_fold_v3 total", dur_v3);
        results.push((*label, dur_v3));
        print_footer();
    }

    // Summary
    print_header("Speedup Summary (compute only, no network)");
    let a_v3 = results[0].1.as_secs_f64();
    let b_v3 = results[1].1.as_secs_f64();
    let c_v3 = results[2].1.as_secs_f64();

    println!("║  A (single):    v3={:>10.3?}    ║", results[0].1);
    println!("║  B (curr dist): v3={:>10.3?}    ║", results[1].1);
    println!("║  C (partition): v3={:>10.3?}    ║", results[2].1);
    print_sep();
    println!(
        "║  v3 compute speedup A/B = {:.2}x                            ║",
        a_v3 / b_v3
    );
    println!(
        "║  v3 compute speedup A/C = {:.2}x                            ║",
        a_v3 / c_v3
    );
    print_footer();
}

// ═══════════════════════════════════════════════════════════════════════════
// Bench 2: Sub-stage profiling
// ═══════════════════════════════════════════════════════════════════════════

fn bench_substages() {
    let configs: Vec<(&str, usize, usize, usize)> = vec![
        ("B: CurrDist (M=16,nv=18)", 16, 18, 3),
        ("C: Partition (M=4,nv=18)", 4, 18, 5),
    ];

    for (label, m, nv, iters) in &configs {
        print_header(&format!("Sub-stages: {}", label));

        let (polys, sums) = gen_polys(*nv, *m);
        let t = polys[0].flattened_ml_extensions.len();
        let length = log2(*m) as usize;
        let new_nv = length + nv;

        println!(
            "║  M={}, nv={}, t={}, length={}, compose_nv={}              ║",
            m, nv, t, length, new_nv
        );
        print_sep();

        // Stage 2a: split_by_last_variables
        let dur_split = bench_median(
            || {
                let start = Instant::now();
                let _splits: Vec<Vec<VirtualPolynomial<Fr>>> = polys
                    .iter()
                    .map(|vp| vp.split_by_last_variables(length))
                    .collect();
                start.elapsed()
            },
            *iters,
        );

        // Prepare splits for later stages
        let all_splits: Vec<Vec<VirtualPolynomial<Fr>>> = polys
            .iter()
            .map(|vp| vp.split_by_last_variables(length))
            .collect();

        // Stage 2b: compute_sum_t
        let rho: Vec<Fr> = (0..length).map(|i| Fr::from(i as u64 + 1)).collect();
        let eq_xr_poly = build_eq_x_r(&rho).unwrap();
        let eq_xr_vec = eq_xr_poly.to_evaluations();

        let dur_sum_t = bench_median(
            || {
                let start = Instant::now();
                let _ = stage2_compute_sum_t(&sums, &eq_xr_vec);
                start.elapsed()
            },
            100,
        );

        // Stage 3: merge_split_mles
        let dur_merge = bench_median(
            || {
                let start = Instant::now();
                let _ = stage3_merge_split_mles(&all_splits, *m, t, new_nv);
                start.elapsed()
            },
            *iters,
        );

        // Prepare merged MLEs for stage 4
        let merged_mles = stage3_merge_split_mles(&all_splits, *m, t, new_nv);

        // Stage 4: compose_poly
        let dur_compose = bench_median(
            || {
                let start = Instant::now();
                let _ = stage4_compose_poly(
                    merged_mles.clone(),
                    polys[0].products.clone(),
                    polys[0].aux_info.max_degree + 1,
                    new_nv,
                );
                start.elapsed()
            },
            *iters,
        );

        // fix_variables on compose-sized MLEs (one round)
        let mut rng = test_rng();
        let big_mles: Vec<DenseMultilinearExtension<Fr>> = (0..t)
            .map(|_| DenseMultilinearExtension::<Fr>::rand(new_nv, &mut rng))
            .collect();
        let r = Fr::from(42u64);

        let dur_fix = bench_median(
            || {
                let start = Instant::now();
                let _: Vec<_> = big_mles
                    .par_iter()
                    .map(|mle| fix_variables(mle, &[r]))
                    .collect();
                start.elapsed()
            },
            *iters,
        );

        // Total (full sum_fold_v3)
        let dur_total = bench_median(
            || {
                let ps: Vec<_> = polys.iter().map(|p| p.deep_copy()).collect();
                let ss = sums.clone();
                let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                let start = Instant::now();
                let _ =
                    <PolyIOP<Fr> as SumCheck<Fr>>::sum_fold_v3(ps, ss, &mut transcript).unwrap();
                start.elapsed()
            },
            *iters,
        );

        // Print breakdown
        print_row_pct("split_by_last_variables", dur_split, dur_total);
        print_row_pct("compute_sum_t", dur_sum_t, dur_total);
        print_row_pct("merge_split_mles", dur_merge, dur_total);
        print_row_pct("compose_poly", dur_compose, dur_total);
        print_row_pct("fix_variables (per round)", dur_fix, dur_total);
        let fix_total = dur_fix * (length as u32);
        print_row_pct(
            &format!("fix_variables × {} rounds", length),
            fix_total,
            dur_total,
        );
        print_sep();

        let setup = dur_split + dur_sum_t + dur_merge + dur_compose;
        let rounds_compute = dur_total - setup;
        print_row_pct("Setup subtotal (split+merge+compose)", setup, dur_total);
        print_row_pct("Rounds compute (total - setup)", rounds_compute, dur_total);
        print_sep();
        print_row("TOTAL sum_fold_v3", dur_total);
        print_footer();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bench 3: fix_variables scaling
// ═══════════════════════════════════════════════════════════════════════════

fn bench_fix_variables_scaling() {
    print_header("fix_variables cost by compose MLE size");

    let mut rng = test_rng();
    let t = 5;

    for nv in [18, 20, 22, 24] {
        let mles: Vec<DenseMultilinearExtension<Fr>> = (0..t)
            .map(|_| DenseMultilinearExtension::<Fr>::rand(nv, &mut rng))
            .collect();
        let r = Fr::from(42u64);

        let iters = if nv <= 20 { 5 } else { 2 };
        let dur = bench_median(
            || {
                let start = Instant::now();
                let _: Vec<_> = mles
                    .par_iter()
                    .map(|mle| fix_variables(mle, &[r]))
                    .collect();
                start.elapsed()
            },
            iters,
        );

        let label = format!("t={} × 2^{} → 2^{}", t, nv, nv - 1);
        print_row(&label, dur);
    }

    print_footer();
}

// ═══════════════════════════════════════════════════════════════════════════
// Bench 4: Serialization cost (for network overhead estimation)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_serialization() {
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use subroutines::poly_iop::prelude::IOPProverMessage;

    print_header("Serialization cost (prover message per round)");

    // Typical prover message: max_degree + 1 evaluations
    for deg in [3, 4, 5] {
        let msg = IOPProverMessage {
            evaluations: (0..=deg).map(|i| Fr::from(i as u64 + 42)).collect(),
        };
        let mut buf = Vec::new();
        msg.serialize_uncompressed(&mut buf).unwrap();
        let bytes = buf.len();

        let dur = bench_median(
            || {
                let start = Instant::now();
                let mut b = Vec::new();
                msg.serialize_uncompressed(&mut b).unwrap();
                let _ = IOPProverMessage::<Fr>::deserialize_uncompressed_unchecked(&b[..]).unwrap();
                start.elapsed()
            },
            1000,
        );

        let label = format!("deg={}: {} bytes, ser+deser", deg, bytes);
        print_row(&label, dur);
    }

    print_sep();
    println!("║  Message ser/deser is sub-microsecond — negligible.          ║");
    println!("║  Network overhead = TCP roundtrip latency × 2 × 4 rounds    ║");
    print_footer();
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let threads = rayon::current_num_threads();
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  SumFold Profiling Suite — {} Rayon threads", threads);
    println!("═══════════════════════════════════════════════════════════════════════");

    // Quick micro-benchmarks
    bench_serialization();
    bench_fix_variables_scaling();

    // Sub-stage breakdown
    bench_substages();

    // Full E2E comparison (heaviest)
    bench_sumfold_e2e();

    // Final analysis
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Analysis");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Current d_sumfold (1.71x speedup with K=4):");
    println!("    Each party holds ALL M=16 instances, each with N/K=2^18 evals.");
    println!("    Compose poly: 2^(log2(M)+nv) = 2^22 evals per round.");
    println!("    Single node:  2^(log2(M)+N)  = 2^24 evals per round.");
    println!("    Theoretical max compute speedup = 2^24 / 2^22 = 4x.");
    println!();
    println!("  Proposed: Two-level SumFold (instance partitioning)");
    println!("    Level 1: Each party folds M/K=4 instances locally (2 rounds).");
    println!("             Compose poly: 2^(log2(4)+18) = 2^20 evals — 4× smaller.");
    println!("    Level 2: Master combines K=4 partial results (2 rounds).");
    println!("    Expected compute speedup ≈ 2^24 / 2^20 = 16x (!)");
    println!("    Even with network overhead, should comfortably exceed 4x.");
    println!();
}
