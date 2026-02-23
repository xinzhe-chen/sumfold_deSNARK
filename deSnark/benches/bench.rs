//! deSNARK single-machine benchmarks.
//!
//! Measures each stage of the proving pipeline and key sub-operations
//! to establish baselines for optimization.
//!
//! Run:
//!   cargo bench -p deSnark
//!
//! Or with print-trace timing:
//!   cargo bench -p deSnark --features print-trace

use std::time::{Duration, Instant};

use ark_bn254::{Bn254, Fr};
use ark_ff::{PrimeField, One, Zero};
use ark_poly::DenseMultilinearExtension;
use ark_std::test_rng;
use std::collections::HashMap;
use std::sync::Arc;

#[allow(unused_imports)]
use ark_poly::MultilinearExtension;

use arithmetic::{fix_variables, VirtualPolynomial};
use deSnark::snark::{
    circuits_to_sumcheck, make_circuit, merge_and_verify_sumfold, prove_sumfold, setup,
};
use deSnark::structs::{Config, GateType};
use subroutines::pcs::prelude::MultilinearKzgPCS;
use subroutines::poly_iop::prelude::{PolyIOP, SumCheck};

type PCS = MultilinearKzgPCS<Bn254>;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Run `f` for `iters` iterations, return average duration.
fn bench_avg<F: FnMut()>(mut f: F, iters: u32) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed() / iters
}

/// Print a table row.
fn row(label: &str, dur: Duration) {
    println!("  {:<50} {:>12.3?}", label, dur);
}

/// Generate m random VirtualPolynomials with identical structure.
fn gen_polys(
    nv: usize,
    m: usize,
    num_multiplicands: usize,
    num_products: usize,
) -> (Vec<VirtualPolynomial<Fr>>, Vec<Fr>) {
    let mut rng = test_rng();
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
        // Compute actual sum over boolean hypercube (needed for SumFold correctness)
        let s = poly.evaluate_over_boolean_hypercube();
        sums.push(s);
        polys.push(poly);
    }
    (polys, sums)
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmarks
// ═══════════════════════════════════════════════════════════════════════════

/// 1. build_partitioned_circuits: sequential vs parallel
fn bench_build_circuits() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Bench: build_partitioned_circuits (seq vs par)                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    let configs = [
        (3, 14, 2), // M=8,  N=2^14, K=4
        (3, 16, 2), // M=8,  N=2^16, K=4
        (4, 16, 2), // M=16, N=2^16, K=4
        (3, 18, 2), // M=8,  N=2^18, K=4
    ];

    for (log_inst, log_cons, log_parties) in configs {
        let config = Config::new(log_inst, log_cons, GateType::Vanilla, log_parties);
        let m = config.num_instances();
        let n_per_party = config.num_constraints() / config.num_parties();

        let dur_seq = bench_avg(|| { let _ = config.build_partitioned_circuits::<Fr>(); }, 3);
        let dur_par = bench_avg(|| { let _ = config.build_partitioned_circuits_par::<Fr>(); }, 3);

        let label_seq = format!("seq  M={}, N/K={}", m, n_per_party);
        let label_par = format!("par  M={}, N/K={}", m, n_per_party);
        row(&label_seq, dur_seq);
        row(&label_par, dur_par);
        println!("  {:<50} {:>12.2}x", "  speedup", dur_seq.as_secs_f64() / dur_par.as_secs_f64());
        println!("  ──────────────────────────────────────────────────────────────────");
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

/// 2. circuits_to_sumcheck conversion
fn bench_circuits_to_sumcheck() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Bench: circuits_to_sumcheck                                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    let configs = [
        (2, 12, 2), // M=4, N=2^12, K=4
        (2, 14, 2), // M=4, N=2^14, K=4
        (3, 14, 2), // M=8, N=2^14, K=4
        (2, 16, 2), // M=4, N=2^16, K=4
    ];

    for (log_inst, log_cons, log_parties) in configs {
        let config = Config::new(log_inst, log_cons, GateType::Vanilla, log_parties);
        let srs = setup::<Bn254, PCS>(&config).unwrap();
        let (pk, _vk, circuits) = make_circuit::<Bn254, PCS>(&config, &srs).unwrap();

        let iters = if log_cons <= 14 { 5 } else { 2 };
        let dur = bench_avg(|| { let _ = circuits_to_sumcheck::<Bn254, PCS>(&pk, &circuits); }, iters);

        let m = config.num_instances();
        let nv = config.log_num_constraints - config.log_num_parties;
        let label = format!("M={}, num_vars={}", m, nv);
        row(&label, dur);
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

/// 3. sum_fold v1 vs v2 vs v3
fn bench_sum_fold_versions() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Bench: sum_fold v1 vs v2 vs v3                                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("  {:<26} {:>12} {:>12} {:>12}", "config", "v1", "v2", "v3");
    println!("  ──────────────────────────────────────────────────────────────────");

    let configs = [
        (10, 4, 3, 2),  // nv=10, m=4
        (12, 4, 3, 2),  // nv=12, m=4
        (14, 4, 3, 2),  // nv=14, m=4
        (12, 8, 3, 2),  // nv=12, m=8
        (14, 8, 3, 2),  // nv=14, m=8
        (16, 4, 3, 2),  // nv=16, m=4
        (16, 8, 3, 2),  // nv=16, m=8
    ];

    let iters = 3;

    for (nv, m, num_mult, num_prod) in configs {
        let (polys_orig, sums_orig) = gen_polys(nv, m, num_mult, num_prod);

        // v1
        let dur_v1 = {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let polys = polys_orig.iter().map(|p| p.deep_copy()).collect();
                let sums = sums_orig.clone();
                let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                let start = Instant::now();
                let _ = <PolyIOP<Fr> as SumCheck<Fr>>::sum_fold(polys, sums, &mut transcript).unwrap();
                total += start.elapsed();
            }
            total / iters
        };

        // v2
        let dur_v2 = {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let polys = polys_orig.iter().map(|p| p.deep_copy()).collect();
                let sums = sums_orig.clone();
                let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                let start = Instant::now();
                let _ = <PolyIOP<Fr> as SumCheck<Fr>>::sum_fold_v2(polys, sums, &mut transcript).unwrap();
                total += start.elapsed();
            }
            total / iters
        };

        // v3
        let dur_v3 = {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let polys = polys_orig.iter().map(|p| p.deep_copy()).collect();
                let sums = sums_orig.clone();
                let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                let start = Instant::now();
                let _ = <PolyIOP<Fr> as SumCheck<Fr>>::sum_fold_v3(polys, sums, &mut transcript).unwrap();
                total += start.elapsed();
            }
            total / iters
        };

        println!(
            "  nv={:2}, m={:<2}               {:>12.3?} {:>12.3?} {:>12.3?}",
            nv, m, dur_v1, dur_v2, dur_v3,
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

/// 4. SumCheck prove (standard, single polynomial)
fn bench_sumcheck_prove() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Bench: SumCheck prove (single polynomial)                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    let mut rng = test_rng();
    for nv in [10, 12, 14, 16, 18, 20] {
        let iters = if nv <= 14 { 10 } else if nv <= 18 { 3 } else { 1 };

        let (poly, _sum) =
            VirtualPolynomial::<Fr>::rand(nv, (3, 4), 2, &mut rng).unwrap();

        let dur = {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                let start = Instant::now();
                let _ = <PolyIOP<Fr> as SumCheck<Fr>>::prove(&poly, &mut transcript).unwrap();
                total += start.elapsed();
            }
            total / iters as u32
        };

        let label = format!("nv={}, degree=3, products=2", nv);
        row(&label, dur);
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

/// 5. prove_sumfold end-to-end (from real circuits)
fn bench_prove_sumfold_e2e() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Bench: prove_sumfold end-to-end (real circuits)                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    let configs = [
        (2, 12, 2), // M=4, nv=10 (12-2)
        (2, 14, 2), // M=4, nv=12
        (3, 14, 2), // M=8, nv=12
        (2, 16, 2), // M=4, nv=14
        (3, 16, 2), // M=8, nv=14
    ];

    for (log_inst, log_cons, log_parties) in configs {
        let config = Config::new(log_inst, log_cons, GateType::Vanilla, log_parties);
        let srs = setup::<Bn254, PCS>(&config).unwrap();
        let (pk, _vk, circuits) = make_circuit::<Bn254, PCS>(&config, &srs).unwrap();

        let m = config.num_instances();
        let nv = config.log_num_constraints - config.log_num_parties;
        let iters = if nv <= 12 { 3 } else { 1 };

        // Phase 1.5: conversion
        let start_conv = Instant::now();
        let _instances = circuits_to_sumcheck::<Bn254, PCS>(&pk, &circuits).unwrap();
        let dur_conv = start_conv.elapsed();

        // Phase 2: sum_fold (in release mode, only runs v2)
        let dur_sumfold = {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let instances_copy = circuits_to_sumcheck::<Bn254, PCS>(&pk, &circuits).unwrap();
                let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                let start = Instant::now();
                let _ = prove_sumfold(instances_copy, &mut transcript).unwrap();
                total += start.elapsed();
            }
            total / iters as u32
        };

        let label_conv = format!("  circuits_to_sumcheck   M={}, nv={}", m, nv);
        let label_fold = format!("  prove_sumfold          M={}, nv={}", m, nv);
        row(&label_conv, dur_conv);
        row(&label_fold, dur_sumfold);
        println!("  ──────────────────────────────────────────────────────────────────");
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

/// 6. fix_variables micro-benchmark
fn bench_fix_variables() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Bench: fix_variables (single MLE, one variable)                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    let mut rng = test_rng();

    for nv in [12, 14, 16, 18, 20] {
        let mle = DenseMultilinearExtension::<Fr>::rand(nv, &mut rng);
        let r = Fr::from(42u64);

        let iters = if nv <= 16 { 50 } else if nv <= 18 { 10 } else { 3 };
        let dur = bench_avg(|| { let _ = fix_variables(&mle, &[r]); }, iters);

        let label = format!("nv={} (2^{} evals → 2^{})", nv, nv, nv - 1);
        row(&label, dur);
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

/// 7. MLE clone overhead (Arc<MLE> → clone → Arc<MLE>)
fn bench_mle_clone_roundtrip() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Bench: MLE Arc clone roundtrip (prover bottleneck)                 ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    let mut rng = test_rng();
    let t = 8; // typical number of MLEs in a VP

    for nv in [12, 14, 16, 18] {
        let mles: Vec<Arc<DenseMultilinearExtension<Fr>>> = (0..t)
            .map(|_| Arc::new(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng)))
            .collect();

        let iters = if nv <= 14 { 20 } else { 5 };

        // Clone out from Arc
        let dur_clone_out = bench_avg(
            || {
                let _: Vec<DenseMultilinearExtension<Fr>> = mles
                    .iter()
                    .map(|x| x.as_ref().clone())
                    .collect();
            },
            iters,
        );

        // Clone back into Arc
        let cloned: Vec<DenseMultilinearExtension<Fr>> = mles
            .iter()
            .map(|x| x.as_ref().clone())
            .collect();
        let dur_clone_back = bench_avg(
            || {
                let _: Vec<Arc<DenseMultilinearExtension<Fr>>> = cloned
                    .iter()
                    .map(|x| Arc::new(x.clone()))
                    .collect();
            },
            iters,
        );

        let label_out = format!("clone out  t={}, nv={}", t, nv);
        let label_back = format!("clone back t={}, nv={}", t, nv);
        let label_total = format!("roundtrip  t={}, nv={}", t, nv);
        row(&label_out, dur_clone_out);
        row(&label_back, dur_clone_back);
        row(&label_total, dur_clone_out + dur_clone_back);
        println!("  ──────────────────────────────────────────────────────────────────");
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

/// 8. verify_proof_eval (sequential MLE evaluation)
fn bench_verify_eval() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Bench: MLE evaluate (simulates verify_proof_eval bottleneck)       ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    let mut rng = test_rng();

    // Simulate: M circuits × (num_selectors + num_witnesses) MLE evaluations
    let num_selectors = 5;
    let num_witnesses = 3;
    let total_mles_per_circuit = num_selectors + num_witnesses;

    for (m, nv) in [(4, 10), (4, 12), (8, 12), (4, 14), (8, 14)] {
        let mles: Vec<DenseMultilinearExtension<Fr>> = (0..m * total_mles_per_circuit)
            .map(|_| DenseMultilinearExtension::<Fr>::rand(nv, &mut rng))
            .collect();

        let point: Vec<Fr> = (0..nv).map(|_| Fr::from(42u64)).collect();

        let iters = if nv <= 12 { 10 } else { 3 };

        // Sequential evaluation
        let dur_seq = bench_avg(
            || {
                for mle in &mles {
                    let _ = mle.evaluate(&point).unwrap();
                }
            },
            iters,
        );

        // Parallel evaluation (what we'd optimize to)
        let dur_par = bench_avg(
            || {
                use rayon::prelude::*;
                let _: Vec<Fr> = mles
                    .par_iter()
                    .map(|mle| mle.evaluate(&point).unwrap())
                    .collect();
            },
            iters,
        );

        let total = m * total_mles_per_circuit;
        let label_seq = format!("seq  M={}, nv={}, {} evals", m, nv, total);
        let label_par = format!("par  M={}, nv={}, {} evals", m, nv, total);
        row(&label_seq, dur_seq);
        row(&label_par, dur_par);
        println!("  {:<50} {:>12.2}x", "  speedup", dur_seq.as_secs_f64() / dur_par.as_secs_f64());
        println!("  ──────────────────────────────────────────────────────────────────");
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

/// 9. stage3_merge_split_mles (the cache-unfriendly triple loop)
fn bench_stage3_merge() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Bench: stage3_merge_split_mles                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    use subroutines::poly_iop::sum_check::stage3_merge_split_mles;

    let configs = [
        (10, 4, 3, 2),
        (12, 4, 3, 2),
        (14, 4, 3, 2),
        (12, 8, 3, 2),
        (14, 8, 3, 2),
    ];

    for (nv, m, num_mult, num_prod) in configs {
        let (polys, _sums) = gen_polys(nv, m, num_mult, num_prod);
        let t = polys[0].flattened_ml_extensions.len();
        let length = (m as f64).log2() as usize;

        let all_splits: Vec<Vec<VirtualPolynomial<Fr>>> = polys
            .iter()
            .map(|vp| vp.split_by_last_variables(length))
            .collect();

        let new_num_vars = length + nv;
        let iters = if nv <= 12 { 10 } else { 3 };

        let dur = bench_avg(
            || { let _ = stage3_merge_split_mles(&all_splits, m, t, new_num_vars); },
            iters,
        );

        let label = format!("nv={}, m={}, t={}", nv, m, t);
        row(&label, dur);
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

/// 10. Full single-machine pipeline: setup → circuit → sumcheck → sumfold → verify
fn bench_full_pipeline() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Bench: Full pipeline breakdown (single-machine, no network)        ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    let configs = [
        (2, 14, 2), // M=4, N=2^14, K=4
        (3, 14, 2), // M=8, N=2^14, K=4
        (2, 16, 2), // M=4, N=2^16, K=4
    ];

    for (log_inst, log_cons, log_parties) in configs {
        let config = Config::new(log_inst, log_cons, GateType::Vanilla, log_parties);
        let m = config.num_instances();
        let nv = config.log_num_constraints - config.log_num_parties;

        println!("  ── M={}, N=2^{}, K={}, nv={} ──", m, log_cons, 1 << log_parties, nv);

        // Phase 0: Setup
        let start = Instant::now();
        let srs = setup::<Bn254, PCS>(&config).unwrap();
        let dur_setup = start.elapsed();
        row("    setup (SRS gen/load)", dur_setup);

        // Phase 1: Make circuit
        let start = Instant::now();
        let (pk, _vk, circuits) = make_circuit::<Bn254, PCS>(&config, &srs).unwrap();
        let dur_circuit = start.elapsed();
        row("    make_circuit (build + preprocess)", dur_circuit);

        // Phase 1.5: Convert to SumCheck instances
        let start = Instant::now();
        let instances = circuits_to_sumcheck::<Bn254, PCS>(&pk, &circuits).unwrap();
        let dur_convert = start.elapsed();
        row("    circuits_to_sumcheck", dur_convert);

        // Phase 2: SumFold
        let start = Instant::now();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let (_folded, sumfold_proof) = prove_sumfold(instances, &mut transcript).unwrap();
        let dur_sumfold = start.elapsed();
        row("    prove_sumfold", dur_sumfold);

        // Phase 2.5: merge_and_verify (K=1 trivial)
        let start = Instant::now();
        let _ = merge_and_verify_sumfold(vec![sumfold_proof]).unwrap();
        let dur_verify_sumfold = start.elapsed();
        row("    merge_and_verify_sumfold (K=1)", dur_verify_sumfold);

        // Phase 3: SumCheck prove on folded instance
        let instances2 = circuits_to_sumcheck::<Bn254, PCS>(&pk, &circuits).unwrap();
        let mut transcript2 = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let (folded2, _) = prove_sumfold(instances2, &mut transcript2).unwrap();
        let start = Instant::now();
        let _ = <PolyIOP<Fr> as SumCheck<Fr>>::prove(&folded2.poly, &mut transcript2).unwrap();
        let dur_prove_folded = start.elapsed();
        row("    SumCheck prove (folded poly)", dur_prove_folded);

        let dur_total = dur_setup + dur_circuit + dur_convert + dur_sumfold + dur_prove_folded;
        row("    TOTAL (excl. PCS)", dur_total);
        println!("  ──────────────────────────────────────────────────────────────────");
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

// ═══════════════════════════════════════════════════════════════════════════
// Trait extension for evaluating VP sum over boolean hypercube
// ═══════════════════════════════════════════════════════════════════════════
trait VPEvalHypercube<F: PrimeField> {
    fn evaluate_over_boolean_hypercube(&self) -> F;
}

impl<F: PrimeField> VPEvalHypercube<F> for VirtualPolynomial<F> {
    fn evaluate_over_boolean_hypercube(&self) -> F {
        let nv = self.aux_info.num_variables;
        let mut sum = F::zero();
        for b in 0..(1 << nv) {
            let point: Vec<F> = (0..nv)
                .map(|i| if (b >> i) & 1 == 1 { F::one() } else { F::zero() })
                .collect();
            sum += self.evaluate(&point).unwrap();
        }
        sum
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let threads = rayon::current_num_threads();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  deSNARK Benchmark Suite — {} threads", threads);
    println!("═══════════════════════════════════════════════════════════════════════");

    // Micro-benchmarks (fast, targeted)
    bench_fix_variables();
    bench_mle_clone_roundtrip();
    bench_stage3_merge();
    bench_verify_eval();

    // Component benchmarks
    bench_build_circuits();
    bench_circuits_to_sumcheck();
    bench_sumcheck_prove();
    bench_sum_fold_versions();
    bench_prove_sumfold_e2e();

    // Full pipeline
    bench_full_pipeline();
}
