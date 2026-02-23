//! Distributed Sum-Fold Demo (SCMN Pattern)
//!
//! This demo follows the Single Code Multiple Node pattern:
//! - Each party has only its OWN polynomial locally
//! - Sums are computed locally and aggregated at master
//! - No polynomial data is sent over the network
//!
//! Run 4 instances:
//! ```bash
//! cargo run --example dist_sum_fold_demo --features distributed -- 0 hosts_4.txt &
//! cargo run --example dist_sum_fold_demo --features distributed -- 1 hosts_4.txt &
//! cargo run --example dist_sum_fold_demo --features distributed -- 2 hosts_4.txt &
//! cargo run --example dist_sum_fold_demo --features distributed -- 3 hosts_4.txt &
//! ```

use ark_bls12_381::Fr;
use ark_std::rand::SeedableRng;
use deNetwork::{DeMultiNet, DeNet};
use log::{debug, error, info};
use std::env;
use std::time::Instant;

use arithmetic::VirtualPolynomial;
use subroutines::poly_iop::sum_check::dist_sum_fold::dist_sum_fold;
use subroutines::poly_iop::sum_check::SumCheck;
use subroutines::poly_iop::PolyIOP;

/// Generate deterministic seed for party i.
/// All parties using the same party_id will generate identical polynomials.
fn party_seed(party_id: usize) -> [u8; 32] {
    [party_id as u8; 32]
}

fn main() {
    // Initialize env_logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| {
            use std::io::Write;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            let secs = now.as_secs() % 100;
            let millis = now.as_millis() % 1000;
            writeln!(
                buf,
                "[{:02}.{:03}] [{}] {}",
                secs,
                millis,
                record.level(),
                record.args()
            )
        })
        .init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        error!("Usage: {} <party_id> <hosts_file>", args[0]);
        error!("Example: {} 0 hosts_4.txt", args[0]);
        std::process::exit(1);
    }

    let party_id: usize = args[1].parse().expect("Invalid party_id");
    let hosts_file = &args[2];

    info!("[Party {}] Starting up...", party_id);
    info!("[Party {}] Hosts file: {}", party_id, hosts_file);

    // Initialize the network
    info!("[Party {}] Connecting to network...", party_id);
    DeMultiNet::init_from_file(hosts_file, party_id);
    let m = DeMultiNet::n_parties();
    info!("[Party {}] Connected! {} parties total", party_id, m);

    // Small delay to ensure network is stable
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Configuration
    let nv = 8; // Number of variables in each VP
    let num_multiplicands = 2;
    let num_products = 2;

    // ═══════════════════════════════════════════════════════════════════════
    // SCMN Pattern: Each party generates only its OWN polynomial
    // Sum is computed locally, aggregated at master
    // ═══════════════════════════════════════════════════════════════════════

    info!(
        "[Party {}] Generating local VP with {} variables",
        party_id, nv
    );

    // Each party uses different seed based on party_id
    let mut rng = ark_std::rand::rngs::StdRng::from_seed(party_seed(party_id));

    let (my_poly, my_sum) = VirtualPolynomial::<Fr>::rand(
        nv,
        (num_multiplicands, num_multiplicands + 1),
        num_products,
        &mut rng,
    )
    .expect("Failed to create polynomial");

    debug!(
        "[Party {}] Local VP created, {} MLEs, sum computed locally",
        party_id,
        my_poly.flattened_ml_extensions.len()
    );

    // For verification: master regenerates all polynomials to compare with distributed result.
    // Using seed = [i as u8; 32] ensures polys[i] is identical to what party i generates
    // as their my_poly (same deterministic RNG seed produces identical polynomial).
    let (all_polys_clone, all_sums_clone) = if DeMultiNet::am_master() {
        let mut polys = Vec::with_capacity(m);
        let mut sums = Vec::with_capacity(m);

        for i in 0..m {
            // Same seed as party i uses for my_poly → polys[i] == party i's my_poly
            let mut rng = ark_std::rand::rngs::StdRng::from_seed(party_seed(i));
            let (poly, sum) = VirtualPolynomial::<Fr>::rand(
                nv,
                (num_multiplicands, num_multiplicands + 1),
                num_products,
                &mut rng,
            )
            .expect("Failed to create polynomial");
            polys.push(poly);
            sums.push(sum);
        }
        (Some(polys), Some(sums))
    } else {
        (None, None)
    };

    info!("[Party {}] Starting distributed sum_fold...", party_id);
    let start = Instant::now();

    // Master uses transcript; workers pass None
    let mut transcript = if DeMultiNet::am_master() {
        Some(<PolyIOP<Fr> as SumCheck<Fr>>::init_transcript())
    } else {
        None
    };

    // ═══════════════════════════════════════════════════════════════════════
    // SCMN: Each party passes its OWN polynomial and locally computed sum
    // Sums are aggregated at master, challenges synchronized to all
    // ═══════════════════════════════════════════════════════════════════════
    let result = dist_sum_fold::<Fr, DeMultiNet>(my_poly, my_sum, transcript.as_mut());

    let duration = start.elapsed();

    // Handle result - master gets Ok with data, workers get Err (expected)
    match result {
        Ok((dist_proof, dist_sum_t, _aux_info, dist_folded_poly, dist_v)) => {
            // Only master reaches here
            info!("[Party {}] ════════════════════════════════", party_id);
            info!(
                "[Party {}] Distributed protocol completed in {:?}",
                party_id, duration
            );
            info!("[Party {}] Proof rounds: {}", party_id, dist_proof.point.len());
            info!(
                "[Party {}] Folded poly vars: {}",
                party_id, dist_folded_poly.aux_info.num_variables
            );

            // Verify against local computation
            if let (Some(polys_clone), Some(sums_clone)) = (all_polys_clone, all_sums_clone) {
                info!(
                    "[Party {}] Running local sum_fold_v3 for comparison...",
                    party_id
                );
                let mut local_transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();

                match <PolyIOP<Fr> as SumCheck<Fr>>::sum_fold_v3(
                    polys_clone,
                    sums_clone,
                    &mut local_transcript,
                ) {
                    Ok((local_proof, local_sum_t, _local_aux, _local_folded_poly, local_v)) => {
                        let sum_t_match = dist_sum_t == local_sum_t;
                        let v_match = dist_v == local_v;
                        let proof_match = dist_proof.point == local_proof.point;

                        info!("[Party {}] sum_t match: {}", party_id, sum_t_match);
                        info!("[Party {}] v match: {}", party_id, v_match);
                        info!("[Party {}] challenges match: {}", party_id, proof_match);

                        if sum_t_match && v_match && proof_match {
                            info!("[Party {}] ✓ Distributed result matches local!", party_id);
                        } else {
                            error!("[Party {}] ✗ Results MISMATCH!", party_id);
                            debug!("[Party {}] dist_sum_t: {:?}", party_id, dist_sum_t);
                            debug!("[Party {}] local_sum_t: {:?}", party_id, local_sum_t);
                            debug!("[Party {}] dist_v: {:?}", party_id, dist_v);
                            debug!("[Party {}] local_v: {:?}", party_id, local_v);
                        }
                    }
                    Err(e) => {
                        error!("[Party {}] Local sum_fold_v3 failed: {:?}", party_id, e);
                    }
                }
            }

            info!("[Party {}] ════════════════════════════════", party_id);
        }
        Err(_) => {
            // Workers reach here - this is expected behavior
            info!("[Party {}] ✓ Completed in {:?}", party_id, duration);
        }
    }

    info!("[Party {}] Closing network connection...", party_id);
    DeMultiNet::deinit();
    info!("[Party {}] Exiting successfully", party_id);
}
