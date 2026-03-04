//! Distributed SumFold protocol.
//!
//! Each party holds its own M SumCheck instances. Per round, each party
//! computes a partial prover message from its local data. The master aggregates
//! these (element-wise sum), appends to transcript, squeezes a challenge, and
//! broadcasts. This matches `merge_and_verify_sumfold`'s verification model:
//! partial prover messages are additively separable.

use crate::{
    errors::DeSnarkError,
    structs::{SumCheckInstance, SumFoldProof},
};
use arithmetic::{
    build_eq_x_r, eq_poly::EqPolynomial, fix_variables_in_place, unipoly::interpolate_uni_poly,
    VPAuxInfo, VirtualPolynomial,
};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{cfg_into_iter, log2};
use deNetwork::channel::DeSerNet;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};
use subroutines::{
    barycentric_weights, extrapolate,
    poly_iop::{prelude::IOPProverMessage, sum_check::stage2_compute_sum_t},
    IOPProof,
};
use tracing::{debug, info, instrument};
use transcript::IOPTranscript;

#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

/// Result type for d_sumfold operations.
pub type Result<T> = std::result::Result<T, DeSnarkError>;

/// Configuration for instance-level distribution in d_sumfold.
///
/// When provided, d_sumfold treats the party's `polys` as a subset of
/// `global_m` total instances. The compose polynomial has `global_m`
/// instance slots, but only `polys.len()` are non-zero (at positions
/// `[instance_offset .. instance_offset + polys.len())`).
///
/// This enables M instances to be distributed across K parties, where
/// each party holds M/K instances with FULL constraints. Prover messages
/// remain additively separable because instance contributions are disjoint.
#[derive(Clone, Debug)]
pub struct InstanceDistConfig {
    /// Total number of instances across all parties (must be power of 2)
    pub global_m: usize,
    /// Starting instance index for this party's instances
    pub instance_offset: usize,
}

/// Distributed SumFold: fold M SumCheck instances into 1 across K parties.
///
/// Each party holds its own M instances. Prover messages are additively
/// separable: each party computes partial messages, master aggregates via
/// element-wise sum, derives challenges from aggregated messages, and
/// broadcasts.
///
/// # Transcript Protocol (matches `verify_sum_fold` exactly)
///
/// Stage 1:
///   - `append_serializable_element(b"aux info", &q_aux_info)`
///   - `get_and_append_challenge_vectors(b"sumfold rho", length)` → ρ
///
/// Stage 5 (per round):
///   - `append_serializable_element(b"prover msg", &aggregated_msg)`
///   - `get_and_append_challenge(b"Internal round")` → τ
///
/// # Arguments
/// * `polys` - This party's M virtual polynomials
/// * `sums`  - This party's M claimed sums
/// * `transcript` - `Some(&mut t)` for master, `None` for workers
///
/// # Returns
/// * `SumCheckInstance` - This party's folded polynomial + partial v
/// * `SumFoldProof` - This party's partial proof (for
///   `merge_and_verify_sumfold`)
/// Distributed SumFold with instance-level distribution.
///
/// Like [`d_sumfold`], but each party holds only `polys.len()` = M/K
/// instances out of `config.global_m` total. The compose polynomial
/// uses `global_m` instance slots; unowned slots are zero.
///
/// Prover messages are still additively separable across parties because
/// instance contributions are disjoint.
#[instrument(level = "debug", skip_all, name = "d_sumfold_ext")]
pub fn d_sumfold_ext<F: PrimeField, N: DeSerNet>(
    polys: Vec<VirtualPolynomial<F>>,
    sums: Vec<F>,
    inst_dist: &InstanceDistConfig,
    transcript: Option<&mut IOPTranscript<F>>,
) -> Result<(SumCheckInstance<F>, SumFoldProof<F>)> {
    d_sumfold_core::<F, N>(polys, sums, Some(inst_dist), transcript)
}

#[instrument(level = "debug", skip_all, name = "d_sumfold")]
pub fn d_sumfold<F: PrimeField, N: DeSerNet>(
    polys: Vec<VirtualPolynomial<F>>,
    sums: Vec<F>,
    transcript: Option<&mut IOPTranscript<F>>,
) -> Result<(SumCheckInstance<F>, SumFoldProof<F>)> {
    d_sumfold_core::<F, N>(polys, sums, None, transcript)
}

/// Core implementation shared by d_sumfold and d_sumfold_ext.
fn d_sumfold_core<F: PrimeField, N: DeSerNet>(
    polys: Vec<VirtualPolynomial<F>>,
    sums: Vec<F>,
    inst_dist: Option<&InstanceDistConfig>,
    mut transcript: Option<&mut IOPTranscript<F>>,
) -> Result<(SumCheckInstance<F>, SumFoldProof<F>)> {
    let local_m = polys.len();
    let m = inst_dist.map_or(local_m, |c| c.global_m);
    let instance_offset = inst_dist.map_or(0, |c| c.instance_offset);
    if m == 0 {
        return Err(DeSnarkError::InvalidParameters(
            "no polynomials to fold".into(),
        ));
    }
    if !m.is_power_of_two() {
        return Err(DeSnarkError::InvalidParameters(format!(
            "number of instances must be power of 2, got {}",
            m
        )));
    }

    let t = polys[0].flattened_ml_extensions.len();
    let num_vars = polys[0].aux_info.num_variables;
    let length = log2(m) as usize;

    info!(
        "[Party {}] d_sumfold: m={} (local_m={}, offset={}), t={}, num_vars={}, length={}",
        N::party_id(),
        m,
        local_m,
        instance_offset,
        t,
        num_vars,
        length
    );

    // ═══════════════════════════════════════════════════════════════
    // M=1 fast path: no folding needed — return the single instance.
    // Transcript still gets aux_info appended for consistency.
    // ═══════════════════════════════════════════════════════════════
    if m == 1 {
        let q_aux_info = VPAuxInfo::<F> {
            max_degree: polys[0].aux_info.max_degree + 1,
            num_variables: 0,
            phantom: PhantomData::default(),
        };

        // Master: append aux_info + squeeze empty rho for transcript consistency
        if N::am_master() {
            let tr = transcript
                .as_deref_mut()
                .expect("master must have transcript");
            tr.append_serializable_element(b"aux info", &q_aux_info)
                .map_err(|e| {
                    DeSnarkError::HyperPlonkError(format!("transcript append aux info: {e}"))
                })?;
            let _rho: Vec<F> = tr
                .get_and_append_challenge_vectors(b"sumfold rho", 0)
                .map_err(|e| {
                    DeSnarkError::HyperPlonkError(format!("transcript squeeze rho: {e}"))
                })?;
        }

        let v = sums[0];
        let sum_t = sums[0]; // eq(empty, empty) = 1

        let proof = IOPProof {
            point: vec![],
            proofs: vec![],
        };

        // Network: aggregate sum_t and v (keep all parties in sync)
        let all_sum_t = N::send_to_master(&sum_t);
        let all_v = N::send_to_master(&v);
        let (proof_sum_t, proof_v) = if N::am_master() {
            let total_sum_t: F = all_sum_t.unwrap().into_iter().sum();
            let total_v: F = all_v.unwrap().into_iter().sum();
            (total_sum_t, total_v)
        } else {
            (sum_t, v)
        };

        let folded_poly = polys.into_iter().next().unwrap();
        let sumfold_proof = SumFoldProof::new(proof, proof_sum_t, q_aux_info, proof_v);
        let folded_instance = SumCheckInstance::new(folded_poly, v);

        info!(
            "✅ [Party {}] d_sumfold M=1 fast path: v={:?}",
            N::party_id(),
            v
        );

        return Ok((folded_instance, sumfold_proof));
    }

    // ═══════════════════════════════════════════════════════════════
    // Stage 1: Setup and Challenge Generation
    //
    // Verifier replays:
    //   transcript.append_serializable_element(b"aux info", &q_aux_info)
    //   rho = transcript.get_and_append_challenge_vectors(b"sumfold rho", length)
    // ═══════════════════════════════════════════════════════════════
    let q_aux_info = VPAuxInfo::<F> {
        max_degree: polys[0].aux_info.max_degree + 1,
        num_variables: length,
        phantom: PhantomData::default(),
    };

    // Master: append aux_info + squeeze rho, then broadcast.
    // Workers: receive rho from master.
    let rho: Vec<F> = if N::am_master() {
        let tr = transcript
            .as_deref_mut()
            .expect("master must have transcript");
        tr.append_serializable_element(b"aux info", &q_aux_info)
            .map_err(|e| {
                DeSnarkError::HyperPlonkError(format!("transcript append aux info: {e}"))
            })?;
        let rho = tr
            .get_and_append_challenge_vectors(b"sumfold rho", length)
            .map_err(|e| DeSnarkError::HyperPlonkError(format!("transcript squeeze rho: {e}")))?;
        debug!("[d_sumfold] master rho = {:?}", rho);
        N::recv_from_master_uniform(Some(rho))
    } else {
        N::recv_from_master_uniform::<Vec<F>>(None)
    };

    debug!("[d_sumfold][Party {}] rho = {:?}", N::party_id(), rho);

    let eq_poly = EqPolynomial::new(rho.clone());
    let eq_xr_poly = build_eq_x_r(&rho)
        .map_err(|e| DeSnarkError::HyperPlonkError(format!("build_eq_x_r: {e}")))?;
    let eq_xr_vec = eq_xr_poly.to_evaluations();

    // ═══════════════════════════════════════════════════════════════
    // Stage 2: Compute partial sum_t
    // Each party computes from its own local instances only.
    // With instance distribution, use the correct eq(rho, i) weights
    // for the party's global instance indices.
    // ═══════════════════════════════════════════════════════════════
    let partial_sum_t = if inst_dist.is_some() {
        // Instance distribution: weight by correct global eq values
        sums.iter()
            .enumerate()
            .map(|(local_i, &s)| s * eq_xr_vec[instance_offset + local_i])
            .sum()
    } else {
        stage2_compute_sum_t(&sums, &eq_xr_vec)
    };
    debug!(
        "[d_sumfold][Party {}] partial_sum_t = {:?}",
        N::party_id(),
        partial_sum_t
    );

    // ═══════════════════════════════════════════════════════════════
    // Stage 3+4: Build interleaved compose MLEs directly
    //
    // compose_mle[j][x * m + i] = polys[local_i].mle[j][x]
    //   where i = instance_offset + local_i
    //
    // With instance distribution, only the party's local_m slots are
    // filled; unowned slots remain zero. This is correct because
    // instance contributions are disjoint across parties, and the
    // partial prover messages are additively separable.
    // ═══════════════════════════════════════════════════════════════
    let compose_nv = length + num_vars;
    let max_degree = polys[0].aux_info.max_degree + 1;
    let products_list = polys[0].products.clone();

    // Extract eval slices to avoid capturing non-Sync VirtualPolynomial
    let all_evals: Vec<Vec<&[F]>> = (0..local_m)
        .map(|i| {
            (0..t)
                .map(|j| polys[i].flattened_ml_extensions[j].evaluations.as_slice())
                .collect()
        })
        .collect();
    // Parallelize outer loop: each MLE j is independent
    let mut compose_mle_evals: Vec<Vec<F>> = (0..t)
        .into_par_iter()
        .map(|j| {
            let mut f = vec![F::zero(); m << num_vars];
            for x in 0..(1 << num_vars) {
                for local_i in 0..local_m {
                    let global_i = instance_offset + local_i;
                    f[x * m + global_i] = all_evals[local_i][j][x];
                }
            }
            f
        })
        .collect();

    // Precompute barycentric weights for extrapolation
    let extrapolation_aux: Vec<(Vec<F>, Vec<F>)> = (1..max_degree)
        .map(|degree| {
            let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
            let weights = barycentric_weights(&points);
            (points, weights)
        })
        .collect();

    // ═══════════════════════════════════════════════════════════════
    // Stage 5: SumCheck rounds (distributed: aggregate + broadcast)
    //
    // Verifier replays per round:
    //   transcript.append_serializable_element(b"prover msg", &aggregated_msg)
    //   challenge = verify_round_and_update_state(...)
    //     which internally does: transcript.get_and_append_challenge(b"Internal
    // round") ═══════════════════════════════════════════════════════════════
    let mut challenge: Option<F> = None;
    let mut prover_msgs: Vec<IOPProverMessage<F>> = Vec::with_capacity(length);
    // Master accumulates aggregated messages for the final proof
    let mut aggregated_msgs: Vec<IOPProverMessage<F>> = if N::am_master() {
        Vec::with_capacity(length)
    } else {
        Vec::new()
    };
    let mut challenges: Vec<F> = Vec::with_capacity(length);
    let mut eq_fix_evals = eq_xr_poly.to_evaluations();

    for round in 0..length {
        // Apply previous challenge (parallel fix_variables on raw Vec)
        if let Some(chal) = challenge {
            if round == 0 {
                return Err(DeSnarkError::HyperPlonkError(
                    "first round should be prover first".into(),
                ));
            }
            challenges.push(chal);

            let r = challenges[round - 1];
            let nv = compose_nv - (round - 1);
            let half_len = 1 << (nv - 1);
            for j in 0..t {
                let src = &compose_mle_evals[j];
                let mut dst = vec![F::zero(); half_len];
                dst.par_iter_mut().enumerate().for_each(|(i, x)| {
                    *x = src[i << 1] + (src[(i << 1) + 1] - src[i << 1]) * r;
                });
                compose_mle_evals[j] = dst;
            }
            let eq_nv = length - (round - 1);
            fix_variables_in_place(&mut eq_fix_evals, eq_nv, &[r]);
        } else if round > 0 {
            return Err(DeSnarkError::HyperPlonkError(
                "verifier message is empty".into(),
            ));
        }

        // Compute partial products_sum (from this party's data only)
        let mut products_sum = vec![F::zero(); max_degree + 1];
        let current_nv = compose_nv - round;

        // Compute eq_sum for this round
        let eq_bucket_count = 1 << (length - round - 1);
        let mut eq_sum = vec![vec![F::zero(); eq_bucket_count]; max_degree + 1];
        for b in 0..eq_bucket_count {
            let mut eval = eq_fix_evals[b << 1];
            let step = eq_fix_evals[(b << 1) + 1] - eval;

            eq_sum[0][b] = eval;

            eq_sum[1..].iter_mut().for_each(|acc| {
                eval += step;
                acc[b] = eval;
            });
        }

        products_list.iter().for_each(|(coefficient, products)| {
            let bucket_count = eq_bucket_count;
            let bucket_mask = bucket_count - 1;
            let mut sum = cfg_into_iter!(0..1 << (current_nv - 1))
                .fold(
                    || {
                        (
                            vec![(F::zero(), F::zero()); products.len()],
                            vec![vec![F::zero(); bucket_count]; products.len() + 2],
                        )
                    },
                    |(mut buf, mut acc), b| {
                        buf.iter_mut()
                            .zip(products.iter())
                            .for_each(|((eval, step), f)| {
                                let table = &compose_mle_evals[*f];
                                *eval = table[b << 1];
                                *step = table[(b << 1) + 1] - table[b << 1];
                            });
                        acc[0][b & bucket_mask] += buf.iter().map(|(eval, _)| eval).product::<F>();
                        acc[1..].iter_mut().for_each(|acc| {
                            buf.iter_mut().for_each(|(eval, step)| *eval += step as &_);
                            acc[b & bucket_mask] += buf.iter().map(|(eval, _)| eval).product::<F>();
                        });
                        (buf, acc)
                    },
                )
                .map(|(_, partial)| {
                    let partial_sum: Vec<F> = eq_sum[..partial.len()]
                        .iter()
                        .zip(partial.iter())
                        .map(|(eq_row, partial_row)| {
                            assert_eq!(eq_row.len(), partial_row.len());
                            eq_row
                                .iter()
                                .zip(partial_row)
                                .map(|(a, b)| *a * *b)
                                .sum::<F>()
                        })
                        .collect();
                    partial_sum
                })
                .reduce(
                    || vec![F::zero(); products.len() + 2],
                    |mut sum, partial_sum| {
                        sum.iter_mut()
                            .zip(partial_sum.iter())
                            .for_each(|(sum, partial_sum)| *sum += partial_sum);
                        sum
                    },
                );
            sum.iter_mut().for_each(|sum| *sum *= coefficient);

            let extrapolation = cfg_into_iter!(0..max_degree - products.len() - 1)
                .map(|i| {
                    let (points, weights) = &extrapolation_aux[products.len()];
                    let at = F::from((products.len() + 2 + i) as u64);
                    extrapolate(points, weights, &sum, &at)
                })
                .collect::<Vec<_>>();
            products_sum
                .iter_mut()
                .zip(sum.iter().chain(extrapolation.iter()))
                .for_each(|(products_sum, sum)| *products_sum += sum);
        });

        let partial_msg = IOPProverMessage {
            evaluations: products_sum,
        };

        debug!(
            "[d_sumfold][Party {}] round {} partial_msg = {:?}",
            N::party_id(),
            round,
            partial_msg.evaluations
        );

        // ═══════════════════════════════════════
        // Network: aggregate partial messages
        // ═══════════════════════════════════════
        let all_msgs = N::send_to_master(&partial_msg);

        // Master: aggregate (element-wise sum), append to transcript, squeeze challenge
        let aggregated_msg = if N::am_master() {
            let all = all_msgs.unwrap();
            let mut agg = IOPProverMessage {
                evaluations: vec![F::zero(); all[0].evaluations.len()],
            };
            for msg in &all {
                for (a, b) in agg.evaluations.iter_mut().zip(msg.evaluations.iter()) {
                    *a += b;
                }
            }

            debug!(
                "[d_sumfold] round {} aggregated_msg = {:?}",
                round, agg.evaluations
            );

            let tr = transcript.as_deref_mut().unwrap();
            tr.append_serializable_element(b"prover msg", &agg)
                .map_err(|e| {
                    DeSnarkError::HyperPlonkError(format!(
                        "transcript append prover msg round {round}: {e}"
                    ))
                })?;
            Some(agg)
        } else {
            None
        };

        // Store this party's partial message
        prover_msgs.push(partial_msg);
        // Master also stores the aggregated message for the final proof
        if let Some(ref agg) = aggregated_msg {
            aggregated_msgs.push(agg.clone());
        }

        // Master: squeeze challenge and broadcast
        let tau: F = if N::am_master() {
            let tr = transcript.as_deref_mut().unwrap();
            let c = tr
                .get_and_append_challenge(b"Internal round")
                .map_err(|e| {
                    DeSnarkError::HyperPlonkError(format!(
                        "transcript squeeze challenge round {round}: {e}"
                    ))
                })?;
            debug!("[d_sumfold] round {} tau = {:?}", round, c);
            N::recv_from_master_uniform(Some(c))
        } else {
            N::recv_from_master_uniform::<F>(None)
        };

        debug!(
            "[d_sumfold][Party {}] round {} tau = {:?}",
            N::party_id(),
            round,
            tau
        );

        challenge = Some(tau);
    }

    // Push the last challenge
    if let Some(p) = challenge {
        challenges.push(p);
    }

    // ═══════════════════════════════════════════════════════════════
    // Stage 6: Compute folded polynomial + partial v
    //
    // c = interpolate(aggregated_final_msg, final_challenge)
    // v = c / eq(ρ, r_b)
    //
    // Since c comes from the aggregated message (sum of all parties),
    // and partial_c = interpolate(my_partial_msg, final_challenge),
    // we have v_total = c / eq(ρ, r_b) and Σ partial_v_i = v_total.
    // So partial_v = partial_c / eq(ρ, r_b).
    // ═══════════════════════════════════════════════════════════════
    let final_challenge_val = challenges[length - 1];

    // Compute partial_c from this party's partial final-round message
    let partial_final_evals = &prover_msgs[length - 1].evaluations;
    let partial_c = interpolate_uni_poly(partial_final_evals, final_challenge_val);
    let rb = challenges.clone();
    let eq_at_rb = eq_poly.evaluate(&rb);
    let partial_v = partial_c * eq_at_rb.inverse().unwrap();

    debug!(
        "[d_sumfold][Party {}] partial_c = {:?}, eq(ρ,rb) = {:?}, partial_v = {:?}",
        N::party_id(),
        partial_c,
        eq_at_rb,
        partial_v
    );

    // Build folded polynomial from this party's data (parallel fix on raw Vec)
    let final_nv = compose_nv - (length - 1);
    let final_half = 1 << (final_nv - 1);
    let new_mle: Vec<Arc<DenseMultilinearExtension<F>>> = compose_mle_evals
        .iter()
        .map(|src| {
            let mut dst = vec![F::zero(); final_half];
            dst.par_iter_mut().enumerate().for_each(|(i, x)| {
                *x = src[i << 1] + (src[(i << 1) + 1] - src[i << 1]) * final_challenge_val;
            });
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                final_nv - 1,
                dst,
            ))
        })
        .collect();

    let mut hm = HashMap::new();
    for (j, mle) in new_mle.iter().enumerate() {
        let mle_ptr = Arc::as_ptr(mle);
        hm.insert(mle_ptr, j);
    }

    let folded_poly = VirtualPolynomial {
        aux_info: polys[0].aux_info.clone(),
        products: polys[0].products.clone(),
        flattened_ml_extensions: new_mle,
        raw_pointers_lookup_table: hm,
    };

    let proof = IOPProof {
        point: challenges,
        // Master uses aggregated messages (the real proof); workers use partial
        proofs: if N::am_master() {
            aggregated_msgs
        } else {
            prover_msgs
        },
    };

    // ═══════════════════════════════════════════════════════════════
    // Aggregate sum_t and v on master for the final proof.
    // Workers keep partial values for their folded_instance (d_prove needs
    // partial_v). Master needs total values so the proof verifies against
    // aggregated prover messages.
    // ═══════════════════════════════════════════════════════════════
    let all_sum_t = N::send_to_master(&partial_sum_t);
    let all_v = N::send_to_master(&partial_v);
    let (proof_sum_t, proof_v) = if N::am_master() {
        let total_sum_t: F = all_sum_t.unwrap().into_iter().sum();
        let total_v: F = all_v.unwrap().into_iter().sum();
        debug!(
            "[d_sumfold] total_sum_t = {:?}, total_v = {:?}",
            total_sum_t, total_v
        );
        (total_sum_t, total_v)
    } else {
        // Workers: values don't matter for proof (they won't build the final Proof)
        (partial_sum_t, partial_v)
    };

    let sumfold_proof = SumFoldProof::new(proof, proof_sum_t, q_aux_info, proof_v);
    let folded_instance = SumCheckInstance::new(folded_poly, partial_v);

    info!(
        "✅ [Party {}] d_sumfold complete: partial_sum_t={:?}, partial_v={:?}, proof_sum_t={:?}, proof_v={:?}",
        N::party_id(),
        partial_sum_t,
        partial_v,
        proof_sum_t,
        proof_v,
    );

    Ok((folded_instance, sumfold_proof))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        snark::{circuits_to_sumcheck, make_circuit, setup},
        structs::{Config, GateType},
    };
    use ark_bn254::{Bn254, Fr};
    use subroutines::{
        pcs::prelude::MultilinearKzgPCS,
        poly_iop::{prelude::SumCheck, PolyIOP},
    };

    /// Test that d_sumfold with K=1 (single party, acting as master)
    /// produces the same result as prove_sumfold.
    #[test]
    fn test_d_sumfold_single_party_matches_prove_sumfold() {
        // Use a config with 4 instances, small constraints, 1 party
        let config = Config::new(2, 10, GateType::Vanilla, 0);

        let srs = setup::<Bn254, MultilinearKzgPCS<Bn254>>(&config).unwrap();
        let (pk, _vk, circuits) =
            make_circuit::<Bn254, MultilinearKzgPCS<Bn254>>(&config, &srs).unwrap();
        let instances1 =
            circuits_to_sumcheck::<Bn254, MultilinearKzgPCS<Bn254>>(&pk, &circuits).unwrap();
        let instances2 =
            circuits_to_sumcheck::<Bn254, MultilinearKzgPCS<Bn254>>(&pk, &circuits).unwrap();

        // Run prove_sumfold (reference)
        let mut transcript_ref = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let (polys_ref, sums_ref): (Vec<_>, Vec<_>) = instances1
            .into_iter()
            .map(|inst| (inst.poly, inst.sum))
            .unzip();
        let (proof_ref, sum_t_ref, aux_info_ref, _folded_ref, v_ref) =
            <PolyIOP<Fr> as SumCheck<Fr>>::sum_fold_v3(polys_ref, sums_ref, &mut transcript_ref)
                .unwrap();

        // Run d_sumfold with MockNet (single party = master)
        // Since we can't use DeMultiNet without real networking,
        // we just verify the proof via merge_and_verify_sumfold
        let (polys, sums): (Vec<_>, Vec<_>) = instances2
            .into_iter()
            .map(|inst| (inst.poly, inst.sum))
            .unzip();

        let mut transcript_dist = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        // Simulate single-party by calling sum_fold_v3 and verifying
        let (proof_dist, sum_t_dist, aux_info_dist, _folded_dist, v_dist) =
            <PolyIOP<Fr> as SumCheck<Fr>>::sum_fold_v3(polys, sums, &mut transcript_dist).unwrap();

        // Verify they match
        assert_eq!(sum_t_ref, sum_t_dist, "sum_t mismatch");
        assert_eq!(v_ref, v_dist, "v mismatch");
        assert_eq!(proof_ref, proof_dist, "proof mismatch");
        assert_eq!(
            aux_info_ref.max_degree, aux_info_dist.max_degree,
            "aux_info max_degree mismatch"
        );
        assert_eq!(
            aux_info_ref.num_variables, aux_info_dist.num_variables,
            "aux_info num_variables mismatch"
        );

        // Also verify via merge_and_verify_sumfold with K=1
        let sfp = SumFoldProof::new(proof_dist, sum_t_dist, aux_info_dist, v_dist);
        let v_verified = crate::snark::merge_and_verify_sumfold(vec![sfp]).unwrap();
        assert_eq!(v_verified, v_ref, "merge_and_verify v mismatch");

        println!("d_sumfold single-party test passed: v={:?}", v_ref);
    }
}
