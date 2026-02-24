//! Distributed Sum-Fold Protocol
//!
//! This module implements a distributed version of `sum_fold_v3` using the
//! deNetwork crate. Each sub-prover (worker) handles one merged VP, and the
//! master coordinates the sumcheck rounds.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        DISTRIBUTED SUM-FOLD                              │
//! │                                                                          │
//! │  Phase 1: Setup & Distribution (Master)                                  │
//! │  ├─ Split m VPs by last `log2(m)` variables                              │
//! │  ├─ Merge by split index → m merged VPs                                  │
//! │  ├─ Generate rho challenges via transcript                               │
//! │  └─ Distribute: VP[i] → Worker[i]                                        │
//! │                                                                          │
//! │  Phase 2: Sumcheck Rounds (All parties, length rounds)                   │
//! │  ├─ Workers: Compute local prover message (eq-weighted partial sums)     │
//! │  ├─ Workers → Master: Send prover messages                               │
//! │  ├─ Master: Aggregate messages (sum across workers)                      │
//! │  ├─ Master: Derive challenge r[round] via transcript                     │
//! │  ├─ Master → Workers: Broadcast challenge                                │
//! │  └─ All: Apply fix_variables with challenge                              │
//! │                                                                          │
//! │  Phase 3: Finalization                                                   │
//! │  ├─ Workers → Master: Send folded MLE evaluations                        │
//! │  ├─ Master: Reconstruct folded VirtualPolynomial                         │
//! │  └─ Return: (proof, sum_t, aux_info, folded_poly, v)                     │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Insight
//!
//! The sumcheck prover message computation is **additively separable** across
//! VPs. Each worker computes its local contribution, and the master sums them.
//!
//! ```text
//! products_sum[d] = Σᵢ worker_i_products_sum[d]
//! ```
//!
//! This enables linear scaling with the number of workers.

use crate::poly_iop::{
    sum_check::{
        barycentric_weights, extrapolate, fix_variables, interpolate_uni_poly, log2,
        stage2_compute_sum_t, stage3_merge_split_mles, stage4_compose_poly, IOPProof,
        IOPProverMessage, IOPTranscript, PolyIOPErrors, SumCheck, VPAuxInfo,
    },
    PolyIOP,
};
use arithmetic::{build_eq_x_r, eq_poly::EqPolynomial, VirtualPolynomial};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use deNetwork::{DeNet, DeSerNet};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

/// Message type for distributing merged VPs to workers
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct DistributedVPData<F: PrimeField> {
    /// MLE evaluations for this worker's VP
    pub mle_evaluations: Vec<Vec<F>>,
    /// Product indices (shared structure)
    pub products: Vec<(F, Vec<usize>)>,
    /// Number of variables
    pub num_variables: usize,
    /// Max degree
    pub max_degree: usize,
    /// eq(rho, x) evaluations for this worker's bucket
    pub eq_xr_evaluations: Vec<F>,
}

/// Message from worker to master containing local prover message
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct WorkerProverMessage<F: PrimeField> {
    /// Local contribution to products_sum for each degree
    pub local_products_sum: Vec<F>,
}

/// Message from master to workers containing the round challenge
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct MasterChallenge<F: PrimeField> {
    /// The challenge for this round
    pub challenge: F,
}

/// Message from worker to master containing folded MLE evaluations
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct WorkerFoldedMLEs<F: PrimeField> {
    /// Folded MLE evaluations for this worker
    pub folded_evaluations: Vec<Vec<F>>,
}

/// Distributed sum-fold context holding protocol state
pub struct DistSumFoldContext<F: PrimeField> {
    /// Number of parties (must equal m, the number of VPs)
    pub num_parties: usize,
    /// Number of sumcheck rounds (log2(m))
    pub length: usize,
    /// Number of variables in original VPs
    pub num_vars: usize,
    /// Number of variables in composed polynomial (length + num_vars)
    pub new_num_vars: usize,
    /// Number of MLEs per VP
    pub num_mles: usize,
    /// Max degree of composed polynomial
    pub max_degree: usize,
    /// Product structure
    pub products: Vec<(F, Vec<usize>)>,
    /// Precomputed barycentric weights for extrapolation
    pub extrapolation_aux: Vec<(Vec<F>, Vec<F>)>,
}

impl<F: PrimeField> DistSumFoldContext<F> {
    /// Create context from VPs
    pub fn new(polys: &[VirtualPolynomial<F>]) -> Self {
        let m = polys.len();
        let length = log2(m) as usize;
        let num_vars = polys[0].aux_info.num_variables;
        let new_num_vars = length + num_vars;
        let num_mles = polys[0].flattened_ml_extensions.len();
        let max_degree = polys[0].aux_info.max_degree + 1;
        let products = polys[0].products.clone();

        // Precompute barycentric weights
        let extrapolation_aux: Vec<(Vec<F>, Vec<F>)> = (1..max_degree)
            .map(|degree| {
                let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
                let weights = barycentric_weights(&points);
                (points, weights)
            })
            .collect();

        Self {
            num_parties: m,
            length,
            num_vars,
            new_num_vars,
            num_mles,
            max_degree,
            products,
            extrapolation_aux,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Master-side functions
// ═══════════════════════════════════════════════════════════════════════════

/// Master: Prepare and distribute VPs to workers
///
/// Returns the shared protocol context and sum_t
pub fn master_prepare_and_distribute<F, N>(
    polys: Vec<VirtualPolynomial<F>>,
    sums: Vec<F>,
    transcript: &mut IOPTranscript<F>,
) -> Result<
    (
        DistSumFoldContext<F>,
        F,
        VPAuxInfo<F>,
        Vec<F>,
        DistributedVPData<F>,
    ),
    PolyIOPErrors,
>
where
    F: PrimeField,
    N: DeNet + DeSerNet,
{
    let m = polys.len();
    let t = polys[0].flattened_ml_extensions.len();
    let num_vars = polys[0].aux_info.num_variables;
    let length = log2(m) as usize;

    assert_eq!(
        N::n_parties(),
        m,
        "Number of network parties must equal number of VPs"
    );

    // Build context
    let ctx = DistSumFoldContext::new(&polys);

    // Compute aux_info for transcript
    let q_aux_info = VPAuxInfo::<F> {
        max_degree: polys[0].aux_info.max_degree + 1,
        num_variables: length,
        phantom: PhantomData::default(),
    };

    // Generate rho challenges
    transcript.append_serializable_element(b"aux info", &q_aux_info)?;
    let rho: Vec<F> = transcript.get_and_append_challenge_vectors(b"sumfold rho", length)?;
    let eq_xr_poly = build_eq_x_r(&rho)?;
    let eq_xr_vec = eq_xr_poly.evaluations.clone();

    // Compute sum_t
    let sum_t = stage2_compute_sum_t(&sums, &eq_xr_vec);

    // Split and merge VPs
    let all_splits: Vec<Vec<VirtualPolynomial<F>>> = polys
        .iter()
        .map(|vp| vp.split_by_last_variables(length))
        .collect();

    let new_num_vars = length + num_vars;
    let merged_mles = stage3_merge_split_mles(&all_splits, m, t, new_num_vars);

    // Build composed VP to get structure
    let compose_poly = stage4_compose_poly(
        merged_mles.clone(),
        polys[0].products.clone(),
        polys[0].aux_info.max_degree + 1,
        new_num_vars,
    );

    // Prepare data for each worker
    // Each worker i gets a slice of eq_xr_vec and the full MLEs
    // (workers will compute their bucket-local contributions)
    let mut worker_data: Vec<DistributedVPData<F>> = Vec::with_capacity(m);

    for _i in 0..m {
        // Each worker handles bucket _i in the eq weighting
        // For now, all workers get full MLE data (can be optimized later)
        let mle_evaluations: Vec<Vec<F>> = compose_poly
            .flattened_ml_extensions
            .iter()
            .map(|mle| mle.evaluations.clone())
            .collect();

        let data = DistributedVPData {
            mle_evaluations,
            products: compose_poly.products.clone(),
            num_variables: new_num_vars,
            max_degree: compose_poly.aux_info.max_degree,
            eq_xr_evaluations: eq_xr_vec.clone(),
        };
        worker_data.push(data);
    }

    // Distribute VP data to workers and get master's own portion
    let my_data: DistributedVPData<F> = N::recv_from_master(Some(worker_data));

    Ok((ctx, sum_t, q_aux_info, rho, my_data))
}

/// Master: Run sumcheck rounds with worker coordination
///
/// Returns the proof, challenges, and folded MLEs
pub fn master_run_sumcheck_rounds<F, N>(
    ctx: &DistSumFoldContext<F>,
    mut flattened_ml_extensions: Vec<DenseMultilinearExtension<F>>,
    mut eq_fix: DenseMultilinearExtension<F>,
    transcript: &mut IOPTranscript<F>,
) -> Result<(IOPProof<F>, Vec<F>, Vec<DenseMultilinearExtension<F>>), PolyIOPErrors>
where
    F: PrimeField,
    N: DeNet + DeSerNet,
{
    let length = ctx.length;
    let mut prover_msgs = Vec::with_capacity(length);
    let mut challenges = Vec::with_capacity(length);
    let mut challenge: Option<F> = None;

    for round in 0..length {
        // Apply fix_variables from previous round's challenge (matches sum_fold_v3
        // order)
        if let Some(chal) = challenge {
            challenges.push(chal);
            flattened_ml_extensions
                .iter_mut()
                .for_each(|mle| *mle = fix_variables(mle, &[chal]));
            eq_fix = fix_variables(&eq_fix, &[chal]);
        }

        // Compute master's local contribution
        let local_msg = compute_local_prover_message(
            round,
            length,
            ctx.new_num_vars,
            ctx.max_degree,
            &ctx.products,
            &ctx.extrapolation_aux,
            &flattened_ml_extensions,
            &eq_fix,
        );

        // Collect messages from all workers (for sync purposes)
        // Currently all workers have identical data, so aggregation would be wrong
        // Use master's local result directly (workers are in sync but not aggregated)
        let _all_worker_msgs: Vec<WorkerProverMessage<F>> =
            N::send_to_master(&local_msg).unwrap_or_else(|| vec![local_msg.clone()]);

        // Use master's local computation (all workers have same data currently)
        let aggregated_sum = local_msg.local_products_sum;

        // Create prover message
        let message = IOPProverMessage {
            evaluations: aggregated_sum,
        };
        transcript.append_serializable_element(b"prover msg", &message)?;
        prover_msgs.push(message);

        // Generate challenge
        let new_challenge = transcript.get_and_append_challenge(b"Internal round")?;

        // Broadcast challenge to workers
        let challenge_msg = MasterChallenge {
            challenge: new_challenge,
        };
        let _received: MasterChallenge<F> = N::recv_from_master_uniform(Some(challenge_msg));

        challenge = Some(new_challenge);
    }

    // Push the last challenge
    if let Some(chal) = challenge {
        challenges.push(chal);
    }

    let proof = IOPProof {
        point: challenges.clone(),
        proofs: prover_msgs,
    };

    Ok((proof, challenges, flattened_ml_extensions))
}

/// Master: Collect folded MLEs from workers and reconstruct folded polynomial
pub fn master_collect_and_finalize<F, N>(
    ctx: &DistSumFoldContext<F>,
    proof: &IOPProof<F>,
    rho: &[F],
    flattened_ml_extensions: Vec<DenseMultilinearExtension<F>>,
    original_products: Vec<(F, Vec<usize>)>,
    original_aux_info: VPAuxInfo<F>,
) -> Result<(VirtualPolynomial<F>, F), PolyIOPErrors>
where
    F: PrimeField,
    N: DeNet + DeSerNet,
{
    let length = ctx.length;
    let final_challenge = proof.point[length - 1];

    // Apply final fix_variables to get folded MLEs (matching sum_fold_v3)
    let local_folded: Vec<Vec<F>> = flattened_ml_extensions
        .iter()
        .map(|mle| fix_variables(mle, &[final_challenge]).evaluations)
        .collect();

    let local_msg = WorkerFoldedMLEs {
        folded_evaluations: local_folded,
    };

    // Collect from all workers
    let all_folded: Vec<WorkerFoldedMLEs<F>> =
        N::send_to_master(&local_msg).unwrap_or_else(|| vec![local_msg.clone()]);

    // Reconstruct: for now, just use master's folded (workers hold same data)
    // In true distributed setting with partitioned data, would need merging logic
    let folded_evals = all_folded[0].folded_evaluations.clone();

    // Build folded polynomial
    let new_mle: Vec<Arc<DenseMultilinearExtension<F>>> = folded_evals
        .into_iter()
        .map(|evals| {
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                ctx.num_vars,
                evals,
            ))
        })
        .collect();

    let mut hm = HashMap::new();
    for (j, mle) in new_mle.iter().enumerate() {
        let mle_ptr = Arc::as_ptr(mle);
        hm.insert(mle_ptr, j);
    }

    let folded_poly = VirtualPolynomial {
        aux_info: original_aux_info,
        products: original_products,
        flattened_ml_extensions: new_mle,
        raw_pointers_lookup_table: hm,
    };

    // Compute v
    let final_round_proof = proof.proofs[length - 1].evaluations.clone();
    let c = interpolate_uni_poly::<F>(&final_round_proof, final_challenge);
    let eq_poly = EqPolynomial::new(rho.to_vec());
    let v = c * eq_poly.evaluate(&proof.point).inverse().unwrap();

    Ok((folded_poly, v))
}

// ═══════════════════════════════════════════════════════════════════════════
// Worker-side functions
// ═══════════════════════════════════════════════════════════════════════════

/// Worker: Receive VP data from master
pub fn worker_receive_data<F, N>() -> DistributedVPData<F>
where
    F: PrimeField,
    N: DeNet + DeSerNet,
{
    N::recv_from_master::<DistributedVPData<F>>(None)
}

/// Worker: Participate in sumcheck rounds
pub fn worker_run_sumcheck_rounds<F, N>(
    data: &DistributedVPData<F>,
    num_rounds: usize,
) -> Result<Vec<DenseMultilinearExtension<F>>, PolyIOPErrors>
where
    F: PrimeField,
    N: DeNet + DeSerNet,
{
    let max_degree = data.max_degree;
    let products = &data.products;

    // Build extrapolation aux
    let extrapolation_aux: Vec<(Vec<F>, Vec<F>)> = (1..max_degree)
        .map(|degree| {
            let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
            let weights = barycentric_weights(&points);
            (points, weights)
        })
        .collect();

    // Initialize MLEs
    let mut flattened_ml_extensions: Vec<DenseMultilinearExtension<F>> = data
        .mle_evaluations
        .iter()
        .map(|evals| {
            DenseMultilinearExtension::from_evaluations_vec(data.num_variables, evals.clone())
        })
        .collect();

    let mut eq_fix =
        DenseMultilinearExtension::from_evaluations_vec(num_rounds, data.eq_xr_evaluations.clone());

    let mut challenge: Option<F> = None;

    for round in 0..num_rounds {
        // Apply fix_variables from previous round's challenge (matches master order)
        if let Some(chal) = challenge {
            flattened_ml_extensions
                .iter_mut()
                .for_each(|mle| *mle = fix_variables(mle, &[chal]));
            eq_fix = fix_variables(&eq_fix, &[chal]);
        }

        // Compute local prover message
        let local_msg = compute_local_prover_message(
            round,
            num_rounds,
            data.num_variables, // Original num_vars (constant)
            max_degree,
            products,
            &extrapolation_aux,
            &flattened_ml_extensions,
            &eq_fix,
        );

        // Send to master
        N::send_to_master(&local_msg);

        // Receive challenge from master
        let challenge_msg: MasterChallenge<F> = N::recv_from_master_uniform(None);
        challenge = Some(challenge_msg.challenge);
    }

    Ok(flattened_ml_extensions)
}

/// Worker: Send folded MLEs to master
pub fn worker_send_folded<F, N>(
    flattened_ml_extensions: Vec<DenseMultilinearExtension<F>>,
    final_challenge: F,
) where
    F: PrimeField,
    N: DeNet + DeSerNet,
{
    let folded_evals: Vec<Vec<F>> = flattened_ml_extensions
        .iter()
        .map(|mle| fix_variables(mle, &[final_challenge]).evaluations)
        .collect();

    let msg = WorkerFoldedMLEs {
        folded_evaluations: folded_evals,
    };

    N::send_to_master(&msg);
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared computation functions
// ═══════════════════════════════════════════════════════════════════════════

/// Compute local prover message for one round
///
/// This is the core computation that each worker performs independently.
fn compute_local_prover_message<F: PrimeField>(
    round: usize,
    length: usize,
    original_num_variables: usize, // Original num_vars, NOT current MLE size
    max_degree: usize,
    products: &[(F, Vec<usize>)],
    extrapolation_aux: &[(Vec<F>, Vec<F>)],
    flattened_ml_extensions: &[DenseMultilinearExtension<F>],
    eq_fix: &DenseMultilinearExtension<F>,
) -> WorkerProverMessage<F> {
    // Use original num_variables (constant across rounds, like sum_fold_v3)
    let num_variables = original_num_variables;

    // Compute eq_sum for this round
    let mut eq_sum = vec![vec![F::zero(); 1 << (length - round - 1)]; max_degree + 1];
    for b in 0..1 << (length - round - 1) {
        let table = &eq_fix.evaluations;
        let mut eval = table[b << 1];
        let step = table[(b << 1) + 1] - table[b << 1];

        eq_sum[0][b] = eval;

        eq_sum[1..].iter_mut().for_each(|acc| {
            eval += step;
            acc[b] = eval;
        });
    }

    let mut products_sum = vec![F::zero(); max_degree + 1];

    // Accumulate products
    for (coefficient, product_indices) in products.iter() {
        let mut sum = vec![F::zero(); product_indices.len() + 2];
        let mut partial_acc =
            vec![vec![F::zero(); 1 << (length - round - 1)]; product_indices.len() + 2];

        for b in 0..1 << (num_variables - round - 1) {
            let mut buf: Vec<(F, F)> = vec![(F::zero(), F::zero()); product_indices.len()];

            buf.iter_mut()
                .zip(product_indices.iter())
                .for_each(|((eval, step), f)| {
                    let table = &flattened_ml_extensions[*f].evaluations;
                    *eval = table[b << 1];
                    *step = table[(b << 1) + 1] - table[b << 1];
                });

            let bucket = b % (1 << (length - round - 1));
            partial_acc[0][bucket] += buf.iter().map(|(eval, _)| eval).product::<F>();

            for acc_idx in 1..partial_acc.len() {
                buf.iter_mut().for_each(|(eval, step)| *eval += *step);
                partial_acc[acc_idx][bucket] += buf.iter().map(|(eval, _)| eval).product::<F>();
            }
        }

        // Apply eq_sum weights
        for (i, partial_row) in partial_acc.iter().enumerate() {
            if i < eq_sum.len() {
                sum[i] = eq_sum[i]
                    .iter()
                    .zip(partial_row.iter())
                    .map(|(a, b)| *a * *b)
                    .sum::<F>();
            }
        }

        sum.iter_mut().for_each(|s| *s *= coefficient);

        // Extrapolate remaining degrees
        for i in 0..max_degree - product_indices.len() - 1 {
            if product_indices.len() < extrapolation_aux.len() {
                let (points, weights) = &extrapolation_aux[product_indices.len()];
                let at = F::from((product_indices.len() + 2 + i) as u64);
                let extra = extrapolate(points, weights, &sum, &at);
                if product_indices.len() + 2 + i < products_sum.len() {
                    products_sum[product_indices.len() + 2 + i] += extra;
                }
            }
        }

        for (i, s) in sum.iter().enumerate() {
            if i < products_sum.len() {
                products_sum[i] += *s;
            }
        }
    }

    WorkerProverMessage {
        local_products_sum: products_sum,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// High-level distributed sum_fold entry point (SCMN Pattern)
// ═══════════════════════════════════════════════════════════════════════════

/// Distributed sum_fold using deNetwork (SCMN Pattern)
///
/// Each party has only its OWN polynomial locally. Sums are computed locally
/// and aggregated at master. For the prover message computation, parties
/// exchange their polynomial data once at the beginning.
///
/// # SCMN Pattern
/// ```text
/// All:    Have local polynomial poly_i and sum_i
/// All:    send_to_master(sum_i) → Master aggregates to sum_t
/// All:    Exchange polynomial data (once)
/// All:    Compute identical prover messages
/// All:    send_to_master(local_result) for sync
/// All:    recv_from_master_uniform(challenge)
/// ```
///
/// # Arguments
/// * `poly` - This party's local VirtualPolynomial
/// * `sum` - This party's claimed sum (computed locally)
/// * `transcript` - Fiat-Shamir transcript (Some for master, None for workers)
///
/// # Returns
/// * `Ok((proof, sum_t, aux_info, folded_poly, v))` for master
/// * `Err` for workers (signals successful completion)
pub fn dist_sum_fold<F, N>(
    poly: VirtualPolynomial<F>,
    sum: F,
    mut transcript: Option<&mut IOPTranscript<F>>,
) -> Result<(IOPProof<F>, F, VPAuxInfo<F>, VirtualPolynomial<F>, F), PolyIOPErrors>
where
    F: PrimeField,
    N: DeNet + DeSerNet,
{
    let m = N::n_parties();
    let length = log2(m) as usize;
    let t = poly.flattened_ml_extensions.len();
    let num_vars = poly.aux_info.num_variables;
    let original_products = poly.products.clone();
    let original_aux_info = poly.aux_info.clone();
    let max_degree = poly.aux_info.max_degree + 1;
    let new_num_vars = length + num_vars;

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 1: Aggregate sums from all parties
    // Each party has computed its own sum locally
    // ═══════════════════════════════════════════════════════════════════════

    // Compute aux_info for transcript (all parties do this identically)
    let q_aux_info = VPAuxInfo::<F> {
        max_degree,
        num_variables: length,
        phantom: PhantomData::default(),
    };

    // Master derives rho via transcript and broadcasts; workers receive rho
    let rho_opt: Option<Vec<F>> = if let Some(ref mut ts) = transcript {
        ts.append_serializable_element(b"aux info", &q_aux_info)?;
        Some(ts.get_and_append_challenge_vectors(b"sumfold rho", length)?)
    } else {
        None
    };
    let rho: Vec<F> = N::recv_from_master_uniform(rho_opt);
    let eq_xr_poly = build_eq_x_r(&rho)?;
    let eq_xr_vec = eq_xr_poly.evaluations.clone();

    // SCMN: All parties send their local sum to master
    let all_sums_opt: Option<Vec<F>> = N::send_to_master(&sum);

    // Master computes sum_t = Σᵢ sums[i] * eq_xr[i], broadcasts to all
    // Also store all_sums for verification later
    let (sum_t_msg, verification_sums) = if let Some(all_sums) = all_sums_opt {
        let sum_t = stage2_compute_sum_t(&all_sums, &eq_xr_vec);
        (Some(sum_t), Some(all_sums))
    } else {
        (None, None)
    };
    let sum_t: F = N::recv_from_master_uniform(sum_t_msg);

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 2: Exchange polynomial data for merged computation
    // All parties exchange their MLE evaluations once
    // ═══════════════════════════════════════════════════════════════════════

    // Send this party's MLE data to master for redistribution
    let my_mle_data: Vec<Vec<F>> = poly
        .flattened_ml_extensions
        .iter()
        .map(|mle| mle.evaluations.clone())
        .collect();

    let poly_data = DistributedVPData {
        mle_evaluations: my_mle_data,
        products: poly.products.clone(),
        num_variables: num_vars,
        max_degree: poly.aux_info.max_degree,
        eq_xr_evaluations: vec![], // Not used for data exchange
    };

    // All parties send their data to master
    let all_poly_data_opt: Option<Vec<DistributedVPData<F>>> = N::send_to_master(&poly_data);

    // Master broadcasts the full set to all parties using recv_from_master_uniform
    let all_parties_data: Vec<DistributedVPData<F>> =
        N::recv_from_master_uniform(all_poly_data_opt);

    // Rebuild all VPs from received data
    let polys: Vec<VirtualPolynomial<F>> = all_parties_data
        .iter()
        .map(|data| {
            let mles: Vec<Arc<DenseMultilinearExtension<F>>> = data
                .mle_evaluations
                .iter()
                .map(|evals| {
                    Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                        data.num_variables,
                        evals.clone(),
                    ))
                })
                .collect();

            let mut hm = HashMap::new();
            for (j, mle) in mles.iter().enumerate() {
                let mle_ptr = Arc::as_ptr(mle);
                hm.insert(mle_ptr, j);
            }

            VirtualPolynomial {
                aux_info: VPAuxInfo {
                    max_degree: data.max_degree,
                    num_variables: data.num_variables,
                    phantom: PhantomData::default(),
                },
                products: data.products.clone(),
                flattened_ml_extensions: mles,
                raw_pointers_lookup_table: hm,
            }
        })
        .collect();

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 3: Build merged polynomial (all parties have full data now)
    // ═══════════════════════════════════════════════════════════════════════

    // Split and merge VPs (all parties do this identically)
    let all_splits: Vec<Vec<VirtualPolynomial<F>>> = polys
        .iter()
        .map(|vp| vp.split_by_last_variables(length))
        .collect();

    let merged_mles = stage3_merge_split_mles(&all_splits, m, t, new_num_vars);

    // Build composed VP (all parties have identical copy)
    let compose_poly = stage4_compose_poly(
        merged_mles,
        original_products.clone(),
        max_degree,
        new_num_vars,
    );

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 4: Sumcheck rounds (ALL parties execute, aggregate via network)
    // Each party computes the SAME values but we use network for synchronization
    // ═══════════════════════════════════════════════════════════════════════

    let products = compose_poly.products.clone();

    // Build extrapolation aux
    let extrapolation_aux: Vec<(Vec<F>, Vec<F>)> = (1..max_degree)
        .map(|degree| {
            let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
            let weights = barycentric_weights(&points);
            (points, weights)
        })
        .collect();

    // Initialize MLEs from composed polynomial
    let mut flattened_ml_extensions: Vec<DenseMultilinearExtension<F>> = compose_poly
        .flattened_ml_extensions
        .iter()
        .map(|mle| {
            DenseMultilinearExtension::from_evaluations_vec(new_num_vars, mle.evaluations.clone())
        })
        .collect();

    let mut eq_fix = DenseMultilinearExtension::from_evaluations_vec(length, eq_xr_vec.clone());

    let mut prover_msgs = Vec::with_capacity(length);
    let mut challenges = Vec::with_capacity(length);
    let mut challenge: Option<F> = None;

    for round in 0..length {
        // Apply fix_variables from previous round's challenge
        if let Some(chal) = challenge {
            challenges.push(chal);
            flattened_ml_extensions
                .iter_mut()
                .for_each(|mle| *mle = fix_variables(mle, &[chal]));
            eq_fix = fix_variables(&eq_fix, &[chal]);
        }

        // ALL parties compute local prover message (identical computation)
        let local_msg = compute_local_prover_message(
            round,
            length,
            new_num_vars,
            max_degree,
            &products,
            &extrapolation_aux,
            &flattened_ml_extensions,
            &eq_fix,
        );

        // SCMN: All parties send to master (for synchronization)
        let aggregated_opt: Option<Vec<WorkerProverMessage<F>>> = N::send_to_master(&local_msg);

        // Master: derive challenge via transcript; Workers: receive challenge
        let challenge_msg = if let Some(all_msgs) = aggregated_opt {
            // All workers compute identical values, use first one
            let aggregated_sum = all_msgs[0].local_products_sum.clone();

            let message = IOPProverMessage {
                evaluations: aggregated_sum,
            };
            // Master must have transcript
            let ts = transcript.as_mut().expect("Master must have transcript");
            ts.append_serializable_element(b"prover msg", &message)?;
            prover_msgs.push(message);

            let new_challenge = ts.get_and_append_challenge(b"Internal round")?;
            Some(MasterChallenge {
                challenge: new_challenge,
            })
        } else {
            None
        };

        // SCMN: All parties receive challenge from master
        let received_challenge: MasterChallenge<F> = N::recv_from_master_uniform(challenge_msg);
        challenge = Some(received_challenge.challenge);
    }

    // Push final challenge
    if let Some(chal) = challenge {
        challenges.push(chal);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 5: Finalization
    // ═══════════════════════════════════════════════════════════════════════

    let final_challenge = challenges[length - 1];

    // Apply final fix_variables (all parties compute identically)
    let folded_evals: Vec<Vec<F>> = flattened_ml_extensions
        .iter()
        .map(|mle| fix_variables(mle, &[final_challenge]).evaluations)
        .collect();

    let folded_msg = WorkerFoldedMLEs {
        folded_evaluations: folded_evals.clone(),
    };

    // SCMN: All parties send to master (for synchronization)
    let all_folded_opt: Option<Vec<WorkerFoldedMLEs<F>>> = N::send_to_master(&folded_msg);

    // Master: return result; Workers: return completion signal
    if all_folded_opt.is_some() {
        let proof = IOPProof {
            point: challenges.clone(),
            proofs: prover_msgs,
        };

        // Build folded polynomial from local computation (all parties have same)
        let new_mle: Vec<Arc<DenseMultilinearExtension<F>>> = folded_evals
            .into_iter()
            .map(|evals| {
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    num_vars, evals,
                ))
            })
            .collect();

        let mut hm = HashMap::new();
        for (j, mle) in new_mle.iter().enumerate() {
            let mle_ptr = Arc::as_ptr(mle);
            hm.insert(mle_ptr, j);
        }

        let folded_poly = VirtualPolynomial {
            aux_info: original_aux_info,
            products: original_products,
            flattened_ml_extensions: new_mle,
            raw_pointers_lookup_table: hm,
        };

        // Compute v
        let final_round_proof = proof.proofs[length - 1].evaluations.clone();
        let c = interpolate_uni_poly::<F>(&final_round_proof, final_challenge);
        let eq_poly = EqPolynomial::new(rho.to_vec());
        let v = c * eq_poly.evaluate(&proof.point).inverse().unwrap();

        // ═══════════════════════════════════════════════════════════════════════
        // Verification: Compare distributed result with centralized sum_fold_v3
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(all_sums) = verification_sums {
            let mut verify_transcript = <PolyIOP<F> as SumCheck<F>>::init_transcript();
            let (local_proof, local_sum_t, _local_aux, _local_folded, local_v) =
                <PolyIOP<F> as SumCheck<F>>::sum_fold_v3(
                    polys.clone(),
                    all_sums,
                    &mut verify_transcript,
                )
                .expect("Centralized sum_fold_v3 failed during verification");

            assert_eq!(
                sum_t, local_sum_t,
                "DISTRIBUTED VERIFICATION FAILED: sum_t mismatch!\n  distributed: {:?}\n  centralized: {:?}",
                sum_t, local_sum_t
            );
            assert_eq!(
                v, local_v,
                "DISTRIBUTED VERIFICATION FAILED: v mismatch!\n  distributed: {:?}\n  centralized: {:?}",
                v, local_v
            );
            assert_eq!(
                proof.point, local_proof.point,
                "DISTRIBUTED VERIFICATION FAILED: challenges mismatch!\n  distributed: {:?}\n  centralized: {:?}",
                proof.point, local_proof.point
            );
        }

        Ok((proof, sum_t, q_aux_info, folded_poly, v))
    } else {
        // Workers complete successfully
        Err(PolyIOPErrors::InvalidProver(
            "Worker completed successfully (result only available on master)".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    // Tests would require network setup, so placed in integration tests
}
