//! Distributed Multilinear KZG Polynomial Commitment Scheme.
//!
//! Ported from HyperPianist's DeMkzg implementation.
//! Provides distributed commit, open, multi-open, verify, and batch-verify
//! operations for multilinear polynomials using deNetwork coordination.

use super::batching::batch_verify_internal;
#[cfg(feature = "distributed")]
use super::batching::d_multi_open_internal;
use crate::pcs::{
    multilinear_kzg::{
        srs::{MultilinearProverParam, MultilinearUniversalParams, MultilinearVerifierParam},
        MultilinearKzgPCS, MultilinearKzgProof,
    },
    prelude::Commitment,
    PCSError, PolynomialCommitmentScheme, StructuredReferenceString,
};
use arithmetic::{evaluate_opt, math::Math, unsafe_allocate_zero_vec, DenseMultilinearExtension};
use ark_ec::{
    pairing::Pairing,
    scalar_mul::{fixed_base::FixedBase, variable_base::VariableBaseMSM},
    AffineRepr, CurveGroup,
};
use ark_ff::PrimeField;
use ark_poly::MultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    end_timer, format, marker::PhantomData, rand::Rng, start_timer, string::ToString, sync::Arc,
    vec, vec::Vec, One, Zero,
};
use std::ops::Mul;
use transcript::IOPTranscript;

#[cfg(feature = "distributed")]
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use crate::pcs::multilinear_kzg::batching::BatchProof;

/// Distributed Multilinear KZG commitment scheme.
///
/// Wraps the standard MultilinearKzgPCS with distributed operations
/// coordinated through deNetwork.
pub struct DeMkzg<E: Pairing> {
    _phantom: PhantomData<E>,
}

/// Data sent from sub-provers to master during distributed open.
#[derive(Default, Clone, CanonicalSerialize, CanonicalDeserialize, PartialEq, Eq, Debug)]
pub struct SentToMasterData<E: Pairing> {
    /// Scalar field elements (e.g., final MLE evaluations)
    pub f_vec: Vec<E::ScalarField>,
    /// G1 affine points (e.g., quotient commitments)
    pub g1_vec: Vec<E::G1Affine>,
}

impl<E: Pairing> SentToMasterData<E> {
    pub(super) fn new() -> Self {
        Self {
            f_vec: Vec::new(),
            g1_vec: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-distributed (local) operations — delegate to MultilinearKzgPCS
// ═══════════════════════════════════════════════════════════════════════════

impl<E: Pairing> DeMkzg<E> {
    /// Generate SRS for testing (delegates to MultilinearKzgPCS).
    pub fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        num_vars: usize,
    ) -> Result<MultilinearUniversalParams<E>, PCSError> {
        MultilinearKzgPCS::<E>::gen_srs_for_testing(rng, num_vars)
    }

    /// Trim SRS to specific size (delegates to MultilinearKzgPCS).
    pub fn trim(
        srs: &MultilinearUniversalParams<E>,
        num_vars: usize,
    ) -> Result<(MultilinearProverParam<E>, MultilinearVerifierParam<E>), PCSError> {
        MultilinearKzgPCS::<E>::trim(srs, None, Some(num_vars))
    }

    /// Standard (non-distributed) commit using MultilinearKzgPCS.
    pub fn commit(
        prover_param: &MultilinearProverParam<E>,
        poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    ) -> Result<Commitment<E>, PCSError> {
        MultilinearKzgPCS::<E>::commit(prover_param, poly)
    }

    /// Standard (non-distributed) open using MultilinearKzgPCS.
    pub fn open(
        prover_param: &MultilinearProverParam<E>,
        poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
        point: &[E::ScalarField],
    ) -> Result<(MultilinearKzgProof<E>, E::ScalarField), PCSError> {
        MultilinearKzgPCS::<E>::open(prover_param, poly, &point.to_vec())
    }

    /// Standard verify using MultilinearKzgPCS.
    pub fn verify(
        verifier_param: &MultilinearVerifierParam<E>,
        commitment: &Commitment<E>,
        point: &[E::ScalarField],
        value: &E::ScalarField,
        proof: &MultilinearKzgProof<E>,
    ) -> Result<bool, PCSError> {
        verify_internal(verifier_param, commitment, point, value, proof)
    }

    /// Batch verify using SumCheck-based batching.
    pub fn batch_verify(
        verifier_param: &MultilinearVerifierParam<E>,
        commitments: &[Commitment<E>],
        points: &[Vec<E::ScalarField>],
        batch_proof: &BatchProof<E, MultilinearKzgPCS<E>>,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        batch_verify_internal::<E, MultilinearKzgPCS<E>>(
            verifier_param,
            commitments,
            points,
            batch_proof,
            transcript,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Distributed operations — require `distributed` feature
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "distributed")]
impl<E: Pairing> DeMkzg<E> {
    /// Distributed commit: each party commits to its local polynomial slice,
    /// then aggregates to master.
    ///
    /// # Returns
    /// * `Ok(Some(commitment))` on master node
    /// * `Ok(None)` on non-master nodes
    pub fn d_commit(
        prover_param: &MultilinearProverParam<E>,
        poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    ) -> Result<Option<Commitment<E>>, PCSError> {
        let commit_timer = start_timer!(|| "DeMkzg::d_commit");

        let sub_prover_id = Net::party_id();
        let num_parties = Net::n_parties();
        let m = num_parties.log_2();
        let total_nv = m + poly.num_vars;
        if prover_param.num_vars < total_nv {
            return Err(PCSError::InvalidParameters(format!(
                "MlE length ({}) exceeds param limit ({})",
                total_nv, prover_param.num_vars
            )));
        }
        let ignored = prover_param.num_vars - total_nv;

        let sub_g_powers_size = 1usize << poly.num_vars;
        let msm_timer = start_timer!(|| format!("msm of size {}", sub_g_powers_size));
        let start = sub_prover_id * sub_g_powers_size;
        let end = start + sub_g_powers_size;
        let sub_comm: E::G1Affine = E::G1MSM::msm_unchecked(
            &prover_param.powers_of_g[ignored].evals[start..end],
            &poly.evaluations,
        )
        .into();
        end_timer!(msm_timer);

        let sub_comms = Net::send_to_master(&sub_comm);
        end_timer!(commit_timer);

        if Net::am_master() {
            let agg_timer = start_timer!(|| "aggregation");
            let comm: E::G1Affine = sub_comms
                .unwrap()
                .iter()
                .fold(E::G1::zero(), |acc, &x| acc + x)
                .into();
            end_timer!(agg_timer);
            Ok(Some(Commitment(comm)))
        } else {
            Ok(None)
        }
    }

    /// Batch distributed commit for multiple polynomials.
    pub fn batch_d_commit(
        prover_param: &MultilinearProverParam<E>,
        polys: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    ) -> Result<Vec<Option<Commitment<E>>>, PCSError> {
        let commit_timer = start_timer!(|| "DeMkzg::batch_d_commit");

        let sub_prover_id = Net::party_id();
        let num_parties = Net::n_parties();
        let m = num_parties.log_2();
        let total_nv = m + polys[0].num_vars;
        if prover_param.num_vars < total_nv {
            return Err(PCSError::InvalidParameters(format!(
                "MlE length ({}) exceeds param limit ({})",
                total_nv, prover_param.num_vars
            )));
        }
        let ignored = prover_param.num_vars - total_nv;

        let sub_g_powers_size = 1usize << polys[0].num_vars;
        let start = sub_prover_id * sub_g_powers_size;
        let end = start + sub_g_powers_size;

        let sub_comms: Vec<E::G1Affine> = polys
            .iter()
            .map(|poly| {
                E::G1MSM::msm_unchecked(
                    &prover_param.powers_of_g[ignored].evals[start..end],
                    &poly.evaluations,
                )
                .into()
            })
            .collect();

        let all_sub_comms = Net::send_to_master(&sub_comms);
        end_timer!(commit_timer);

        if Net::am_master() {
            let agg_timer = start_timer!(|| "aggregation");
            let all_sub_comms = all_sub_comms.unwrap();
            let comms: Vec<Option<Commitment<E>>> = (0..polys.len())
                .map(|i| {
                    Some(Commitment(
                        all_sub_comms
                            .iter()
                            .map(|sc| sc[i])
                            .fold(E::G1::zero(), |acc, x| acc + x)
                            .into(),
                    ))
                })
                .collect();
            end_timer!(agg_timer);
            Ok(comms)
        } else {
            Ok(vec![None; polys.len()])
        }
    }

    /// Distributed open: each party computes partial quotient commitments,
    /// then master aggregates and completes the remaining rounds.
    ///
    /// # Returns
    /// * `Ok(Some(proof))` on master
    /// * `Ok(None)` on non-master
    pub fn d_open(
        prover_param: &MultilinearProverParam<E>,
        polynomial: &DenseMultilinearExtension<E::ScalarField>,
        point: &[E::ScalarField],
    ) -> Result<Option<MultilinearKzgProof<E>>, PCSError> {
        d_open_internal::<E>(prover_param, polynomial, point)
    }

    /// Distributed multi-open: batch opening of multiple polynomials at
    /// multiple points using SumCheck-based batching.
    ///
    /// # Returns
    /// * `Ok(Some(batch_proof))` on master
    /// * `Ok(None)` on non-master
    pub fn d_multi_open(
        prover_param: &MultilinearProverParam<E>,
        polynomials: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
        points: &[Vec<E::ScalarField>],
        evals: &[E::ScalarField],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Option<BatchProof<E, MultilinearKzgPCS<E>>>, PCSError> {
        d_multi_open_internal::<E, MultilinearKzgPCS<E>>(
            prover_param,
            polynomials,
            points,
            evals,
            transcript,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal implementations
// ═══════════════════════════════════════════════════════════════════════════

/// Distributed open: two-phase protocol.
///
/// Phase 1 (all parties): Each party computes quotient polynomial MSMs for
/// its local evaluations (sub_nv rounds), producing partial proofs.
///
/// Phase 2 (master only): Master aggregates partial proofs and runs
/// log₂(K) additional rounds over the party-dimension values.
///
/// Total proof contains (sub_nv + log₂(K)) quotient commitments.
#[cfg(feature = "distributed")]
fn d_open_internal<E: Pairing>(
    prover_param: &MultilinearProverParam<E>,
    polynomial: &DenseMultilinearExtension<E::ScalarField>,
    point: &[E::ScalarField],
) -> Result<Option<MultilinearKzgProof<E>>, PCSError> {
    let open_timer = start_timer!(|| format!(
        "DeMkzg::d_open with {} variables",
        polynomial.num_vars
    ));

    let sub_prover_id = Net::party_id();
    let num_parties = Net::n_parties();
    let m = num_parties.log_2();
    let sub_nv = polynomial.num_vars;
    let total_nv = m + sub_nv;

    if prover_param.num_vars < total_nv {
        return Err(PCSError::InvalidParameters(format!(
            "MlE length ({}) exceeds param limit ({})",
            total_nv, prover_param.num_vars
        )));
    }
    // From g^eq(t,x) with total_nv - 1 variables
    let ignored = prover_param.num_vars - total_nv + 1;

    // Phase 1: Local computation
    let mut sub_qs_comms = Vec::new();
    let mut f = polynomial.to_evaluations();
    let mut q = unsafe_allocate_zero_vec::<E::ScalarField>(1 << (sub_nv - 1));
    let mut r = unsafe_allocate_zero_vec::<E::ScalarField>(1 << (sub_nv - 1));

    for (i, (&point_at_k, gi)) in point[..sub_nv]
        .iter()
        .zip(prover_param.powers_of_g[ignored..ignored + sub_nv].iter())
        .enumerate()
    {
        let k = sub_nv - 1 - i;
        let cur_dim = 1 << k;
        let ith_round = start_timer!(|| format!("{}-th round", i));

        let ith_round_eval = start_timer!(|| format!("{}-th round eval", i));
        #[cfg(feature = "parallel")]
        {
            q[..cur_dim]
                .par_iter_mut()
                .zip(r[..cur_dim].par_iter_mut())
                .enumerate()
                .for_each(|(b, (q, r))| {
                    *q = f[(b << 1) + 1] - f[b << 1];
                    *r = f[b << 1] + (*q * point_at_k);
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for b in 0..cur_dim {
                q[b] = f[(b << 1) + 1] - f[b << 1];
                r[b] = f[b << 1] + (q[b] * point_at_k);
            }
        }
        (r, f) = (f, r);
        end_timer!(ith_round_eval);

        // MSM over G1 — typically the bottleneck
        let msm_timer = start_timer!(|| format!("msm of size {} at round {}", cur_dim, i));
        let start_idx = sub_prover_id * cur_dim;
        let end_idx = start_idx + cur_dim;
        sub_qs_comms
            .push(E::G1MSM::msm_unchecked(&gi.evals[start_idx..end_idx], &q[..cur_dim]).into());
        end_timer!(msm_timer);

        end_timer!(ith_round);
    }

    // Package data for master
    let sub_data = SentToMasterData::<E> {
        f_vec: vec![f[0]],
        g1_vec: sub_qs_comms,
    };

    drop(q);
    drop(r);
    drop(f);

    let sub_data_vec = Net::send_to_master(&sub_data);
    end_timer!(open_timer);

    if Net::am_master() {
        // Phase 2: Master aggregates and completes
        let agg_timer = start_timer!(|| "aggregation + master rounds");
        let sub_data_vec = sub_data_vec.unwrap();

        // Aggregate quotient commitments from all parties
        #[cfg(feature = "parallel")]
        let mut proofs: Vec<E::G1Affine> = (0..sub_nv)
            .into_par_iter()
            .map(|i| {
                sub_data_vec
                    .iter()
                    .map(|pi_data| pi_data.g1_vec[i])
                    .fold(E::G1::zero(), |acc, x| acc + x)
                    .into()
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let mut proofs: Vec<E::G1Affine> = (0..sub_nv)
            .map(|i| {
                sub_data_vec
                    .iter()
                    .map(|pi_data| pi_data.g1_vec[i])
                    .fold(E::G1::zero(), |acc, x| acc + x)
                    .into()
            })
            .collect();

        // Collect final evaluations from all parties
        let mut f: Vec<E::ScalarField> = sub_data_vec
            .iter()
            .map(|pi_data| pi_data.f_vec[0])
            .collect();
        let mut q = unsafe_allocate_zero_vec::<E::ScalarField>(1 << (m - 1));
        let mut r = unsafe_allocate_zero_vec::<E::ScalarField>(1 << (m - 1));

        // Master completes the remaining log₂(K) rounds
        for (i, (&point_at_k, gi)) in point[sub_nv..]
            .iter()
            .zip(prover_param.powers_of_g[ignored + sub_nv..].iter())
            .enumerate()
        {
            let ith_round = start_timer!(|| format!("master {}-th round", i));
            let k = m - 1 - i;
            let cur_dim = 1 << k;

            #[cfg(feature = "parallel")]
            {
                q[..cur_dim]
                    .par_iter_mut()
                    .zip(r[..cur_dim].par_iter_mut())
                    .enumerate()
                    .for_each(|(b, (q, r))| {
                        *q = f[(b << 1) + 1] - f[b << 1];
                        *r = f[b << 1] + (*q * point_at_k);
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for b in 0..cur_dim {
                    q[b] = f[(b << 1) + 1] - f[b << 1];
                    r[b] = f[b << 1] + (q[b] * point_at_k);
                }
            }
            (r, f) = (f, r);

            let msm_timer = start_timer!(|| format!("msm of size {} at round {}", cur_dim, i));
            proofs.push(E::G1MSM::msm_unchecked(&gi.evals, &q[..cur_dim]).into());
            end_timer!(msm_timer);

            end_timer!(ith_round);
        }

        end_timer!(agg_timer);
        Ok(Some(MultilinearKzgProof { proofs }))
    } else {
        Ok(None)
    }
}

/// Internal verify implementation for multilinear KZG.
///
/// Checks that `value` is the evaluation at `point` of the polynomial
/// committed inside `commitment`, using num_var pairing products.
fn verify_internal<E: Pairing>(
    verifier_param: &MultilinearVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &MultilinearKzgProof<E>,
) -> Result<bool, PCSError> {
    let verify_timer = start_timer!(|| "verify");
    let num_var = point.len();

    if num_var > verifier_param.num_vars {
        return Err(PCSError::InvalidParameters(format!(
            "point length ({}) exceeds param limit ({})",
            num_var, verifier_param.num_vars
        )));
    }

    let prepare_inputs_timer = start_timer!(|| "prepare pairing inputs");

    let scalar_size = E::ScalarField::MODULUS_BIT_SIZE as usize;
    let window_size = FixedBase::get_mul_window_size(num_var);

    let h_table =
        FixedBase::get_window_table(scalar_size, window_size, verifier_param.h.into_group());
    let h_mul: Vec<E::G2> = FixedBase::msm(scalar_size, window_size, &h_table, point);

    let ignored = verifier_param.num_vars - num_var;
    let h_vec: Vec<_> = (0..num_var)
        .map(|i| -h_mul[i] + verifier_param.h_mask[ignored + i])
        .collect();
    let h_vec: Vec<E::G2Affine> = E::G2::normalize_batch(&h_vec);
    end_timer!(prepare_inputs_timer);

    let pairing_product_timer = start_timer!(|| "pairing product");

    let mut ps: Vec<E::G1Prepared> = proof
        .proofs
        .iter()
        .map(|&x| E::G1Prepared::from(x))
        .collect();

    #[cfg(feature = "parallel")]
    let mut hs: Vec<E::G2Prepared> = h_vec
        .into_par_iter()
        .take(num_var)
        .map(E::G2Prepared::from)
        .collect();

    #[cfg(not(feature = "parallel"))]
    let mut hs: Vec<E::G2Prepared> = h_vec
        .into_iter()
        .take(num_var)
        .map(E::G2Prepared::from)
        .collect();

    ps.push(E::G1Prepared::from(
        (verifier_param.g.mul(*value) - commitment.0.into_group()).into_affine(),
    ));
    hs.push(E::G2Prepared::from(verifier_param.h));

    let res = E::multi_pairing(ps, hs) == ark_ec::pairing::PairingOutput(E::TargetField::one());

    end_timer!(pairing_product_timer);
    end_timer!(verify_timer);
    Ok(res)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_std::{test_rng, UniformRand};

    type E = Bls12_381;
    type Fr = <E as Pairing>::ScalarField;

    #[test]
    fn test_demkzg_commit_open_verify() {
        let mut rng = test_rng();
        let num_vars = 4;

        let srs = DeMkzg::<E>::gen_srs_for_testing(&mut rng, num_vars).unwrap();
        let (pk, vk) = DeMkzg::<E>::trim(&srs, num_vars).unwrap();

        let evals: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, evals,
        ));

        let commitment = DeMkzg::<E>::commit(&pk, &poly).unwrap();
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let (proof, eval) = DeMkzg::<E>::open(&pk, &poly, &point).unwrap();
        let result = DeMkzg::<E>::verify(&vk, &commitment, &point, &eval, &proof).unwrap();
        assert!(result, "Verification should pass");
    }

    #[test]
    fn test_demkzg_wrong_eval_fails() {
        let mut rng = test_rng();
        let num_vars = 3;

        let srs = DeMkzg::<E>::gen_srs_for_testing(&mut rng, num_vars).unwrap();
        let (pk, vk) = DeMkzg::<E>::trim(&srs, num_vars).unwrap();

        let evals: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, evals,
        ));

        let commitment = DeMkzg::<E>::commit(&pk, &poly).unwrap();
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let (proof, _eval) = DeMkzg::<E>::open(&pk, &poly, &point).unwrap();

        let wrong_eval = Fr::rand(&mut rng);
        let result = DeMkzg::<E>::verify(&vk, &commitment, &point, &wrong_eval, &proof).unwrap();
        assert!(!result, "Verification should fail with wrong evaluation");
    }

    #[test]
    fn test_demkzg_verify_internal() {
        let mut rng = test_rng();
        let num_vars = 5;

        let srs = DeMkzg::<E>::gen_srs_for_testing(&mut rng, num_vars).unwrap();
        let (pk, vk) = DeMkzg::<E>::trim(&srs, num_vars).unwrap();

        let evals: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, evals,
        ));

        let commitment = DeMkzg::<E>::commit(&pk, &poly).unwrap();
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let (proof, eval) = DeMkzg::<E>::open(&pk, &poly, &point).unwrap();

        // Test via verify_internal directly
        assert!(verify_internal::<E>(&vk, &commitment, &point, &eval, &proof).unwrap());
    }
}
