//! Sumcheck-based distributed batch opening and verification for DeMkzg.
//!
//! Ported from HyperPianist's deMultilinear_kzg/batching.rs.
//! Adapted to work with sumfold_deSNARK's type system.

use crate::{
    pcs::{
        multilinear_kzg::util::eq_eval,
        prelude::{Commitment, PCSError},
        PolynomialCommitmentScheme,
    },
    poly_iop::{prelude::SumCheck, PolyIOP},
    IOPProof,
};
use arithmetic::{
    bit_decompose, build_eq_x_r_vec, build_eq_x_r_with_coeff, math::Math,
    DenseMultilinearExtension, VPAuxInfo, VirtualPolynomial,
};
use ark_ec::{pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, CurveGroup};
use ark_std::{end_timer, log2, start_timer, One, Zero};
use std::{collections::BTreeMap, iter, marker::PhantomData, ops::Deref, sync::Arc};
use transcript::IOPTranscript;

#[cfg(feature = "distributed")]
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use crate::pcs::multilinear_kzg::batching::BatchProof;

/// Distributed multi-open using SumCheck-based batching.
///
/// Steps:
/// 1. Master gets challenge point t from transcript, broadcasts to workers
/// 2. Build eq(t,i) for i in [0..k]
/// 3. Build tilde_g_i(b) = eq(t, i) * f_i(b)
/// 4. Compute tilde_eq_i(b) = eq(b, point_i) (adjusted for party index)
/// 5. Run distributed sumcheck on sum_i tilde_eq_i * tilde_g_i
/// 6. Build g'(X) = sum_i tilde_eq_i(a2) * tilde_g_i(X)
/// 7. Distributed open g'(X) at point a2
#[cfg(feature = "distributed")]
pub(crate) fn d_multi_open_internal<E, PCS>(
    prover_param: &PCS::ProverParam,
    polynomials: Vec<PCS::Polynomial>,
    points: &[PCS::Point],
    evals: &[PCS::Evaluation],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<Option<BatchProof<E, PCS>>, PCSError>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        ProverParam = super::super::multilinear_kzg::srs::MultilinearProverParam<E>,
        Proof = super::super::multilinear_kzg::MultilinearKzgProof<E>,
    >,
{
    let open_timer = start_timer!(|| format!("d_multi_open {} polynomials", polynomials.len()));

    let num_var = polynomials[0].num_vars;
    let k = polynomials.len();
    let ell = log2(k) as usize;

    // Append eval points and evaluations to transcript (master only).
    // This matches the non-distributed batch_verify_internal which also
    // appends them before generating challenge t, ensuring Fiat-Shamir
    // transcript consistency between prover and verifier.
    if Net::am_master() {
        for eval_point in points.iter() {
            transcript.append_serializable_element(b"eval_point", eval_point)?;
        }
        for eval in evals.iter() {
            transcript.append_field_element(b"eval", eval)?;
        }
    }

    // Step 1: Master generates challenge point t, broadcasts to all
    let t = if Net::am_master() {
        let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;
        Net::recv_from_master_uniform(Some(t))
    } else {
        Net::recv_from_master_uniform(None)
    };

    // Step 2: eq(t, i) for i in [0..k]
    let eq_t_i_list = build_eq_x_r_vec(t.as_ref())?;

    // Step 3: Build tilde_g_i = eq(t,i) * f_i, then merge by shared points
    let timer = start_timer!(|| format!("compute tilde g for {} polynomials", k));
    let point_indices = points
        .iter()
        .fold(BTreeMap::<_, _>::new(), |mut indices, point| {
            let idx = indices.len();
            indices.entry(point).or_insert(idx);
            indices
        });
    let deduped_points =
        BTreeMap::from_iter(point_indices.iter().map(|(point, idx)| (*idx, *point)))
            .into_values()
            .collect::<Vec<_>>();

    let mut point_ids = vec![vec![]; point_indices.len()];
    for (i, point) in points.iter().enumerate() {
        point_ids[point_indices[point]].push(i);
    }

    let mut tilde_gs = polynomials;
    // Scale each polynomial by eq(t, i)
    #[cfg(feature = "parallel")]
    {
        tilde_gs
            .par_iter_mut()
            .zip(&eq_t_i_list)
            .for_each(|(poly, coeff)| {
                Arc::make_mut(poly)
                    .evaluations
                    .iter_mut()
                    .for_each(|eval| *eval *= coeff);
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (poly, coeff) in tilde_gs.iter_mut().zip(&eq_t_i_list) {
            Arc::make_mut(poly)
                .evaluations
                .iter_mut()
                .for_each(|eval| *eval *= coeff);
        }
    }

    // Merge polynomials that share the same opening point
    let (mut merged_tilde_gs, merged_tilde_gs_copy): (Vec<_>, Vec<_>) = point_ids
        .iter()
        .map(|ids| {
            let mut merged = Arc::new(DenseMultilinearExtension::zero());
            for &idx in ids {
                *Arc::get_mut(&mut merged).unwrap() += tilde_gs[idx].deref();
            }
            (Arc::new(DenseMultilinearExtension::clone(&merged)), merged)
        })
        .unzip();
    end_timer!(timer);

    // Step 4: Build tilde_eq_i adjusted for party index
    let timer = start_timer!(|| "compute tilde eq");
    let num_party_vars = Net::n_parties().log_2();
    let index_vec: Vec<E::ScalarField> = bit_decompose(Net::party_id() as u64, num_party_vars)
        .into_iter()
        .map(|x| E::ScalarField::from(x))
        .collect();

    let tilde_eqs: Vec<_> = deduped_points
        .iter()
        .map(|point| {
            let coeff = eq_eval(&point[num_var..], &index_vec).unwrap();
            build_eq_x_r_with_coeff(&point[..num_var], &coeff).unwrap()
        })
        .collect();
    end_timer!(timer);

    // Step 5: Build virtual polynomial and run distributed SumCheck
    let timer = start_timer!(|| format!("distributed sumcheck of {} variables", num_var));

    let proof = {
        let step = start_timer!(|| "add mle");
        let mut sum_check_vp = VirtualPolynomial::new(num_var);
        for (merged_tilde_g, tilde_eq) in
            merged_tilde_gs_copy.into_iter().zip(tilde_eqs.into_iter())
        {
            sum_check_vp.add_mle_list([merged_tilde_g, tilde_eq], E::ScalarField::one())?;
        }
        end_timer!(step);

        // Use distributed SumCheck prove
        let transcript_opt = if Net::am_master() {
            Some(transcript)
        } else {
            None
        };

        match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::d_prove::<Net>(
            &sum_check_vp,
            transcript_opt,
        ) {
            Ok(p) => p,
            Err(_e) => {
                return Err(PCSError::InvalidProver(
                    "Sumcheck in distributed batch proving failed".to_string(),
                ));
            },
        }
    };
    end_timer!(timer);

    // Step 6: Build g'(X) at sumcheck's point a2
    let a2 = if Net::am_master() {
        let a2 = proof.as_ref().unwrap().point.clone();
        Net::recv_from_master_uniform(Some(a2))
    } else {
        Net::recv_from_master_uniform(None)
    };

    let step = start_timer!(|| "evaluate at a2");
    for (merged_tilde_g, point) in merged_tilde_gs.iter_mut().zip(deduped_points.iter()) {
        let eq_i_a2 = eq_eval(&a2, point)?;
        Arc::get_mut(merged_tilde_g)
            .unwrap()
            .evaluations
            .iter_mut()
            .for_each(|x| *x *= eq_i_a2);
    }

    let g_prime = merged_tilde_gs.into_iter().fold(
        Arc::new(DenseMultilinearExtension::zero()),
        |mut acc, b| {
            if acc.is_zero() {
                return b;
            }
            if b.is_zero() {
                return acc;
            }
            *Arc::get_mut(&mut acc).unwrap() += &*b;
            acc
        },
    );
    end_timer!(step);

    // Step 7: Open g' at a2 using distributed open.
    // g' has num_var local variables, but a2 has num_var + num_party_vars
    // dimensions (from the distributed sumcheck). DeMkzg::d_open handles
    // this by splitting the open into local rounds + master-aggregated
    // party-dimension rounds.
    let step = start_timer!(|| "pcs open g'");
    let g_prime_proof_opt = super::deMkzg::DeMkzg::<E>::d_open(prover_param, &g_prime, &a2)?;
    end_timer!(step);

    end_timer!(open_timer);

    if Net::am_master() {
        Ok(Some(BatchProof {
            sum_check_proof: proof.unwrap(),
            f_i_eval_at_point_i: evals.to_vec(),
            g_prime_proof: g_prime_proof_opt.unwrap(),
        }))
    } else {
        Ok(None)
    }
}

/// Batch verify: verifier-side (non-distributed, runs on verifier).
///
/// Steps:
/// 1. Get challenge point t from transcript
/// 2. Build g' commitment homomorphically
/// 3. Verify sum via SumCheck
/// 4. Verify PCS opening
pub(crate) fn batch_verify_internal<E, PCS>(
    verifier_param: &PCS::VerifierParam,
    f_i_commitments: &[Commitment<E>],
    points: &[PCS::Point],
    proof: &BatchProof<E, PCS>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
    >,
{
    let open_timer = start_timer!(|| "batch verification");

    // Append eval points and evaluations (matches d_multi_open_internal
    // and the non-distributed batch_verify_internal).
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in proof.f_i_eval_at_point_i.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }

    let k = f_i_commitments.len();
    let ell = log2(k) as usize;
    let num_var = proof.sum_check_proof.point.len();

    // Step 1: Challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    // Sumcheck point a2
    let a2 = &proof.sum_check_proof.point[..num_var];

    // Step 2: Build g' commitment homomorphically
    let step = start_timer!(|| "build homomorphic commitment");
    let eq_t_list = build_eq_x_r_vec(t.as_ref())?;

    let mut scalars = vec![];
    let mut bases = vec![];
    for (i, point) in points.iter().enumerate() {
        let eq_i_a2 = eq_eval(a2, point)?;
        scalars.push(eq_i_a2 * eq_t_list[i]);
        bases.push(f_i_commitments[i].0);
    }
    let g_prime_commit = E::G1MSM::msm_unchecked(&bases, &scalars);
    end_timer!(step);

    // Step 3: Verify sum via SumCheck
    let mut sum = E::ScalarField::zero();
    for (i, &e) in eq_t_list.iter().enumerate().take(k) {
        sum += e * proof.f_i_eval_at_point_i[i];
    }
    let aux_info = VPAuxInfo {
        max_degree: 2,
        num_variables: num_var,
        phantom: PhantomData,
    };
    let subclaim = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
        sum,
        &proof.sum_check_proof,
        &aux_info,
        transcript,
    ) {
        Ok(p) => p,
        Err(_e) => {
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch verification failed".to_string(),
            ));
        },
    };
    let tilde_g_eval = subclaim.expected_evaluation;

    // Step 4: Verify PCS opening
    let res = PCS::verify(
        verifier_param,
        &Commitment(g_prime_commit.into()),
        a2.to_vec().as_ref(),
        &tilde_g_eval,
        &proof.g_prime_proof,
    )?;

    end_timer!(open_timer);
    Ok(res)
}
