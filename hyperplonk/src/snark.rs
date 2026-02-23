// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use crate::{
    errors::HyperPlonkErrors,
    mock::MockCircuit,
    structs::{HyperPlonkIndex, HyperPlonkProof, HyperPlonkProvingKey, HyperPlonkVerifyingKey},
    utils::{build_f, prover_sanity_check, PcsAccumulator},
    HyperPlonkSNARK,
};

use arithmetic::{build_eq_x_r_vec, VirtualPolynomial};
use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;
use ark_std::{time::Instant, Zero};
use rayon::iter::IntoParallelRefIterator;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use std::{sync::Arc, time::Duration};
use subroutines::{
    pcs::prelude::{Commitment, PolynomialCommitmentScheme},
    poly_iop::{
        prelude::{PermutationCheck, ZeroCheck},
        PolyIOP,
    },
    BatchProof, IOPProof,
};
use transcript::IOPTranscript;

impl<E, PCS> HyperPlonkSNARK<E, PCS> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    // Ideally we want to access polynomial as PCS::Polynomial, instead of instantiating it here.
    // But since PCS::Polynomial can be both univariate or multivariate in our implementation
    // we cannot bound PCS::Polynomial with a property trait bound.
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
        BatchProof = BatchProof<E, PCS>,
    >,
{
    type Index = HyperPlonkIndex<E::ScalarField>;
    type ProvingKey = HyperPlonkProvingKey<E, PCS>;
    type VerifyingKey = HyperPlonkVerifyingKey<E, PCS>;
    type Proof = HyperPlonkProof<E, Self, PCS>;

    fn preprocess(
        index: &Self::Index,
        pcs_srs: &PCS::SRS,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey, Duration), HyperPlonkErrors> {
        let start = Instant::now();
        let num_vars = index.num_variables();
        let supported_ml_degree = num_vars;

        // extract PCS prover and verifier keys from SRS
        let (pcs_prover_param, pcs_verifier_param) =
            PCS::trim(pcs_srs, None, Some(supported_ml_degree))?;

        // build permutation oracles
        let mut permutation_oracles = vec![];
        let mut perm_comms = vec![];
        let chunk_size = 1 << num_vars;
        for i in 0..index.num_witness_columns() {
            let perm_oracle = Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                num_vars,
                &index.permutation[i * chunk_size..(i + 1) * chunk_size],
            ));
            let perm_comm = PCS::commit(&pcs_prover_param, &perm_oracle)?;
            permutation_oracles.push(perm_oracle);
            perm_comms.push(perm_comm);
        }

        // build selector oracles and commit to it
        let selector_oracles: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> = index
            .selectors
            .iter()
            .map(|s| Arc::new(DenseMultilinearExtension::from(s)))
            .collect();

        let selector_commitments = selector_oracles
            .par_iter()
            .map(|poly| PCS::commit(&pcs_prover_param, poly))
            .collect::<Result<Vec<_>, _>>()?;
        let duration_sel = start.elapsed();
        Ok((
            Self::ProvingKey {
                params: index.params.clone(),
                permutation_oracles,
                selector_oracles,
                selector_commitments: selector_commitments.clone(),
                permutation_commitments: perm_comms.clone(),
                pcs_param: pcs_prover_param,
            },
            Self::VerifyingKey {
                params: index.params.clone(),
                pcs_param: pcs_verifier_param,
                selector_commitments,
                perm_commitments: perm_comms,
            },
            duration_sel,
        ))
    }

    fn mul_prove(
        pk: &Self::ProvingKey,
        circuits: Vec<MockCircuit<E::ScalarField>>,
    ) -> Result<
        (
            Vec<VirtualPolynomial<E::ScalarField>>,
            Vec<VirtualPolynomial<E::ScalarField>>,
            Vec<Vec<PCS::Commitment>>,
            Vec<Vec<PCS::Commitment>>,
            Duration,
        ),
        HyperPlonkErrors,
    > {
        let mul_prove = Instant::now();
        let mut transcript = IOPTranscript::<E::ScalarField>::new(b"mul_prove");
        let mut f_hats = Vec::new();
        let mut perm_f_hats = Vec::new();
        let mut f_commitments = Vec::new();
        let mut perm_f_commitments = Vec::new();
        let mut duration_wit = Duration::from_secs(0);
        let mut duration_f_hat = Duration::from_secs(0);
        let mut duration_perm_hat = Duration::from_secs(0);
        for circuit in circuits.iter() {
            let pub_input = &circuit.public_inputs;
            let witness = circuit.witnesses.clone();
            prover_sanity_check(&pk.params, &pub_input, &witness)?;
            let _num_vars = pk.params.num_variables();

            let witness_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> = witness
                .iter()
                .map(|w| Arc::new(DenseMultilinearExtension::from(w)))
                .collect();
            let start = Instant::now();
            let witness_commits = witness_polys
                .par_iter()
                .map(|x| PCS::commit(&pk.pcs_param, x).unwrap())
                .collect::<Vec<_>>();
            duration_wit += start.elapsed();
            for w_com in witness_commits.iter() {
                transcript.append_serializable_element(b"w", w_com)?;
            }

            let fx = build_f(
                &pk.params.gate_func,
                pk.params.num_variables(),
                &pk.selector_oracles,
                &witness_polys,
            )?;
            let f_hat = <Self as ZeroCheck<E::ScalarField>>::mul_prove(&fx, &mut transcript)?;
            let f_hat_start = Instant::now();
            let f_hat_comms: Vec<PCS::Commitment> = f_hat
                .flattened_ml_extensions
                .iter()
                .map(|mle| PCS::commit(&pk.pcs_param, mle))
                .collect::<Result<Vec<_>, _>>()?;
            for comm in f_hat_comms.iter() {
                transcript.append_serializable_element(b"f_hat", comm)?;
            }
            f_hats.push(f_hat);
            f_commitments.push(f_hat_comms);
            duration_f_hat += f_hat_start.elapsed();

            let (perm_check_proof, prod_poly, frac_poly, perm_f_hat) =
                <Self as PermutationCheck<E, PCS>>::prove(
                    &pk.pcs_param,
                    &witness_polys,
                    &witness_polys,
                    &pk.permutation_oracles,
                    &mut transcript,
                )
                .map_err(|e| HyperPlonkErrors::from(e))?;

            let prod_comm = PCS::commit(&pk.pcs_param, &prod_poly)?;
            let frac_comm = PCS::commit(&pk.pcs_param, &frac_poly)?;
            transcript.append_serializable_element(b"prod_poly", &prod_comm)?;
            transcript.append_serializable_element(b"frac_poly", &frac_comm)?;

            let perm_com_start = Instant::now();
            let perm_f_hat_comms: Vec<PCS::Commitment> = perm_f_hat
                .flattened_ml_extensions
                .iter()
                .map(|mle| PCS::commit(&pk.pcs_param, mle))
                .collect::<Result<Vec<_>, _>>()?;
            for comm in perm_f_hat_comms.iter() {
                transcript.append_serializable_element(b"perm_f_hat", comm)?;
            }
            perm_f_hats.push(perm_f_hat);
            perm_f_commitments.push(perm_f_hat_comms);
            duration_perm_hat += perm_com_start.elapsed();
        }
        let mul_prove_duration =
            mul_prove.elapsed() - duration_f_hat - duration_perm_hat - duration_wit;
        println!(
            "----------------- mul_prove Duration ----------------------{:?}",
            mul_prove_duration
        );
        Ok((
            f_hats,
            perm_f_hats,
            f_commitments,
            perm_f_commitments,
            duration_wit,
        ))
    }

    fn prove(
        f_hats: Vec<VirtualPolynomial<<E as Pairing>::ScalarField>>,
        perm_f_hats: Vec<VirtualPolynomial<<E as Pairing>::ScalarField>>,
        f_commitments: Vec<Vec<PCS::Commitment>>,
        perm_f_commitments: Vec<Vec<PCS::Commitment>>,
        f_q_proof: &IOPProof<E::ScalarField>,
        perm_q_proof: &IOPProof<E::ScalarField>,
        pk: &Self::ProvingKey,
        transcript: &mut self::IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Vec<Vec<<E as Pairing>::ScalarField>>,
            Vec<Vec<<E as Pairing>::ScalarField>>,
            BatchProof<E, PCS>,
        ),
        HyperPlonkErrors,
    > {
        // let start = start_timer!(||"prove");
        let f_num_vars = f_hats[0].aux_info.num_variables;
        let f_t = f_hats[0].flattened_ml_extensions.len();
        let f_m = f_hats.len();

        let perm_num_vars = perm_f_hats[0].aux_info.num_variables;
        let perm_t = perm_f_hats[0].flattened_ml_extensions.len();
        let perm_m = perm_f_hats.len();

        let f_rb = f_q_proof.point.clone();
        let perm_rb = perm_q_proof.point.clone();

        let f_eval_point = [
            f_rb.clone(),
            vec![E::ScalarField::zero(); f_num_vars - f_rb.len()],
        ]
        .concat();
        let perm_eval_point = [
            perm_rb.clone(),
            vec![E::ScalarField::zero(); perm_num_vars - perm_rb.len()],
        ]
        .concat();

        let mut pcs_acc = PcsAccumulator::<E, PCS>::new(f_num_vars);

        for (poly, comms) in f_hats.iter().zip(f_commitments.iter()) {
            for (mle, comm) in poly.flattened_ml_extensions.iter().zip(comms.iter()) {
                pcs_acc.insert_poly_and_points(mle, comm, &f_eval_point);
            }
        }

        for (poly, comms) in perm_f_hats.iter().zip(perm_f_commitments.iter()) {
            for (mle, comm) in poly.flattened_ml_extensions.iter().zip(comms.iter()) {
                pcs_acc.insert_poly_and_points(mle, comm, &perm_eval_point);
            }
        }

        let pcs_param = &pk.pcs_param;
        let start_prove = Instant::now();
        let batch_opening_proof = pcs_acc.multi_open(pcs_param, transcript)?;
        println!(
            "--------------Prove Duration----------------{:?}",
            start_prove.elapsed()
        );
        // println!("proof{:?}",batch_opening_proof);
        let evaluations: Vec<E::ScalarField> = batch_opening_proof
            .f_i_eval_at_point_i
            .iter()
            .cloned()
            .collect();

        let f_eq_rb_vec = build_eq_x_r_vec(&f_rb)?;
        let mut f_folded_evals = vec![vec![E::ScalarField::zero(); f_t]; f_m];

        // 计算 f_hats 的折叠评估值
        let mut eval_idx = 0;
        for i in 0..f_m {
            for j in 0..f_t {
                f_folded_evals[i][j] = f_eq_rb_vec[i] * evaluations[eval_idx];
                eval_idx += 1;
            }
        }

        let perm_eq_rb_vec = build_eq_x_r_vec(&perm_rb)?;
        let mut perm_folded_evals = vec![vec![E::ScalarField::zero(); perm_t]; perm_m];

        // 计算 perm_f_hats 的折叠评估值
        for i in 0..perm_m {
            for j in 0..perm_t {
                perm_folded_evals[i][j] = perm_eq_rb_vec[i] * evaluations[eval_idx];
                eval_idx += 1;
            }
        }
        // end_timer!(start);
        Ok((f_folded_evals, perm_folded_evals, batch_opening_proof))
    }

    fn verify(
        polys: Vec<(
            Vec<VirtualPolynomial<E::ScalarField>>,
            Vec<Vec<E::ScalarField>>,
        )>, // (polys, folded_evals) pairs
        commitments: Vec<Vec<PCS::Commitment>>,
        q_proofs: Vec<IOPProof<E::ScalarField>>,
        batch_opening_proof: PCS::BatchProof,
        vk: &Self::VerifyingKey,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, HyperPlonkErrors> {
        let start_verify = Instant::now();
        let (f_hats, f_folded_evals) = &polys[0];
        let (perm_f_hats, perm_folded_evals) = &polys[1];

        let f_num_vars = f_hats[0].aux_info.num_variables;
        let f_t = f_hats[0].flattened_ml_extensions.len();
        let f_m = f_hats.len();

        let perm_num_vars = perm_f_hats[0].aux_info.num_variables;
        let perm_t = perm_f_hats[0].flattened_ml_extensions.len();
        let perm_m = perm_f_hats.len();

        let f_rb = q_proofs[0].point.clone();
        let perm_rb = q_proofs[1].point.clone();

        let f_eval_point = [
            f_rb.clone(),
            vec![E::ScalarField::zero(); f_num_vars - f_rb.len()],
        ]
        .concat();
        let perm_eval_point = [
            perm_rb.clone(),
            vec![E::ScalarField::zero(); perm_num_vars - perm_rb.len()],
        ]
        .concat();

        // 构造 PcsAccumulator 并验证批量打开证明
        // let mut pcs_acc = PcsAccumulator::<E, PCS>::new(f_num_vars);
        let mut all_commitments = Vec::new();
        let mut all_points = Vec::new();

        let mut commitment_idx = 0;

        for (poly, comms) in f_hats.iter().zip(commitments.iter().skip(commitment_idx)) {
            for (mle, comm) in poly.flattened_ml_extensions.iter().zip(comms.iter()) {
                // pcs_acc.insert_poly_and_points(mle, comm, &f_eval_point);
                all_commitments.push(comm.clone());
                all_points.push(f_eval_point.clone());
            }
        }
        commitment_idx += f_hats.len();

        for (poly, comms) in perm_f_hats
            .iter()
            .zip(commitments.iter().skip(commitment_idx))
        {
            for (mle, comm) in poly.flattened_ml_extensions.iter().zip(comms.iter()) {
                // pcs_acc.insert_poly_and_points(mle, comm, &perm_eval_point);
                all_commitments.push(comm.clone());
                all_points.push(perm_eval_point.clone());
            }
        }

        let compute_fold_eval = Instant::now();

        let is_valid_opening = PCS::batch_verify(
            &vk.pcs_param,
            &all_commitments,
            &all_points,
            &batch_opening_proof,
            transcript,
        )?;
        if !is_valid_opening {
            return Ok(false);
        }

        let evaluations: Vec<E::ScalarField> = batch_opening_proof
            .f_i_eval_at_point_i
            .iter()
            .cloned()
            .collect();

        let f_eq_rb_vec = build_eq_x_r_vec(&f_rb)?;
        let mut computed_f_folded_evals = vec![vec![E::ScalarField::zero(); f_t]; f_m];
        let mut eval_idx = 0;
        for i in 0..f_m {
            for j in 0..f_t {
                computed_f_folded_evals[i][j] = f_eq_rb_vec[i] * evaluations[eval_idx];
                eval_idx += 1;
            }
        }

        let perm_eq_rb_vec = build_eq_x_r_vec(&perm_rb)?;
        let mut computed_perm_folded_evals = vec![vec![E::ScalarField::zero(); perm_t]; perm_m];
        for i in 0..perm_m {
            for j in 0..perm_t {
                computed_perm_folded_evals[i][j] = perm_eq_rb_vec[i] * evaluations[eval_idx];
                eval_idx += 1;
            }
        }

        let f_evals_match = computed_f_folded_evals
            .iter()
            .zip(f_folded_evals.iter())
            .all(|(computed, provided)| {
                computed.iter().zip(provided.iter()).all(|(c, p)| *c == *p)
            });
        let perm_evals_match = computed_perm_folded_evals
            .iter()
            .zip(perm_folded_evals.iter())
            .all(|(computed, provided)| {
                computed.iter().zip(provided.iter()).all(|(c, p)| *c == *p)
            });

        println!("compute_fold_eval time:{:?}", compute_fold_eval.elapsed());

        if !f_evals_match || !perm_evals_match {
            return Ok(false);
        }
        println!(
            "--------------Verify Duration----------------{:?}",
            start_verify.elapsed()
        );
        Ok(true)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::custom_gate::CustomizedGates;
    use ark_bls12_381::Bls12_381;
    use ark_std::{rand::rngs::StdRng, test_rng, time::Instant};
    use subroutines::{
        pcs::prelude::{MultilinearKzgPCS, PolynomialCommitmentScheme},
        poly_iop::PolyIOP,
        SumCheck,
    };

    #[test]
    fn test_hyperplonk_e2e() -> Result<(), HyperPlonkErrors> {
        let mock_gate = CustomizedGates::vanilla_plonk_gate();
        let nv = 16;
        let log_partition = 3;
        let num_constraints = 1 << nv;
        let num_partition = 1 << log_partition;
        test_hyperplonk_helper::<Bls12_381>(
            mock_gate,
            num_constraints,
            num_partition,
            nv - log_partition,
        )
    }

    fn test_hyperplonk_helper<E: Pairing>(
        mock_gate: CustomizedGates,
        num_constraints: usize,
        num_partitions: usize,
        support_size: usize,
    ) -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, support_size)?;

        let partition_circuits = MockCircuit::<E::ScalarField>::partition_circuit::<StdRng>(
            num_constraints,
            &mock_gate,
            num_partitions,
        );

        let mut transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();

        let (pk, vk, duration_sel) = <PolyIOP<E::ScalarField> as HyperPlonkSNARK<
            E,
            MultilinearKzgPCS<E>,
        >>::preprocess(&partition_circuits[0].index, &pcs_srs)?;

        let prove = Instant::now();
        let (f_hats, perm_f_hats, f_hat_commitments, perm_f_commitments, duration_wit) =
            <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::mul_prove(
                &pk,
                partition_circuits,
            )?;
        let duration_com = duration_sel + duration_wit;
        println!("Commit duration: {:?}", duration_com);

        let sums = vec![E::ScalarField::zero(); f_hats.len()];

        let start = Instant::now();
        let (q_proof, q_sum, q_aux_info, fold_poly, fold_sum) =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::sum_fold(
                f_hats.clone(),
                sums,
                &mut transcript,
            )?;
        let duration_fold1 = start.elapsed();

        let start = Instant::now();
        let mut transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();
        let _subclaim = <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
            q_sum,
            &q_proof,
            &q_aux_info,
            &mut transcript,
        )?;
        let duration_verify1 = start.elapsed();

        let start = Instant::now();
        let mut transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();
        let proof = <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
            &fold_poly.deep_copy(),
            &mut transcript,
        )?;
        let duration_check1 = start.elapsed();

        let start = Instant::now();
        let mut transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();
        let _subclaim = <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
            fold_sum,
            &proof,
            &fold_poly.aux_info,
            &mut transcript,
        )?;
        let duration_verify2 = start.elapsed();

        let mut transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();
        let sums = vec![E::ScalarField::zero(); perm_f_hats.len()];

        let start = Instant::now();
        let (perm_q_proof, perm_q_sum, perm_q_aux_info, perm_fold_poly, perm_fold_sum) =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::sum_fold(
                perm_f_hats.clone(),
                sums,
                &mut transcript,
            )?;
        let duration_fold2 = start.elapsed();

        let start = Instant::now();
        let mut transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();
        let _perm_subclaim = <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
            perm_q_sum,
            &perm_q_proof,
            &perm_q_aux_info,
            &mut transcript,
        )?;
        let duration_verify3 = start.elapsed();

        let mut transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();

        let start = Instant::now();
        let perm_proof = <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
            &perm_fold_poly.deep_copy(),
            &mut transcript,
        )?;
        let duration_check2 = start.elapsed();

        let mut transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();

        let start = Instant::now();
        let _perm_subclaim = <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
            fold_sum,
            &perm_proof,
            &perm_fold_poly.aux_info,
            &mut transcript,
        )?;
        let duration_verify4 = start.elapsed();

        let sumcheck_fold_duration = duration_fold1 + duration_fold2;
        let sumcheck_prove_duration = duration_check1 + duration_check2;
        let sumcheck_verify_duration =
            duration_verify1 + duration_verify2 + duration_verify3 + duration_verify4;
        println!("SumFold duration: {:?}", sumcheck_fold_duration);
        println!("SumCheck prove duration: {:?}", sumcheck_prove_duration);
        println!("SumCheck verify duration: {:?}", sumcheck_verify_duration);

        let mut transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();

        let (f_folded_evals, perm_folded_evals, batch_opening_proof) =
            <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::prove(
                f_hats.clone(),
                perm_f_hats.clone(),
                f_hat_commitments.clone(),
                perm_f_commitments.clone(),
                &q_proof,
                &perm_q_proof,
                &pk,
                &mut transcript,
            )?;

        let polys = vec![(f_hats, f_folded_evals), (perm_f_hats, perm_folded_evals)];
        let commitments = [f_hat_commitments, perm_f_commitments].concat();
        let q_proofs = vec![q_proof, perm_q_proof];
        let _prove_duration = prove.elapsed();

        let mut transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();

        let is_valid =
            <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::verify(
                polys,
                commitments,
                q_proofs,
                batch_opening_proof,
                &vk,
                &mut transcript,
            )?;
        assert!(is_valid, "HyperPlonk verification failed");
        println!("Total prove duration: {:?}", _prove_duration);
        Ok(())
    }
}
