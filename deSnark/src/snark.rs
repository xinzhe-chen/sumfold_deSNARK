//! deSnark protocol functions.

use crate::{
    errors::DeSnarkError,
    structs::{
        BenchmarkTimings, Config, HyperPlonkProvingKey, HyperPlonkVerifyingKey, MockCircuit, Proof,
        SumCheckInstance, SumFoldProof,
    },
};
use arithmetic::{build_eq_x_r_vec, eq_poly::EqPolynomial, VPAuxInfo, VirtualPolynomial};
use ark_ec::{pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_into_iter, rand::Rng, test_rng, time::Instant};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use hyperplonk::{
    prelude::{build_f, eval_f},
    HyperPlonkSNARK,
};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
use subroutines::{
    pcs::PolynomialCommitmentScheme,
    poly_iop::{
        prelude::{PolyIOP, SumCheck},
        sum_check::verify_sum_fold,
    },
    BatchProof, Commitment, DeMkzg, IOPProof, MultilinearKzgPCS,
};
use tracing::{debug, info, instrument, warn};
use transcript::IOPTranscript;

/// Result type for deSnark operations.
pub type Result<T> = std::result::Result<T, DeSnarkError>;

/// Type aliases for clarity
pub type ProvingKey<E, PCS> = HyperPlonkProvingKey<E, PCS>;
pub type VerifyingKey<E, PCS> = HyperPlonkVerifyingKey<E, PCS>;

/// PCS trait bounds required for HyperPlonk compatibility.
pub trait HyperPlonkPCS<E: Pairing>:
    PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
        BatchProof = BatchProof<E, Self>,
        SRS: CanonicalSerialize + CanonicalDeserialize,
    > + Sized
{
}

impl<E, PCS> HyperPlonkPCS<E> for PCS
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
        BatchProof = BatchProof<E, PCS>,
        SRS: CanonicalSerialize + CanonicalDeserialize,
    >,
{
}

/// Phase 0: Generate or load SRS from config.
///
/// If `config.srs_path` is set:
/// - Tries to load SRS from file, validates it is large enough via `PCS::trim`
/// - If file missing or SRS too small, generates a new SRS and saves it
///
/// If `config.srs_path` is `None`, generates SRS without caching.
///
/// The SRS log-size equals `log_num_constraints` (the full nv), because
/// `d_commit` internally adds `log(K)` party variables to each local MLE,
/// so the SRS must support `(N/K) * K = N` evaluations.
///
/// WARNING: Uses `test_rng()` — for testing only, not production.
#[instrument(level = "debug", skip_all, name = "setup")]
pub fn setup<E: Pairing, PCS: HyperPlonkPCS<E>>(config: &Config) -> Result<PCS::SRS> {
    // d_commit extends local MLEs (num_vars = log(N/K)) with log(K) party dims,
    // so the SRS must support log(N/K) + log(K) = log(N) = log_num_constraints.
    let supported_log_size = config.log_num_constraints;
    info!(
        "SRS setup: log_size = {} (log_constraints = {}, log_parties = {})",
        supported_log_size, config.log_num_constraints, config.log_num_parties
    );

    // Try loading from cache file
    if let Some(ref path) = config.srs_path {
        if let Some(srs) = try_load_srs::<E, PCS>(path, supported_log_size) {
            return Ok(srs);
        }
    }

    // Generate fresh SRS
    let mut rng = test_rng();
    let srs = PCS::gen_srs_for_testing(&mut rng, supported_log_size)
        .map_err(|e| DeSnarkError::InvalidParameters(format!("SRS generation failed: {e}")))?;
    info!("✅ SRS generated successfully");

    // Save to cache file
    if let Some(ref path) = config.srs_path {
        save_srs::<E, PCS>(&srs, path);
    }

    Ok(srs)
}

/// Try to load SRS from file and validate it is large enough.
/// Returns `None` if file doesn't exist, deserialization fails, or SRS is too
/// small.
fn try_load_srs<E: Pairing, PCS: HyperPlonkPCS<E>>(
    path: &str,
    supported_log_size: usize,
) -> Option<PCS::SRS> {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            info!("SRS cache miss ({}): {}", path, e);
            return None;
        },
    };
    let mut reader = std::io::BufReader::new(file);
    let srs: PCS::SRS = match CanonicalDeserialize::deserialize_uncompressed_unchecked(&mut reader)
    {
        Ok(s) => s,
        Err(e) => {
            warn!("SRS cache corrupted ({}): {}", path, e);
            return None;
        },
    };

    // Validate: try trimming to the required size
    match PCS::trim(&srs, None, Some(supported_log_size)) {
        Ok(_) => {
            info!("✅ SRS loaded from cache ({})", path);
            Some(srs)
        },
        Err(e) => {
            warn!(
                "SRS cache too small ({}, need log_size={}): {}",
                path, supported_log_size, e
            );
            None
        },
    }
}

/// Save SRS to file. Logs a warning on failure but does not propagate the
/// error.
fn save_srs<E: Pairing, PCS: HyperPlonkPCS<E>>(srs: &PCS::SRS, path: &str) {
    match std::fs::File::create(path) {
        Ok(file) => {
            let mut writer = std::io::BufWriter::new(file);
            if let Err(e) = srs.serialize_uncompressed(&mut writer) {
                warn!("Failed to write SRS cache ({}): {}", path, e);
            } else {
                info!("✅ SRS cached to {}", path);
            }
        },
        Err(e) => {
            warn!("Failed to create SRS cache file ({}): {}", path, e);
        },
    }
}

/// Phase 1: Generate circuit, keys, and mock circuits.
///
/// Internally calls HyperPlonk preprocess to generate proving and verifying
/// keys.
///
/// # Arguments
/// * `config` - Protocol configuration
/// * `srs` - Structured Reference String
///
/// # Returns
/// * `ProvingKey` - Key for proving
/// * `VerifyingKey` - Key for verification
/// * `Vec<MockCircuit>` - M circuits (each with index + public_inputs +
///   witnesses)
#[instrument(level = "debug", skip_all, name = "make_circuit")]
pub fn make_circuit<E: Pairing, PCS: HyperPlonkPCS<E>>(
    config: &Config,
    srs: &PCS::SRS,
) -> Result<(
    ProvingKey<E, PCS>,
    VerifyingKey<E, PCS>,
    Vec<MockCircuit<E::ScalarField>>,
)> {
    // 1. Generate M partitioned mock circuits
    let num_instances = config.num_instances();
    let constraints_per_party = config.num_constraints() / config.num_parties();
    info!(
        "Building {} mock circuits (constraints_per_party = {}, gate = {:?})",
        num_instances, constraints_per_party, config.gate_type
    );
    let circuits = config.build_partitioned_circuits::<E::ScalarField>();
    info!(
        "Circuits built: {} instances, {} witness columns, {} selector columns, {} public inputs each",
        circuits.len(),
        circuits[0].index.params.num_witness_columns(),
        circuits[0].index.params.num_selector_columns(),
        circuits[0].public_inputs.len()
    );

    // 2. Preprocess using the first circuit's index
    info!(
        "Preprocessing: num_variables = {}, num_constraints = {}",
        circuits[0].index.num_variables(),
        circuits[0].index.params.num_constraints
    );
    let (mut pk, mut vk, _duration) =
        <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, PCS>>::preprocess(&circuits[0].index, srs)
            .map_err(|e| DeSnarkError::HyperPlonkError(e.to_string()))?;

    // Override PCS params: preprocess trims to log(N/K), but d_commit needs
    // log(N) because it extends local MLEs with log(K) party variables.
    let d_commit_num_vars = config.log_num_constraints;
    let (pcs_prover_param, pcs_verifier_param) = PCS::trim(srs, None, Some(d_commit_num_vars))
        .map_err(|e| DeSnarkError::HyperPlonkError(format!("PCS trim for d_commit: {e}")))?;
    pk.pcs_param = pcs_prover_param;
    vk.pcs_param = pcs_verifier_param;
    info!(
        "PCS params overridden: d_commit supports {} vars (was {})",
        d_commit_num_vars,
        circuits[0].index.num_variables()
    );
    info!(
        "PK: {} selector commitments, {} permutation commitments, {} permutation oracles",
        pk.selector_commitments.len(),
        pk.permutation_commitments.len(),
        pk.permutation_oracles.len()
    );
    info!(
        "VK: {} selector commitments, {} permutation commitments",
        vk.selector_commitments.len(),
        vk.perm_commitments.len()
    );

    Ok((pk, vk, circuits))
}

/// Convert MockCircuits into SumCheck instances (one per circuit).
///
/// Each circuit's constraint polynomial `f(w_0(x), ..., w_d(x))` is built
/// from its gate function, selector oracles (from pk), and witness MLEs.
/// The claimed sum for each is 0 (a valid circuit satisfies all constraints
/// on the boolean hypercube).
///
/// This is the bridge from any circuit representation to the
/// circuit-agnostic `SumCheckInstance`.
///
/// # Arguments
/// * `pk` - Proving key (contains selector oracles and gate function)
/// * `circuits` - M mock circuits, each with witnesses and public inputs
///
/// # Returns
/// * `Vec<SumCheckInstance>` - One instance per circuit (VP + zero sum)
#[instrument(level = "debug", skip_all, name = "circuits_to_sumcheck")]
pub fn circuits_to_sumcheck<E: Pairing, PCS: HyperPlonkPCS<E>>(
    pk: &ProvingKey<E, PCS>,
    circuits: &[MockCircuit<E::ScalarField>],
) -> Result<Vec<SumCheckInstance<E::ScalarField>>> {
    let num_vars = pk.params.num_variables();
    let gate_func = &pk.params.gate_func;

    let instances: Vec<SumCheckInstance<E::ScalarField>> = circuits
        .iter()
        .map(|circuit| {
            let witness_mles: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> = circuit
                .witnesses
                .iter()
                .map(|w| Arc::new(DenseMultilinearExtension::from(w)))
                .collect();

            // Use each circuit's own selectors so each VP independently
            // satisfies its constraint (sum = 0). This is necessary because
            // each MockCircuit may have different selectors.
            let selector_mles: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> = circuit
                .index
                .selectors
                .iter()
                .map(|s| Arc::new(DenseMultilinearExtension::from(s)))
                .collect();

            let poly = build_f(gate_func, num_vars, &selector_mles, &witness_mles)
                .map_err(|e| DeSnarkError::HyperPlonkError(e.to_string()))?;

            Ok(SumCheckInstance::new(poly, E::ScalarField::from(0u64)))
        })
        .collect::<Result<Vec<_>>>()?;

    info!(
        "Converted {} circuits to SumCheck instances (num_vars = {}, max_degree = {})",
        instances.len(),
        instances[0].aux_info().num_variables,
        instances[0].aux_info().max_degree,
    );

    Ok(instances)
}

/// Phase 2: SumFold protocol - aggregate SumCheck instances.
///
/// Takes M SumCheck instances (each claiming sum = 0 over the boolean
/// hypercube) and folds them into a single instance via
/// the SumFold interactive argument.
///
/// Runs all three sum_fold versions (v1, v2, v3) and cross-validates
/// that they produce identical results.
///
/// # Arguments
/// * `instances` - M SumCheck instances to fold
/// * `transcript` - Fiat-Shamir transcript (threaded from caller)
///
/// # Returns
/// * `SumCheckInstance` - Folded single-instance (1 VP + 1 sum)
/// * `SumFoldProof` - SumFold proof with all verification data
pub fn prove_sumfold<F: PrimeField>(
    instances: Vec<SumCheckInstance<F>>,
    transcript: &mut IOPTranscript<F>,
) -> Result<(SumCheckInstance<F>, SumFoldProof<F>)> {
    if instances.is_empty() {
        return Err(DeSnarkError::InvalidParameters(
            "no instances to fold".into(),
        ));
    }
    if !instances.len().is_power_of_two() {
        return Err(DeSnarkError::InvalidParameters(format!(
            "number of instances must be power of 2, got {}",
            instances.len()
        )));
    }

    let m = instances.len();
    info!("prove_sumfold: folding {} instances", m);

    // Extract polys and sums from instances
    let (polys, sums): (Vec<_>, Vec<_>) = instances
        .into_iter()
        .map(|inst| (inst.poly, inst.sum))
        .unzip();

    // In debug mode, save copies for v1/v2 before v3 consumes the originals
    #[cfg(debug_assertions)]
    let (polys_v1, sums_v1, polys_v2, sums_v2) = {
        let p1 = polys.iter().map(|p| p.deep_copy()).collect::<Vec<_>>();
        let s1 = sums.clone();
        let p2 = polys.iter().map(|p| p.deep_copy()).collect::<Vec<_>>();
        let s2 = sums.clone();
        (p1, s1, p2, s2)
    };

    // ═══════════════════════════════════════════════════════════════
    // Run v3 (always) — production path, uses originals directly
    // ═══════════════════════════════════════════════════════════════
    let start = Instant::now();
    let (_proof_v3, _sum_t_v3, _aux_info_v3, folded_poly_v3, v_v3) =
        <PolyIOP<F> as SumCheck<F>>::sum_fold_v3(polys, sums, transcript)
            .map_err(|e| DeSnarkError::HyperPlonkError(format!("sum_fold v3 failed: {e}")))?;
    let dur_v3 = start.elapsed();
    info!("sum_fold v3: {:?}", dur_v3);
    info!("v v3: {:?}", v_v3);

    // ═══════════════════════════════════════════════════════════════
    // Run v1 & v2 and cross-validate (debug only)
    // ═══════════════════════════════════════════════════════════════
    #[cfg(debug_assertions)]
    {
        // Run v1
        let start = Instant::now();
        let mut transcript_v1 = <PolyIOP<F> as SumCheck<F>>::init_transcript();
        let (proof_v1, sum_t_v1, aux_info_v1, folded_poly_v1, v_v1) =
            <PolyIOP<F> as SumCheck<F>>::sum_fold(polys_v1, sums_v1, &mut transcript_v1)
                .map_err(|e| DeSnarkError::HyperPlonkError(format!("sum_fold v1 failed: {e}")))?;
        let dur_v1 = start.elapsed();
        info!("sum_fold v1: {:?}", dur_v1);
        info!("v v1: {:?}", v_v1);

        // Run v2
        let start = Instant::now();
        let mut transcript_v2 = <PolyIOP<F> as SumCheck<F>>::init_transcript();
        let (proof_v2, sum_t_v2, aux_info_v2, folded_poly_v2, v_v2) =
            <PolyIOP<F> as SumCheck<F>>::sum_fold_v2(polys_v2, sums_v2, &mut transcript_v2)
                .map_err(|e| DeSnarkError::HyperPlonkError(format!("sum_fold v2 failed: {e}")))?;
        let dur_v2 = start.elapsed();
        info!("sum_fold v2: {:?}", dur_v2);
        info!("v v2: {:?}", v_v2);

        // v1 vs v3
        assert_eq!(sum_t_v1, _sum_t_v3, "sum_t mismatch: v1 vs v3");
        assert_eq!(v_v1, v_v3, "v mismatch: v1 vs v3");
        assert_eq!(proof_v1, _proof_v3, "proof mismatch: v1 vs v3");
        assert_eq!(
            aux_info_v1.max_degree, _aux_info_v3.max_degree,
            "aux_info max_degree mismatch: v1 vs v3"
        );
        assert_eq!(
            aux_info_v1.num_variables, _aux_info_v3.num_variables,
            "aux_info num_variables mismatch: v1 vs v3"
        );
        for j in 0..folded_poly_v1.flattened_ml_extensions.len() {
            assert_eq!(
                folded_poly_v1.flattened_ml_extensions[j].evaluations,
                folded_poly_v3.flattened_ml_extensions[j].evaluations,
                "folded MLE[{}] mismatch: v1 vs v3",
                j
            );
        }

        // v2 vs v3
        assert_eq!(sum_t_v2, _sum_t_v3, "sum_t mismatch: v2 vs v3");
        assert_eq!(v_v2, v_v3, "v mismatch: v2 vs v3");
        assert_eq!(proof_v2, _proof_v3, "proof mismatch: v2 vs v3");
        assert_eq!(
            aux_info_v2.max_degree, _aux_info_v3.max_degree,
            "aux_info max_degree mismatch: v2 vs v3"
        );
        assert_eq!(
            aux_info_v2.num_variables, _aux_info_v3.num_variables,
            "aux_info num_variables mismatch: v2 vs v3"
        );
        for j in 0..folded_poly_v2.flattened_ml_extensions.len() {
            assert_eq!(
                folded_poly_v2.flattened_ml_extensions[j].evaluations,
                folded_poly_v3.flattened_ml_extensions[j].evaluations,
                "folded MLE[{}] mismatch: v2 vs v3",
                j
            );
        }

        info!(
            "✅ All 3 versions match! sum_t={:?}, v={:?}",
            sum_t_v1, v_v1
        );
        info!(
            "Timing: v1={:?}, v2={:?}, v3={:?} | speedup v1/v3={:.2}x, v2/v3={:.2}x",
            dur_v1,
            dur_v2,
            dur_v3,
            dur_v1.as_secs_f64() / dur_v3.as_secs_f64(),
            dur_v2.as_secs_f64() / dur_v3.as_secs_f64(),
        );
    }

    // Return v3 result (best single-machine version)
    let folded_instance = SumCheckInstance::new(folded_poly_v3, v_v3);
    let sumfold_proof = SumFoldProof::new(_proof_v3, _sum_t_v3, _aux_info_v3, v_v3);
    Ok((folded_instance, sumfold_proof))
}

/// Combine and verify K parties' SumFold proofs (pure verifier — no witness
/// data).
///
/// In the distributed SumFold protocol, all K parties share the same
/// Fiat-Shamir transcript: challenges are derived from **aggregated**
/// prover messages (element-wise sum across parties). Each party
/// contributes its partial prover messages, partial sum_t, and partial v.
///
/// This function:
/// 1. **Combines** K partial proofs into a single SumCheck proof by summing
///    prover messages element-wise per round
/// 2. **Verifies** the combined proof via `verify_sum_fold` (log₂(M) rounds)
/// 3. **Checks** consistency: `c == v_total · eq(ρ, r_b)`
///
/// # Arguments
/// * `party_proofs` - K SumFold proofs (partial contributions from each party)
///   All must share the same challenges (from the distributed protocol).
///
/// # Returns
/// * `F` - Verified total claimed sum `v_total = Σᵢ vᵢ`
#[instrument(level = "debug", skip_all, name = "merge_and_verify")]
pub fn merge_and_verify_sumfold<F: PrimeField>(party_proofs: Vec<SumFoldProof<F>>) -> Result<F> {
    let k = party_proofs.len();
    if k == 0 {
        return Err(DeSnarkError::InvalidParameters(
            "no proofs to verify".into(),
        ));
    }

    let num_rounds = party_proofs[0].q_aux_info.num_variables;
    info!(
        "merge_and_verify_sumfold: combining {} parties' proofs ({} rounds)",
        k, num_rounds
    );

    // Validate: all proofs have compatible structure
    for (i, sfp) in party_proofs.iter().enumerate().skip(1) {
        if sfp.q_aux_info.num_variables != num_rounds
            || sfp.q_aux_info.max_degree != party_proofs[0].q_aux_info.max_degree
        {
            return Err(DeSnarkError::InvalidParameters(format!(
                "Party {i}'s aux_info incompatible with party 0"
            )));
        }
        if sfp.proof.proofs.len() != num_rounds {
            return Err(DeSnarkError::InvalidParameters(format!(
                "Party {i} has {} round messages, expected {}",
                sfp.proof.proofs.len(),
                num_rounds
            )));
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 1: Combine K partial proofs into a single SumCheck proof
    // Sum prover messages element-wise per round
    // ═══════════════════════════════════════════════════════════════
    let mut combined_msgs = party_proofs[0].proof.proofs.clone();
    for sfp in &party_proofs[1..] {
        for (round, msg) in sfp.proof.proofs.iter().enumerate() {
            for (j, eval) in msg.evaluations.iter().enumerate() {
                combined_msgs[round].evaluations[j] += eval;
            }
        }
    }

    // Shared challenges (all parties derived the same challenges
    // from the aggregated prover messages in the distributed protocol)
    let combined_proof = IOPProof {
        point: party_proofs[0].proof.point.clone(),
        proofs: combined_msgs,
    };

    let combined_sum_t: F = party_proofs.iter().map(|p| p.sum_t).sum();
    let v_total: F = party_proofs.iter().map(|p| p.v).sum();

    debug!(
        "Combined {} proofs: sum_t={:?}, v_total={:?}",
        k, combined_sum_t, v_total
    );

    // ═══════════════════════════════════════════════════════════════
    // Step 2: Verify combined proof as a single SumCheck (log₂(M) rounds)
    // ═══════════════════════════════════════════════════════════════
    let (subclaim, rho) =
        verify_sum_fold(combined_sum_t, &combined_proof, &party_proofs[0].q_aux_info).map_err(
            |e| {
                DeSnarkError::HyperPlonkError(format!(
                    "Combined sumfold SumCheck verification failed: {e}"
                ))
            },
        )?;

    // ═══════════════════════════════════════════════════════════════
    // Step 3: Consistency check: c = v_total · eq(ρ, r_b)
    // ═══════════════════════════════════════════════════════════════
    let eq_poly = EqPolynomial::new(rho);
    let eq_val = eq_poly.evaluate(&subclaim.point);
    let expected_c = v_total * eq_val;
    if subclaim.expected_evaluation != expected_c {
        return Err(DeSnarkError::HyperPlonkError(format!(
            "Combined sumfold consistency check failed: \
             c={:?} != v_total·eq(ρ,rb)={:?}",
            subclaim.expected_evaluation, expected_c
        )));
    }

    info!(
        "✅ Combined proof verified: v_total={:?}, {} rounds",
        v_total, num_rounds
    );

    Ok(v_total)
}

/// Phase 3: HyperPianist proof - distributed SumCheck on the folded instance.
///
/// Runs `d_prove` (two-phase distributed SumCheck) on the folded VP.
/// Master returns `Some(Proof)` with the SumCheck proof;
/// workers return `None` (they participated in network aggregation only).
///
/// # Arguments
/// * `_pk` - Proving key (will be used for PCS openings in future)
/// * `instance` - Folded SumCheck instance (single VP + sum)
/// * `transcript` - Fiat-Shamir transcript (carries SumFold state)
///
/// # Returns
/// * `Option<IOPProof>` - `Some` on master, `None` on workers
pub fn prove_hyper_pianist<E: Pairing, PCS: HyperPlonkPCS<E>>(
    _pk: &ProvingKey<E, PCS>,
    instance: &SumCheckInstance<E::ScalarField>,
    transcript: Option<&mut IOPTranscript<E::ScalarField>>,
) -> Result<Option<IOPProof<E::ScalarField>>> {
    // Run distributed SumCheck on the folded instance
    let sumcheck_proof = <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::d_prove::<Net>(
        &instance.poly,
        transcript,
    )
    .map_err(|e| DeSnarkError::HyperPlonkError(format!("d_prove failed: {e}")))?;

    // Master has the proof; workers return None
    match sumcheck_proof {
        Some(sumcheck_proof) => {
            info!(
                "✅ d_prove complete: {} rounds, point dimension = {}",
                sumcheck_proof.proofs.len(),
                sumcheck_proof.point.len()
            );
            // TODO: PCS opening phase (accumulate polys, evaluate, d_multi_open)
            Ok(Some(sumcheck_proof))
        },
        None => Ok(None),
    }
}

/// Verify network connectivity with a synchronized round-trip challenge.
///
/// Protocol:
/// 1. Master generates a random challenge `r`
/// 2. Master broadcasts `r` to all workers
/// 3. Each worker sends `r + party_id` back to master
/// 4. Master verifies all responses
/// 5. Master broadcasts pass/fail to all workers
/// 6. All workers check the result
///
/// Must be called after `Net::init_from_file`.
pub fn verify_network() -> Result<()> {
    let party_id = Net::party_id();
    let n_parties = Net::n_parties();
    info!(
        "[Party {}] Starting network verification ({} parties)...",
        party_id, n_parties
    );

    // Step 1-2: Master generates and broadcasts random challenge
    let r: u64 = Net::recv_from_master_uniform(Net::am_master().then(|| {
        let r: u64 = test_rng().gen();
        info!("[Party 0] Broadcasting challenge: {}", r);
        r
    }));
    info!("[Party {}] Received challenge: {}", party_id, r);

    // Step 3: Each party sends (r + party_id) back to master
    // collected[i] is guaranteed to be party i's response (indexed by party ID in
    // DeMultiNet)
    let response = r.wrapping_add(party_id as u64);
    let collected: Option<Vec<u64>> = Net::send_to_master(&response);

    // Step 4: Master forwards all collected responses to every party
    let all_responses: Vec<u64> = Net::recv_from_master_uniform(collected);

    // Step 5: Each party verifies all responses locally
    for (i, resp) in all_responses.iter().enumerate() {
        let expected = r.wrapping_add(i as u64);
        if *resp != expected {
            return Err(DeSnarkError::NetworkError(format!(
                "Party {} verification failed: got {}, expected {}",
                i, resp, expected
            )));
        } else {
            debug!(
                "[Party {}] Verified response from party {}: got {}, expected {}",
                party_id, i, resp, expected
            );
        }
    }

    info!("✅ [Party {}] Network verification passed", party_id);
    Ok(())
}

/// Inner distributed proving pipeline - circuit-agnostic.
///
/// Takes virtual polynomials and their claimed sums, then runs:
/// 1. Distributed SumFold to aggregate instances into one
/// 2. Distributed SumCheck (HyperPianist) on the folded instance
/// 3. Assembles the combined proof
///
/// When `inst_dist` is `Some`, uses instance-level distribution:
/// each party's `polys` is a subset of the global instances, and
/// d_sumfold/d_prove/d_open operate with `party_vars = 0`.
///
/// # Arguments
/// * `polys` - This party's virtual polynomials (M for constraint-dist, M/K for instance-dist)
/// * `sums`  - This party's claimed sums
/// * `inst_dist` - Instance distribution config (None for constraint distribution)
///
/// # Returns
/// * `Option<Proof>` - Combined proof (`Some` on master, `None` on workers)
#[instrument(level = "debug", skip_all, name = "dist_prove_sumcheck")]
pub fn dist_prove_sumcheck<E: Pairing, PCS: HyperPlonkPCS<E>>(
    polys: Vec<VirtualPolynomial<E::ScalarField>>,
    sums: Vec<E::ScalarField>,
    inst_dist: Option<&crate::d_sumfold::InstanceDistConfig>,
) -> Result<(Option<Proof<E, PCS>>, f64, f64)> {
    // ═══════════════════════════════════════════════════════════════
    // K=1 fast path: bypass all networking, use local prove directly.
    // This is the optimal path when running a single party — no
    // send_to_master / recv_from_master overhead per round.
    // ═══════════════════════════════════════════════════════════════
    if Net::n_parties() == 1 {
        let mut transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();

        // Local SumFold (uses sum_fold_v3 — best single-machine path)
        let sumfold_timer = Instant::now();
        let instances: Vec<SumCheckInstance<E::ScalarField>> = polys
            .into_iter()
            .zip(sums)
            .map(|(poly, sum)| SumCheckInstance::new(poly, sum))
            .collect();
        let (folded_instance, sumfold_proof) =
            prove_sumfold(instances, &mut transcript)?;
        let sumfold_ms = sumfold_timer.elapsed().as_secs_f64() * 1000.0;

        #[cfg(debug_assertions)]
        {
            let v_total = merge_and_verify_sumfold(vec![sumfold_proof.clone()])?;
            info!("✅ [K=1] SumFold verify passed: v_total={:?}", v_total);
        }

        // Local SumCheck (standard prove, no network overhead)
        // d_prove with K=1 appends aux_info then runs num_vars rounds;
        // prove() does the same, so the transcript protocol matches.
        let sumcheck_timer = Instant::now();
        let hp_proof =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
                &folded_instance.poly,
                &mut transcript,
            )
            .map_err(|e| DeSnarkError::HyperPlonkError(format!("prove failed: {e}")))?;
        let sumcheck_ms = sumcheck_timer.elapsed().as_secs_f64() * 1000.0;

        // Concatenate sumfold + HyperPianist into one big proof
        let num_sumfold_rounds = sumfold_proof.proof.proofs.len();
        let mut combined_point = sumfold_proof.proof.point;
        combined_point.extend(hp_proof.point);
        let mut combined_proofs = sumfold_proof.proof.proofs;
        combined_proofs.extend(hp_proof.proofs);

        let proof = Proof {
            proof: IOPProof {
                point: combined_point,
                proofs: combined_proofs,
            },
            num_sumfold_rounds,
            sum_t: sumfold_proof.sum_t,
            q_aux_info: sumfold_proof.q_aux_info,
            v: sumfold_proof.v,
            selector_commits: None,
            witness_commits: None,
            batch_openings: None,
        };

        info!("✅ [K=1] dist_prove_sumcheck: local fast path complete (sumfold={:.1}ms, sumcheck={:.1}ms)", sumfold_ms, sumcheck_ms);
        return Ok((Some(proof), sumfold_ms, sumcheck_ms));
    }

    // ═══════════════════════════════════════════════════════════════
    // K>1 distributed path: use network-coordinated d_sumfold + d_prove
    // ═══════════════════════════════════════════════════════════════

    // Create a single transcript threaded through all proving phases.
    // Master holds the transcript; workers receive challenges via network.
    let mut transcript = <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();

    // Phase 2: Distributed SumFold aggregation
    let sumfold_timer = Instant::now();
    let transcript_opt = if Net::am_master() {
        Some(&mut transcript)
    } else {
        None
    };
    let (folded_instance, sumfold_proof) = if let Some(idc) = inst_dist {
        crate::d_sumfold::d_sumfold_ext::<E::ScalarField, Net>(polys, sums, idc, transcript_opt)?
    } else {
        crate::d_sumfold::d_sumfold::<E::ScalarField, Net>(polys, sums, transcript_opt)?
    };
    let sumfold_ms = sumfold_timer.elapsed().as_secs_f64() * 1000.0;

    #[cfg(debug_assertions)]
    if Net::am_master() {
        let v_total = merge_and_verify_sumfold(vec![sumfold_proof.clone()])?;
        info!("✅ SumFold verify passed: v_total={:?}", v_total);
    }

    // Phase 3: HyperPianist distributed SumCheck (operates on folded instance)
    // With instance distribution, party_vars = 0 (no constraint partitioning).
    let sumcheck_timer = Instant::now();
    let party_vars = if inst_dist.is_some() { 0 } else { ark_std::log2(Net::n_parties()) as usize };
    let transcript_opt = if Net::am_master() {
        Some(&mut transcript)
    } else {
        None
    };
    let hp_proof = <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::d_prove_with_party_vars::<Net>(
        &folded_instance.poly,
        transcript_opt,
        party_vars,
    )
    .map_err(|e| DeSnarkError::HyperPlonkError(format!("d_prove failed: {e}")))?;
    let sumcheck_ms = sumcheck_timer.elapsed().as_secs_f64() * 1000.0;

    // Concatenate sumfold + HyperPianist into one big proof
    let num_sumfold_rounds = sumfold_proof.proof.proofs.len();
    let proof = hp_proof.map(|hp| {
        let mut combined_point = sumfold_proof.proof.point;
        combined_point.extend(hp.point);
        let mut combined_proofs = sumfold_proof.proof.proofs;
        combined_proofs.extend(hp.proofs);
        Proof {
            proof: IOPProof {
                point: combined_point,
                proofs: combined_proofs,
            },
            num_sumfold_rounds,
            sum_t: sumfold_proof.sum_t,
            q_aux_info: sumfold_proof.q_aux_info,
            v: sumfold_proof.v,
            selector_commits: None,
            witness_commits: None,
            batch_openings: None,
        }
    });

    Ok((proof, sumfold_ms, sumcheck_ms))
}

/// Distributed SNARK prove - complete end-to-end pipeline.
///
/// Outer layer: handles circuit-specific operations (setup, circuit
/// generation, conversion to SumCheck instances), then delegates to
/// [`dist_prove_sumcheck`] for the circuit-agnostic distributed proving.
///
/// The network must be initialized (via `Net::init_from_file`) before calling.
/// The caller is responsible for `Net::deinit()` after this returns.
///
/// # Flow
/// 1. verify_network() - round-trip connectivity check
/// 2. setup(config) -> SRS
/// 3. make_circuit(config, srs) -> (PK, VK, Vec<MockCircuit>)
/// 4. circuits_to_sumcheck(pk, circuits) -> Vec<SumCheckInstance>
/// 5. dist_prove_sumcheck(polys, sums) -> Proof  (inner layer)
/// 6. verify_proof_eval(proof, pk, circuits, aux) — master-side eval check
///
/// # Arguments
/// * `config` - Protocol configuration
///
/// # Returns
/// * `VerifyingKey` - For verification
/// * `Option<Proof>` - Final SNARK proof (`Some` on master, `None` on workers)

/// Verify the proof against circuit data (master-side eval check).
///
/// Replays the full prover transcript (SumFold → HyperPianist) so that
/// the verifier derives the same Fiat-Shamir challenges as the prover,
/// then checks the final subclaim against the circuit polynomials.
///
/// When the proof contains PCS data (selector_commits, witness_commits,
/// batch_openings), verification proceeds as:
/// 1. Replay SumFold transcript, then verify HyperPianist SumCheck
/// 2. Extract evaluations from the batch opening proof
/// 3. Fold evaluations with eq(r_b, ·) weights and check gate equation
/// 4. PCS batch_verify on commitments + evaluations + opening proof
///
/// Without PCS data, falls back to local MLE evaluation (legacy path).
fn verify_proof_eval<E: Pairing, PCS: HyperPlonkPCS<E>>(
    proof: &Proof<E, PCS>,
    pk: &ProvingKey<E, PCS>,
    vk: &VerifyingKey<E, PCS>,
    circuits: &[MockCircuit<E::ScalarField>],
    instances_aux: &VPAuxInfo<E::ScalarField>,
) -> Result<()> {
    let num_vars = instances_aux.num_variables;
    let total_hp_rounds = proof.proof.proofs.len() - proof.num_sumfold_rounds;

    // ═══════════════════════════════════════════════════════════════
    // Step 1: Replay full transcript and verify HyperPianist SumCheck
    //
    // The prover's transcript was threaded: SumFold → d_prove.
    // We must replay SumFold operations first so the transcript state
    // matches, then verify the HyperPianist portion.
    // ═══════════════════════════════════════════════════════════════
    let mut transcript = <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();

    // Replay SumFold transcript operations
    transcript
        .append_serializable_element(b"aux info", &proof.q_aux_info)
        .map_err(|e| DeSnarkError::HyperPlonkError(format!("transcript replay: {e}")))?;
    let _rho: Vec<E::ScalarField> = transcript
        .get_and_append_challenge_vectors(b"sumfold rho", proof.num_sumfold_rounds)
        .map_err(|e| DeSnarkError::HyperPlonkError(format!("transcript replay: {e}")))?;
    for i in 0..proof.num_sumfold_rounds {
        transcript
            .append_serializable_element(b"prover msg", &proof.proof.proofs[i])
            .map_err(|e| {
                DeSnarkError::HyperPlonkError(format!("transcript replay round {i}: {e}"))
            })?;
        transcript
            .get_and_append_challenge(b"Internal round")
            .map_err(|e| {
                DeSnarkError::HyperPlonkError(format!("transcript replay challenge {i}: {e}"))
            })?;
    }

    // d_prove uses extended aux_info (num_variables includes party variables)
    let mut hp_aux_info = instances_aux.clone();
    hp_aux_info.num_variables = total_hp_rounds;

    let hp_proof = IOPProof {
        point: proof.proof.point[proof.num_sumfold_rounds..].to_vec(),
        proofs: proof.proof.proofs[proof.num_sumfold_rounds..].to_vec(),
    };

    let subclaim = <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
        proof.v,
        &hp_proof,
        &hp_aux_info,
        &mut transcript,
    )
    .map_err(|e| {
        DeSnarkError::HyperPlonkError(format!(
            "SumCheck verification of HyperPianist proof failed: {e}"
        ))
    })?;

    // ═══════════════════════════════════════════════════════════════
    // Step 2 + 3: Gate equation check + PCS verification
    // ═══════════════════════════════════════════════════════════════
    let r_b = &proof.proof.point[..proof.num_sumfold_rounds];
    let gate_func = &pk.params.gate_func;
    let num_selectors = circuits[0].index.selectors.len();
    let num_witnesses = circuits[0].witnesses.len();
    // Derive num_instances from proof witness commits (selector commits are
    // shared and have only num_sel entries; witness commits have M × num_wit).
    let num_instances = if let Some(ref wit_commits) = proof.witness_commits {
        wit_commits.len() / num_witnesses
    } else {
        circuits.len()
    };

    if let (Some(ref batch_proof), Some(ref sel_commits), Some(ref wit_commits)) = (
        &proof.batch_openings,
        &proof.selector_commits,
        &proof.witness_commits,
    ) {
        // ───────────────────────────────────────────────────────────
        // PCS path (commitment-folded):
        //
        // The batch proof contains (num_sel + num_wit) FOLDED
        // evaluations (prover already folded with eq(r_b, i)):
        //   evals[0..num_sel]            — folded selector evals
        //   evals[num_sel..num_sel+num_wit] — folded witness evals
        //
        // Gate check uses folded evals directly.
        // batch_verify uses folded commitments reconstructed via MSM.
        // ───────────────────────────────────────────────────────────
        let evals = &batch_proof.f_i_eval_at_point_i;
        let expected_num_evals = num_selectors + num_witnesses;
        if evals.len() != expected_num_evals {
            return Err(DeSnarkError::HyperPlonkError(format!(
                "batch proof eval count mismatch: got {}, expected {} (num_sel+num_wit={} + {})",
                evals.len(),
                expected_num_evals,
                num_selectors,
                num_witnesses
            )));
        }

        // Evals from batch proof are already folded by the prover
        let folded_sel_evals = evals[..num_selectors].to_vec();
        let folded_wit_evals = evals[num_selectors..].to_vec();

        let gate_eval = eval_f(gate_func, &folded_sel_evals, &folded_wit_evals)
            .map_err(|e| DeSnarkError::HyperPlonkError(format!("eval_f: {e}")))?;

        if subclaim.expected_evaluation != gate_eval {
            return Err(DeSnarkError::HyperPlonkError(format!(
                "PCS-path gate eval mismatch: subclaim={:?}, gate_eval={:?}",
                subclaim.expected_evaluation, gate_eval
            )));
        }
        info!(
            "✅ Gate equation check passed (PCS path): gate_eval={:?}",
            gate_eval
        );

        // ───────────────────────────────────────────────────────────
        // PCS batch_verify on FOLDED commitments
        //
        // Reconstruct folded commitments from individual commits
        // using eq(r_b, i) weights, then batch_verify on the
        // (num_sel + num_wit) folded commitments.
        // ───────────────────────────────────────────────────────────
        let eq_rb_vec = build_eq_x_r_vec(r_b)
            .map_err(|e| DeSnarkError::HyperPlonkError(format!("build_eq_x_r_vec: {e}")))?;

        // Selector commits are shared — scalar multiply with Σ_i eq(r_b,i)
        let mut folded_commits: Vec<Commitment<E>> =
            Vec::with_capacity(num_selectors + num_witnesses);
        let eq_rb_sum: E::ScalarField = eq_rb_vec.iter().copied().sum();
        for j in 0..num_selectors {
            let comm = E::G1MSM::msm_unchecked(&[sel_commits[j].0], &[eq_rb_sum]);
            folded_commits.push(Commitment(comm.into()));
        }

        // Fold witness commits: C_wit_j_fold = Σ_i eq(r_b,i) * C_wit_j^i
        for j in 0..num_witnesses {
            let bases: Vec<_> = (0..num_instances)
                .map(|i| wit_commits[i * num_witnesses + j].0)
                .collect();
            let comm_proj = E::G1MSM::msm_unchecked(&bases, &eq_rb_vec);
            folded_commits.push(Commitment(comm_proj.into()));
        }

        // Opening point = (r_phase1, r_party), i.e. everything after r_b
        let pcs_point = proof.proof.point[proof.num_sumfold_rounds..].to_vec();
        let points: Vec<Vec<E::ScalarField>> = vec![pcs_point; folded_commits.len()];

        // Replay PCS transcript with folded commitments
        let mut pcs_transcript =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();
        for c in &folded_commits {
            pcs_transcript
                .append_serializable_element(b"pcs_cm", c)
                .map_err(|e| DeSnarkError::HyperPlonkError(format!("pcs transcript: {e}")))?;
        }

        let pcs_ok = PCS::batch_verify(
            &vk.pcs_param,
            &folded_commits,
            &points,
            batch_proof,
            &mut pcs_transcript,
        )
        .map_err(|e| DeSnarkError::HyperPlonkError(format!("PCS batch_verify: {e}")))?;

        if !pcs_ok {
            return Err(DeSnarkError::HyperPlonkError(
                "PCS batch_verify failed: opening proof does not match commitments".to_string(),
            ));
        }
        info!(
            "✅ PCS batch_verify passed ({} folded commits)",
            folded_commits.len()
        );
    } else {
        // ───────────────────────────────────────────────────────────
        // Legacy path: evaluate MLEs locally (no PCS openings)
        //
        // The folded polynomial is:
        //   P(x) = Σ_p coeff_p · Π_{j∈prod_p} [Σ_i eq(r_b,i)·mle_j^i(x)]
        //
        // We fold individual MLE evaluations first (with eq(r_b,·) weights),
        // then compute the gate function on the folded evaluations.
        // ───────────────────────────────────────────────────────────
        let r_phase1 = &subclaim.point[..num_vars];

        let eq_rb_vec = build_eq_x_r_vec(r_b)
            .map_err(|e| DeSnarkError::HyperPlonkError(format!("build_eq_x_r_vec: {e}")))?;

        let mut folded_sel_evals = vec![E::ScalarField::from(0u64); num_selectors];
        let mut folded_wit_evals = vec![E::ScalarField::from(0u64); num_witnesses];

        // Selectors are shared: fold with scalar sum
        let eq_rb_sum: E::ScalarField = eq_rb_vec.iter().copied().sum();
        for (j, sel) in circuits[0].index.selectors.iter().enumerate() {
            let mle = DenseMultilinearExtension::from(sel);
            folded_sel_evals[j] = eq_rb_sum * mle.evaluate(r_phase1).unwrap();
        }
        // Witnesses are per-instance: fold with individual eq(r_b,i) weights
        for (i, circuit) in circuits.iter().enumerate() {
            let w = eq_rb_vec[i];
            for (j, wit) in circuit.witnesses.iter().enumerate() {
                let mle = DenseMultilinearExtension::from(wit);
                folded_wit_evals[j] += w * mle.evaluate(r_phase1).unwrap();
            }
        }

        let folded_eval = eval_f(gate_func, &folded_sel_evals, &folded_wit_evals)
            .map_err(|e| DeSnarkError::HyperPlonkError(format!("eval_f: {e}")))?;

        if subclaim.expected_evaluation != folded_eval {
            return Err(DeSnarkError::HyperPlonkError(format!(
                "Circuit-level eval mismatch: subclaim={:?}, folded_eval={:?}",
                subclaim.expected_evaluation, folded_eval
            )));
        }
        info!(
            "✅ Circuit-level eval verification passed (legacy): folded_eval={:?}",
            folded_eval
        );
    }

    Ok(())
}

pub fn dist_prove<E: Pairing>(
    config: &Config,
) -> Result<(
    VerifyingKey<E, MultilinearKzgPCS<E>>,
    Option<Proof<E, MultilinearKzgPCS<E>>>,
    BenchmarkTimings,
)> {
    let mut timings = BenchmarkTimings::default();

    // Step 0: Verify network connectivity (skip for K=1 — no network)
    if Net::n_parties() > 1 {
        verify_network()?;
    }

    // Phase 0: Load/generate SRS (not counted in setup_ms, matching HP)
    let srs = setup::<E, MultilinearKzgPCS<E>>(config)?;

    // Determine distribution mode:
    // - Constraint distribution (default): each party gets all M instances with N/K constraints
    // - K=1: fast path (no distribution)
    //
    // NOTE: Instance distribution (each party gets M/K instances with full N
    // constraints) is DISABLED because the folded VirtualPolynomial after
    // d_sumfold has degree > 1 products (e.g. q_L * w_1 in vanilla plonk).
    // With instance distribution and party_vars=0, d_prove aggregates partial
    // per-party evaluations, but the product structure means per-party
    // contributions are NOT additive shares of the global folded polynomial
    // (cross terms between instances on different parties are missing).
    // This causes verification failure ("PCS-path gate eval mismatch") and
    // subsequent protocol desync between master and workers.
    let use_instance_dist = false;

    let inst_dist_config = if use_instance_dist {
        Some(crate::d_sumfold::InstanceDistConfig {
            global_m: config.num_instances(),
            instance_offset: Net::party_id() * (config.num_instances() / config.num_parties()),
        })
    } else {
        None
    };

    // Phase 1a: Build circuits (not counted in setup_ms, matching HP)
    let circuits = if use_instance_dist {
        config.build_instance_distributed_circuits::<E::ScalarField>(Net::party_id())
    } else {
        config.build_partitioned_circuits_shared_sel::<E::ScalarField>()
    };
    let global_num_instances = config.num_instances();
    info!(
        "Circuits built: {} local instances (global M={}), {} witness columns, {} selector columns, {} public inputs each, mode={}",
        circuits.len(),
        global_num_instances,
        circuits[0].index.params.num_witness_columns(),
        circuits[0].index.params.num_selector_columns(),
        circuits[0].public_inputs.len(),
        if use_instance_dist { "instance-dist" } else { "constraint-dist" }
    );

    // Phase 1b: Preprocess (key extraction + PCS trim — this IS setup_ms,
    //           matching HyperPianist's d_preprocess scope)
    let setup_timer = Instant::now();
    info!(
        "Preprocessing: num_variables = {}, num_constraints = {}",
        circuits[0].index.num_variables(),
        circuits[0].index.params.num_constraints
    );
    let (mut pk, mut vk, _duration) =
        <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::preprocess(
            &circuits[0].index,
            &srs,
        )
        .map_err(|e| DeSnarkError::HyperPlonkError(e.to_string()))?;

    // Override PCS params: preprocess trims to log(N/K), but d_commit needs
    // log(N) because it extends local MLEs with log(K) party variables.
    let d_commit_num_vars = config.log_num_constraints;
    let (pcs_prover_param, pcs_verifier_param) =
        MultilinearKzgPCS::<E>::trim(&srs, None, Some(d_commit_num_vars))
            .map_err(|e| DeSnarkError::HyperPlonkError(format!("PCS trim for d_commit: {e}")))?;
    pk.pcs_param = pcs_prover_param;
    vk.pcs_param = pcs_verifier_param;

    // ═══════════════════════════════════════════════════════════════
    // Phase 1.5a (setup): Commit SHARED selector polynomials
    //
    // In real PLONK, selectors encode the fixed circuit structure and are
    // identical across all M instances. We commit them once during setup,
    // matching HyperPianist's d_preprocess which commits selector and
    // permutation oracles as part of key extraction (setup_ms).
    // ═══════════════════════════════════════════════════════════════
    let num_instances = circuits.len();
    let num_witnesses = circuits[0].witnesses.len();
    let num_selectors = circuits[0].index.selectors.len();
    let num_vars = pk.params.num_variables();
    let is_single_party = Net::n_parties() == 1;
    let use_standard_commit = is_single_party || use_instance_dist;

    // Build selector MLEs once (shared across all M instances)
    let selector_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> =
        circuits[0].index.selectors.iter()
            .map(|s| Arc::new(DenseMultilinearExtension::from(s)))
            .collect();

    // Commit selectors (num_sel only, NOT M × num_sel)
    let selector_commit_opts: Vec<Option<Commitment<E>>> = if use_standard_commit {
        selector_polys.iter()
            .map(|p| MultilinearKzgPCS::<E>::commit(&pk.pcs_param, p).map(Some))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| DeSnarkError::HyperPlonkError(format!("commit selectors: {e}")))?
    } else {
        // Constraint distribution: d_commit extends local polys with party dimension
        DeMkzg::<E>::batch_d_commit(&pk.pcs_param, &selector_polys)
            .map_err(|e| DeSnarkError::HyperPlonkError(format!("d_commit selectors: {e}")))?
    };
    info!(
        "✅ Setup: committed {} shared selector polys (d_commit={})",
        num_selectors, !use_standard_commit,
    );
    timings.setup_ms = setup_timer.elapsed().as_secs_f64() * 1000.0;

    // ═══════════════════════════════════════════════════════════════
    // Phase 1.5b (prover): Commit witness polynomials only
    //
    // Each of the M instances has its own witnesses. This matches
    // HyperPianist's d_prove which commits witness polys as the first
    // step of the proving phase.
    // ═══════════════════════════════════════════════════════════════
    let prover_timer = Instant::now();
    // Reset comm stats so they only cover the proving phase
    Net::reset_stats();
    // CPU measurement: scoped precisely to the proving phase
    let prove_cpu_before = {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        let user = usage.ru_utime.tv_sec as f64 * 1e3 + usage.ru_utime.tv_usec as f64 * 1e-3;
        let sys  = usage.ru_stime.tv_sec as f64 * 1e3 + usage.ru_stime.tv_usec as f64 * 1e-3;
        user + sys
    };
    let pcs_timer = Instant::now();

    // Build witness MLEs for all M instances
    let mut witness_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> =
        Vec::with_capacity(num_instances * num_witnesses);
    for circuit in &circuits {
        for w in &circuit.witnesses {
            witness_polys.push(Arc::new(DenseMultilinearExtension::from(w)));
        }
    }

    // Commit witnesses only (selectors already committed during setup)
    let witness_commit_opts: Vec<Option<Commitment<E>>> = if use_standard_commit {
        if use_instance_dist {
            // Instance distribution: gather all parties' witness commits on master
            let local_wit_commits: Vec<Commitment<E>> = witness_polys
                .iter()
                .map(|p| MultilinearKzgPCS::<E>::commit(&pk.pcs_param, p))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| DeSnarkError::HyperPlonkError(format!("commit witnesses: {e}")))?;
            let all_wit = Net::send_to_master(&local_wit_commits);
            if Net::am_master() {
                let mut all_wit_flat: Vec<Option<Commitment<E>>> = Vec::new();
                for party_commits in all_wit.unwrap() {
                    all_wit_flat.extend(party_commits.into_iter().map(Some));
                }
                all_wit_flat
            } else {
                local_wit_commits.into_iter().map(Some).collect()
            }
        } else {
            // K=1: standard commit
            witness_polys.iter()
                .map(|p| MultilinearKzgPCS::<E>::commit(&pk.pcs_param, p).map(Some))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| DeSnarkError::HyperPlonkError(format!("commit witnesses: {e}")))?
        }
    } else {
        // Constraint distribution: d_commit with party aggregation
        DeMkzg::<E>::batch_d_commit(&pk.pcs_param, &witness_polys)
            .map_err(|e| DeSnarkError::HyperPlonkError(format!("d_commit witnesses: {e}")))?
    };
    timings.d_commit_ms = pcs_timer.elapsed().as_secs_f64() * 1000.0;
    info!(
        "✅ PCS d_commit: {} witnesses ({} instances × {}) in {:.1}ms",
        witness_polys.len(), num_instances, num_witnesses, timings.d_commit_ms
    );

    // ═══════════════════════════════════════════════════════════════
    // Phase 1.5c: Convert circuits to SumCheck instances
    // ═══════════════════════════════════════════════════════════════
    // Reuse the selector/witness MLEs built above. Selector MLEs are shared
    // across all instances (num_sel entries); witness MLEs are per-instance
    // (M × num_wit entries).
    let mut polys: Vec<VirtualPolynomial<E::ScalarField>> = Vec::with_capacity(num_instances);
    let mut sums: Vec<E::ScalarField> = Vec::with_capacity(num_instances);
    for i in 0..num_instances {
        let witness_mles =
            &witness_polys[i * num_witnesses..(i + 1) * num_witnesses];
        let poly = build_f(&pk.params.gate_func, num_vars, &selector_polys, witness_mles)
            .map_err(|e| DeSnarkError::HyperPlonkError(e.to_string()))?;
        polys.push(poly);
        sums.push(E::ScalarField::from(0u64));
    }

    // Save aux_info before polys are consumed (needed for verification)
    let instances_aux = polys[0].aux_info.clone();

    // ═══════════════════════════════════════════════════════════════
    // Phase 2+3: Circuit-agnostic distributed proving (SumFold + SumCheck)
    // ═══════════════════════════════════════════════════════════════
    let (iop_proof, sumfold_ms, sumcheck_ms) = dist_prove_sumcheck::<E, MultilinearKzgPCS<E>>(polys, sums, inst_dist_config.as_ref())?;
    timings.sumfold_ms = sumfold_ms;
    timings.sumcheck_ms = sumcheck_ms;

    // ═══════════════════════════════════════════════════════════════
    // Phase 4: Commitment folding + PCS batch opening
    //
    // After SumFold, r_b is known. We fold M per-instance polynomials
    // into 1 using eq(r_b, i) weights, then open only (num_sel + num_wit)
    // folded polynomials instead of M*(num_sel + num_wit).
    //
    // Steps:
    //   4a. Broadcast r_b + pcs_point to all parties
    //   4b. Compute eq(r_b, i) folding weights
    //   4c. Fold selector and witness polynomials locally
    //   4d. Master folds commitments via MSM
    //   4e. Evaluate folded polys at local_point, aggregate across parties
    //   4f. d_multi_open on (num_sel + num_wit) folded polynomials
    // ═══════════════════════════════════════════════════════════════

    // 4a: Extract r_b + PCS opening point
    // K=1: extract directly from proof (no network broadcast needed)
    // K>1: master broadcasts to all workers
    let (r_b_vec, pcs_point): (Vec<E::ScalarField>, Vec<E::ScalarField>) = if is_single_party {
        let proof_ref = iop_proof.as_ref().unwrap();
        let nsf = proof_ref.num_sumfold_rounds;
        (
            proof_ref.proof.point[..nsf].to_vec(),
            proof_ref.proof.point[nsf..].to_vec(),
        )
    } else if Net::am_master() {
        let proof_ref = iop_proof.as_ref().unwrap();
        let nsf = proof_ref.num_sumfold_rounds;
        let r_b = proof_ref.proof.point[..nsf].to_vec();
        let pcs_pt = proof_ref.proof.point[nsf..].to_vec();
        Net::recv_from_master_uniform(Some((r_b, pcs_pt)))
    } else {
        Net::recv_from_master_uniform(None)
    };

    info!(
        "PCS opening point dimension: {} (num_vars={} + party_vars={})",
        pcs_point.len(),
        num_vars,
        pcs_point.len().saturating_sub(num_vars)
    );

    // 4b: Compute eq(r_b, i) weights for polynomial/commitment folding
    let eq_rb_vec = build_eq_x_r_vec(&r_b_vec)
        .map_err(|e| DeSnarkError::HyperPlonkError(format!("build_eq_x_r_vec for folding: {e}")))?;

    let fold_timer = Instant::now();

    // 4c: Each party folds its local polynomials with eq(r_b, i) weights.
    //
    //     Selectors are shared: sel_j_fold(x) = sel_j(x) · (Σ_i eq(r_b,i))
    //     Witnesses are per-instance: wit_j_fold(x) = Σ_i eq(r_b,i) · wit_j^i(x)
    //
    //     Instance-dist: use global eq weight eq_rb_vec[offset + i] for local instance i.
    //     Each party computes a PARTIAL fold; these are additive shares of the global fold.
    let instance_offset = inst_dist_config.as_ref().map_or(0, |c| c.instance_offset);

    // Selectors are shared: scale by Σ_i eq(r_b, i) (scalar multiply)
    let eq_rb_sum: E::ScalarField = eq_rb_vec[instance_offset..instance_offset + num_instances]
        .iter().copied().sum();
    let folded_sel_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> =
        cfg_into_iter!(0..num_selectors)
            .map(|j| {
                let poly_nv = selector_polys[j].num_vars;
                let folded_evals: Vec<E::ScalarField> = selector_polys[j].evaluations
                    .iter()
                    .map(|&v| eq_rb_sum * v)
                    .collect();
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    poly_nv,
                    folded_evals,
                ))
            })
            .collect();

    let folded_wit_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> =
        cfg_into_iter!(0..num_witnesses)
            .map(|j| {
                let poly_nv = witness_polys[j].num_vars;
                let n = 1 << poly_nv;
                let mut folded_evals = vec![E::ScalarField::from(0u64); n];
                for i in 0..num_instances {
                    let w = eq_rb_vec[instance_offset + i];
                    let src = &witness_polys[i * num_witnesses + j].evaluations;
                    for (k, v) in folded_evals.iter_mut().enumerate() {
                        *v += w * src[k];
                    }
                }
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    poly_nv,
                    folded_evals,
                ))
            })
            .collect();

    // 4d: Master folds commitments
    //
    //     Selectors are shared: C_sel_j_fold = C_sel_j · (Σ_i eq(r_b,i))
    //     Witnesses are per-instance: C_wit_j_fold = Σ_i eq(r_b,i) · C_wit_j^i
    //
    //     Constraint-dist / K=1: master has all M witness commits (local).
    //     Instance-dist: master has global_M witness commits (from gather).
    let folded_sel_commit_opts: Vec<Option<Commitment<E>>>;
    let folded_wit_commit_opts: Vec<Option<Commitment<E>>>;

    if Net::am_master() {
        let sel_commits: Vec<Commitment<E>> = selector_commit_opts
            .iter()
            .map(|c| c.clone().unwrap())
            .collect();
        let wit_commits: Vec<Commitment<E>> = witness_commit_opts
            .iter()
            .map(|c| c.clone().unwrap())
            .collect();

        // Selectors are shared: scalar multiply with Σ_i eq(r_b,i)
        let eq_rb_sum_for_commits: E::ScalarField = eq_rb_vec.iter().copied().sum();
        folded_sel_commit_opts = cfg_into_iter!(0..num_selectors)
            .map(|j| {
                let comm = E::G1MSM::msm_unchecked(&[sel_commits[j].0], &[eq_rb_sum_for_commits]);
                Some(Commitment(comm.into()))
            })
            .collect();

        // Witnesses are per-instance: fold via MSM
        folded_wit_commit_opts = cfg_into_iter!(0..num_witnesses)
            .map(|j| {
                let bases: Vec<_> = (0..global_num_instances)
                    .map(|i| wit_commits[i * num_witnesses + j].0)
                    .collect();
                let comm_proj = E::G1MSM::msm_unchecked(&bases, &eq_rb_vec);
                Some(Commitment(comm_proj.into()))
            })
            .collect();
    } else {
        folded_sel_commit_opts = vec![None; num_selectors];
        folded_wit_commit_opts = vec![None; num_witnesses];
    }

    timings.fold_ms = fold_timer.elapsed().as_secs_f64() * 1000.0;
    info!(
        "✅ Commitment folding completed in {:.1}ms ({} shared sel + {} wit → {} folded polys)",
        timings.fold_ms,
        num_selectors,
        num_instances * num_witnesses,
        num_selectors + num_witnesses,
    );

    // 4e: Each party evaluates (num_sel + num_wit) FOLDED polys at local_point
    let local_point = &pcs_point[..num_vars];
    let folded_total = num_selectors + num_witnesses;

    let mut local_evals: Vec<E::ScalarField> = Vec::with_capacity(folded_total);
    for sel_poly in &folded_sel_polys {
        local_evals.push(sel_poly.evaluate(local_point).unwrap());
    }
    for wit_poly in &folded_wit_polys {
        local_evals.push(wit_poly.evaluate(local_point).unwrap());
    }

    // Aggregate local evaluations across parties to get global evaluations
    // K=1: local_evals ARE global_evals (no party dimension)
    // Instance-dist: gather partial evals, sum directly (no eq_party weighting — party_vars=0)
    // Constraint-dist: aggregate with eq(r_party, party_id) weights
    let global_evals: Vec<E::ScalarField> = if is_single_party {
        local_evals
    } else if use_instance_dist {
        // Instance distribution: partial folds are additive shares → sum directly
        let all_local_evals: Option<Vec<Vec<E::ScalarField>>> = Net::send_to_master(&local_evals);
        if Net::am_master() {
            let all_evals = all_local_evals.unwrap();
            let num_polys = all_evals[0].len();
            let mut global = vec![E::ScalarField::from(0u64); num_polys];
            for party_evals in &all_evals {
                for (j, &eval) in party_evals.iter().enumerate() {
                    global[j] += eval;
                }
            }
            Net::recv_from_master_uniform(Some(global))
        } else {
            Net::recv_from_master_uniform(None)
        }
    } else {
        // Constraint distribution: aggregate with eq(r_party, party_id) weights
        let all_local_evals: Option<Vec<Vec<E::ScalarField>>> = Net::send_to_master(&local_evals);
        if Net::am_master() {
            let all_evals = all_local_evals.unwrap();
            let r_party = &pcs_point[num_vars..];
            let eq_party = build_eq_x_r_vec(r_party)
                .map_err(|e| DeSnarkError::HyperPlonkError(format!("build_eq_x_r_vec: {e}")))?;
            let num_polys = all_evals[0].len();
            let mut global = vec![E::ScalarField::from(0u64); num_polys];
            for (party_idx, party_evals) in all_evals.iter().enumerate() {
                let w = eq_party[party_idx];
                for (j, &eval) in party_evals.iter().enumerate() {
                    global[j] += w * eval;
                }
            }
            Net::recv_from_master_uniform(Some(global))
        } else {
            Net::recv_from_master_uniform(None)
        }
    };

    // 4f: d_multi_open on FOLDED polynomials
    let mut all_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> =
        Vec::with_capacity(folded_total);
    all_polys.extend(folded_sel_polys.iter().cloned());
    all_polys.extend(folded_wit_polys.iter().cloned());

    let points: Vec<Vec<E::ScalarField>> = vec![pcs_point.clone(); folded_total];

    let mut pcs_transcript =
        <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::init_transcript();

    if Net::am_master() {
        // Append FOLDED commits to PCS transcript
        let folded_commits: Vec<Commitment<E>> = folded_sel_commit_opts
            .iter()
            .chain(folded_wit_commit_opts.iter())
            .map(|c| c.clone().unwrap())
            .collect();
        for c in &folded_commits {
            pcs_transcript
                .append_serializable_element(b"pcs_cm", c)
                .map_err(|e| DeSnarkError::HyperPlonkError(format!("pcs transcript: {e}")))?;
        }
    }

    let open_timer = Instant::now();
    // K=1: use standard multi_open (no d_prove / network overhead)
    // Instance-dist: d_multi_open with party_vars=0 (each party has additive share)
    // Constraint-dist: d_multi_open with default party_vars=log2(K)
    let batch_proof_opt: Option<BatchProof<E, MultilinearKzgPCS<E>>> = if is_single_party {
        let bp = MultilinearKzgPCS::<E>::multi_open(
            &pk.pcs_param,
            &all_polys,
            &points,
            &global_evals,
            &mut pcs_transcript,
        )
        .map_err(|e| DeSnarkError::HyperPlonkError(format!("multi_open: {e}")))?;
        Some(bp)
    } else if use_instance_dist {
        // Instance distribution: party_vars=0, each party's partial fold is an additive share.
        // KZG homomorphic property: sum of partial-fold commits = commit of full fold.
        DeMkzg::<E>::d_multi_open_with_party_vars(
            &pk.pcs_param,
            all_polys,
            &points,
            &global_evals,
            &mut pcs_transcript,
            0,
        )
        .map_err(|e| DeSnarkError::HyperPlonkError(format!("d_multi_open (instance-dist): {e}")))?
    } else {
        DeMkzg::<E>::d_multi_open(
            &pk.pcs_param,
            all_polys,
            &points,
            &global_evals,
            &mut pcs_transcript,
        )
        .map_err(|e| DeSnarkError::HyperPlonkError(format!("d_multi_open: {e}")))?
    };
    timings.multi_open_ms = open_timer.elapsed().as_secs_f64() * 1000.0;
    info!(
        "✅ PCS multi_open completed in {:.1}ms ({} folded polys: {} sel + {} wit)",
        timings.multi_open_ms,
        folded_total,
        num_selectors,
        num_witnesses
    );

    // ═══════════════════════════════════════════════════════════════
    // Phase 5: Assemble complete proof (master only)
    // ═══════════════════════════════════════════════════════════════
    let proof = if Net::am_master() {
        let mut proof = iop_proof.unwrap();
        proof.selector_commits = Some(
            selector_commit_opts
                .into_iter()
                .map(|c| c.unwrap())
                .collect(),
        );
        proof.witness_commits = Some(
            witness_commit_opts
                .into_iter()
                .map(|c| c.unwrap())
                .collect(),
        );
        proof.batch_openings = batch_proof_opt;
        Some(proof)
    } else {
        None
    };
    timings.prover_ms = prover_timer.elapsed().as_secs_f64() * 1000.0;
    timings.prove_wall_ms = timings.prover_ms;
    // CPU measurement: scoped precisely to the proving phase
    timings.prove_cpu_ms = {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        let user = usage.ru_utime.tv_sec as f64 * 1e3 + usage.ru_utime.tv_usec as f64 * 1e-3;
        let sys  = usage.ru_stime.tv_sec as f64 * 1e3 + usage.ru_stime.tv_usec as f64 * 1e-3;
        (user + sys) - prove_cpu_before
    };
    // Capture comm stats from the proving phase only (before verify)
    let prove_stats = Net::stats();
    timings.comm_sent = prove_stats.bytes_sent;
    timings.comm_recv = prove_stats.bytes_recv;

    // ═══════════════════════════════════════════════════════════════
    // Phase 6 (master only): Verify the proof
    // ═══════════════════════════════════════════════════════════════
    let verifier_timer = Instant::now();
    if let Some(ref proof) = proof {
        verify_proof_eval::<E, MultilinearKzgPCS<E>>(proof, &pk, &vk, &circuits, &instances_aux)?;
    }
    timings.verifier_ms = verifier_timer.elapsed().as_secs_f64() * 1000.0;

    Ok((vk, proof, timings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::GateType;

    #[test]
    fn test_build_partitioned_circuits() {
        use ark_bn254::Fr;
        let config = Config::new(2, 10, GateType::Vanilla, 2);
        let circuits = config.build_partitioned_circuits::<Fr>();

        // M = 4 instances
        assert_eq!(circuits.len(), 4);

        // Each partition: 1024 / 4 = 256 constraints
        for circuit in &circuits {
            assert_eq!(circuit.index.params.num_constraints, 256);
            assert!(circuit.is_satisfied());
        }
    }

    /// End-to-end test: build circuits → convert to SumCheck instances →
    /// prove_sumfold. Validates that v1, v2, v3 produce identical results
    /// on real circuit polynomials.
    #[test]
    fn test_prove_sumfold_e2e() {
        use ark_bn254::{Bn254, Fr};
        use subroutines::MultilinearKzgPCS;

        // ν=2 → M=4 instances, μ=10 → N=1024, κ=2 → K=4 parties
        let config = Config::new(2, 10, GateType::Vanilla, 2);

        // Setup
        let srs = setup::<Bn254, MultilinearKzgPCS<Bn254>>(&config).expect("SRS generation failed");

        // Make circuit
        let (pk, _vk, circuits) = make_circuit::<Bn254, MultilinearKzgPCS<Bn254>>(&config, &srs)
            .expect("make_circuit failed");

        // Convert to SumCheck instances
        let instances = circuits_to_sumcheck::<Bn254, MultilinearKzgPCS<Bn254>>(&pk, &circuits)
            .expect("circuits_to_sumcheck failed");

        assert_eq!(instances.len(), 4);
        for inst in &instances {
            assert_eq!(inst.sum, Fr::from(0u64));
        }

        // Prove sumfold (runs v1, v2, v3 and cross-validates)
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let (folded, _proof) =
            prove_sumfold(instances, &mut transcript).expect("prove_sumfold failed");

        // The folded polynomial should have the same num_variables as the original
        // instances
        assert_eq!(
            folded.poly.aux_info.num_variables,
            config.log_num_constraints - config.log_num_parties
        );
        println!(
            "Folded instance: num_vars={}, max_degree={}, v={:?}",
            folded.poly.aux_info.num_variables, folded.poly.aux_info.max_degree, folded.sum
        );
    }

    /// Test merge_and_verify_sumfold with K=1 (trivial combine).
    ///
    /// With K=1 the combined proof equals the original — verification
    /// replays the same transcript and succeeds.
    /// K>1 requires the distributed SumFold prover (shared challenges).
    #[test]
    fn test_merge_and_verify_sumfold() {
        use ark_bn254::{Bn254, Fr};
        use subroutines::MultilinearKzgPCS;

        // ν=2 → M=4 instances, μ=10 → N=1024, κ=2 → K=4 parties
        let config = Config::new(2, 10, GateType::Vanilla, 2);

        let srs = setup::<Bn254, MultilinearKzgPCS<Bn254>>(&config).expect("SRS generation failed");
        let (pk, _vk, _) = make_circuit::<Bn254, MultilinearKzgPCS<Bn254>>(&config, &srs)
            .expect("make_circuit failed");

        // K=1: single party folds M=4 instances
        let circuits = config.build_partitioned_circuits::<Fr>();
        let instances = circuits_to_sumcheck::<Bn254, MultilinearKzgPCS<Bn254>>(&pk, &circuits)
            .expect("circuits_to_sumcheck failed");
        assert_eq!(instances.len(), 4);

        for inst in &instances {
            assert_eq!(inst.sum, Fr::from(0u64));
        }

        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let (_folded, proof) =
            prove_sumfold(instances, &mut transcript).expect("prove_sumfold failed");

        // Combine + verify with K=1 (trivial: combined proof == original)
        let v_total =
            merge_and_verify_sumfold(vec![proof]).expect("merge_and_verify_sumfold failed");

        println!("Verified: v_total={:?}", v_total);
    }

    /// Test prove_sumfold with M=1 (single instance, no folding).
    ///
    /// This was previously a crash: length=log2(1)=0 caused usize underflow.
    #[test]
    fn test_prove_sumfold_m1() {
        use ark_bn254::{Bn254, Fr};
        use subroutines::MultilinearKzgPCS;

        // ν=0 → M=1 instance, μ=10 → N=1024, κ=0 → K=1 party
        let config = Config::new(0, 10, GateType::Vanilla, 0);

        let srs = setup::<Bn254, MultilinearKzgPCS<Bn254>>(&config).expect("SRS generation failed");
        let (pk, _vk, circuits) = make_circuit::<Bn254, MultilinearKzgPCS<Bn254>>(&config, &srs)
            .expect("make_circuit failed");

        let instances = circuits_to_sumcheck::<Bn254, MultilinearKzgPCS<Bn254>>(&pk, &circuits)
            .expect("circuits_to_sumcheck failed");
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].sum, Fr::from(0u64));

        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let (folded, proof) = prove_sumfold(instances, &mut transcript)
            .expect("prove_sumfold with M=1 should succeed");

        // M=1: 0 rounds, folded poly is the original
        assert_eq!(proof.proof.proofs.len(), 0);
        assert_eq!(proof.proof.point.len(), 0);
        assert_eq!(proof.q_aux_info.num_variables, 0);
        assert_eq!(
            folded.poly.aux_info.num_variables,
            config.log_num_constraints - config.log_num_parties
        );
        println!("M=1 test passed: v={:?}, sum_t={:?}", proof.v, proof.sum_t);
    }

    /// Test merge_and_verify_sumfold with M=1 proof (verification round-trip).
    #[test]
    fn test_merge_and_verify_sumfold_m1() {
        use ark_bn254::{Bn254, Fr};
        use subroutines::MultilinearKzgPCS;

        let config = Config::new(0, 10, GateType::Vanilla, 0);

        let srs = setup::<Bn254, MultilinearKzgPCS<Bn254>>(&config).expect("SRS generation failed");
        let (pk, _vk, circuits) = make_circuit::<Bn254, MultilinearKzgPCS<Bn254>>(&config, &srs)
            .expect("make_circuit failed");

        let instances = circuits_to_sumcheck::<Bn254, MultilinearKzgPCS<Bn254>>(&pk, &circuits)
            .expect("circuits_to_sumcheck failed");
        assert_eq!(instances.len(), 1);

        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let (_folded, proof) =
            prove_sumfold(instances, &mut transcript).expect("prove_sumfold failed");

        // Verify the M=1 proof
        let v_total = merge_and_verify_sumfold(vec![proof])
            .expect("merge_and_verify_sumfold with M=1 should succeed");

        println!("M=1 verification passed: v_total={:?}", v_total);
    }
}
