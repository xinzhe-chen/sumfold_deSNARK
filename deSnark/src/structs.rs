//! Core data structures for deSnark distributed SNARK protocol.

use arithmetic::{VPAuxInfo, VirtualPolynomial};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalSerialize, Compress};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Deserialize;
use std::path::Path;
use subroutines::pcs::PolynomialCommitmentScheme;
use subroutines::poly_iop::prelude::IOPProof;
use subroutines::{BatchProof, Commitment};

/// Top-level TOML structure: `[config]` + `[network]`.
#[derive(Deserialize)]
struct TomlTop {
    config: Config,
    network: NetworkConfig,
}

// Re-export hyperplonk structures as the single source of truth
pub use hyperplonk::prelude::{
    CustomizedGates, HyperPlonkIndex, HyperPlonkParams, HyperPlonkProvingKey,
    HyperPlonkVerifyingKey, MockCircuit, SelectorColumn, WitnessColumn,
};

/// Gate type for the circuit - factory for CustomizedGates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateType {
    /// Vanilla PLONK gate: q_L w_1 + q_R w_2 + q_O w_3 + q_M w_1 w_2 + q_C = 0
    #[default]
    Vanilla,
}

impl GateType {
    /// Convert to HyperPlonk CustomizedGates.
    pub fn to_gate(&self) -> CustomizedGates {
        match self {
            GateType::Vanilla => CustomizedGates::vanilla_plonk_gate(),
        }
    }
}

/// Configuration for the distributed SNARK protocol.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct Config {
    /// ν (nu) - log of number of instances
    pub log_num_instances: usize,
    /// μ (mu) - log of number of constraints
    pub log_num_constraints: usize,
    /// Gate type
    pub gate_type: GateType,
    /// κ (kappa) - log of number of parties
    pub log_num_parties: usize,
    /// Optional path for SRS file cache.
    /// If set, SRS is loaded from this file when it exists and is large enough;
    /// otherwise generated and saved here.
    pub srs_path: Option<String>,
}

impl Config {
    /// Create a new Config.
    pub fn new(
        log_num_instances: usize,
        log_num_constraints: usize,
        gate_type: GateType,
        log_num_parties: usize,
    ) -> Self {
        Self {
            log_num_instances,
            log_num_constraints,
            gate_type,
            log_num_parties,
            srs_path: None,
        }
    }

    /// Load Config and NetworkConfig from a TOML file.
    ///
    /// The TOML file should have `[config]` and `[network]` sections:
    /// ```toml
    /// [config]
    /// log_num_instances = 2
    /// log_num_constraints = 10
    /// gate_type = "vanilla"
    /// log_num_parties = 2
    ///
    /// [network]
    /// hosts_file = "hosts.txt"
    /// ```
    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<(Config, NetworkConfig), String> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read {}: {e}", path.as_ref().display()))?;
        let top: TomlTop = toml::from_str(&content)
            .map_err(|e| format!("Failed to parse TOML: {e}"))?;
        Ok((top.config, top.network))
    }

    // Computed properties - no redundant storage
    /// M - number of instances
    pub fn num_instances(&self) -> usize {
        1 << self.log_num_instances
    }
    /// N - number of constraints
    pub fn num_constraints(&self) -> usize {
        1 << self.log_num_constraints
    }
    /// K - number of parties
    pub fn num_parties(&self) -> usize {
        1 << self.log_num_parties
    }
    /// t - number of witness columns
    pub fn num_witness_columns(&self) -> usize {
        self.gate_type.to_gate().num_witness_columns()
    }
    /// Number of selector columns
    pub fn num_selector_columns(&self) -> usize {
        self.gate_type.to_gate().num_selector_columns()
    }

    /// Build a single mock circuit with all constraints.
    pub fn build_mock_circuit<F: PrimeField>(&self) -> MockCircuit<F> {
        MockCircuit::new(self.num_constraints(), &self.gate_type.to_gate())
    }

    /// Build M mock circuits for one sub-prover (sequential).
    /// Each sub-prover has all M instances, but each instance only has N/K constraints.
    /// Uses a single RNG across all instances so each circuit has different random data.
    pub fn build_partitioned_circuits<F: PrimeField>(&self) -> Vec<MockCircuit<F>> {
        let constraints_per_party = self.num_constraints() / self.num_parties();
        let gate = self.gate_type.to_gate();
        let mut rng = ark_std::test_rng();
        (0..self.num_instances())
            .map(|_| MockCircuit::new_with_rng(constraints_per_party, &gate, &mut rng))
            .collect()
    }

    /// Build M mock circuits in parallel using Rayon.
    /// Each sub-prover has all M instances, but each instance only has N/K constraints.
    /// Uses per-instance seeds derived from a master RNG for deterministic parallelism.
    #[cfg(feature = "parallel")]
    pub fn build_partitioned_circuits_par<F: PrimeField>(&self) -> Vec<MockCircuit<F>> {
        use ark_std::rand::{RngCore, SeedableRng};
        let constraints_per_party = self.num_constraints() / self.num_parties();
        let gate = self.gate_type.to_gate();
        let mut master_rng = ark_std::test_rng();
        let seeds: Vec<u64> = (0..self.num_instances())
            .map(|_| master_rng.next_u64())
            .collect();
        seeds
            .into_par_iter()
            .map(|seed| {
                let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(seed);
                MockCircuit::new_with_rng(constraints_per_party, &gate, &mut rng)
            })
            .collect()
    }
}

/// SumCheck instance: a single virtual polynomial and its claimed sum.
///
/// This is the circuit-agnostic representation used by the SumFold protocol.
/// Any circuit type (Plonk, R1CS, custom) can produce a `SumCheckInstance`
/// via conversion to `VirtualPolynomial`.
#[derive(Clone, Debug)]
pub struct SumCheckInstance<F: PrimeField> {
    /// Virtual polynomial representing the constraint system
    pub poly: VirtualPolynomial<F>,
    /// Claimed sum (typically zero for a satisfiable circuit)
    pub sum: F,
}

impl<F: PrimeField> SumCheckInstance<F> {
    pub fn new(poly: VirtualPolynomial<F>, sum: F) -> Self {
        Self { poly, sum }
    }

    pub fn aux_info(&self) -> &VPAuxInfo<F> {
        &self.poly.aux_info
    }
}

/// Proof for the distributed SNARK.
///
/// Contains a single combined SumCheck proof whose rounds are:
///   sumfold rounds (log₂(M)) ++ HyperPianist rounds (num_vars + log₂(K))
/// plus the SumFold metadata needed for verification,
/// and PCS commitments and opening proofs.
///
/// ## PCS layout
///
/// The batch opening covers polynomials in this order (all at point r_phase1):
///
/// | Group | Count | Description |
/// |-------|-------|-------------|
/// | Selector polys | `num_selectors` | Shared across all M instances |
/// | Folded witness polys | `num_witnesses` | wit_j_fold = Σ_i eq(r_b,i) · wit_j^i |
///
/// Total opened polynomials = `num_selectors + num_witnesses` (after folding).
/// The proof stores M × num_witnesses individual witness commits so the
/// verifier can reconstruct the folded commitments via MSM.
#[derive(Clone, Debug)]
pub struct Proof<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    /// Combined SumCheck proof: sumfold rounds ++ HyperPianist rounds
    pub proof: IOPProof<E::ScalarField>,
    /// Number of sumfold rounds (= log₂(M)), for splitting during verification
    pub num_sumfold_rounds: usize,
    /// Weighted sum: sum_t = Σᵢ eq(ρ, i) · sᵢ
    pub sum_t: E::ScalarField,
    /// Aux info for the folding SumCheck (max_degree, num_variables = log₂(m))
    pub q_aux_info: VPAuxInfo<E::ScalarField>,
    /// Claimed sum of the folded polynomial
    pub v: E::ScalarField,
    /// PCS commitments to selector polynomials (shared across instances)
    pub selector_commits: Option<Vec<Commitment<E>>>,
    /// PCS commitments to witness polynomials (M instances × num_witnesses)
    pub witness_commits: Option<Vec<Commitment<E>>>,
    /// PCS batch opening proof
    pub batch_openings: Option<BatchProof<E, PCS>>,
}

impl<E: Pairing, PCS: PolynomialCommitmentScheme<E>> Proof<E, PCS> {
    /// Compute the serialized proof size in bytes.
    ///
    /// Includes the IOP proof, SumFold metadata, commitments, and opening
    /// proofs. Uses compressed serialization for compactness.
    pub fn proof_size_bytes(&self) -> usize {
        let mut size = 0usize;
        // IOP proof (point + round messages)
        size += self.proof.serialized_size(Compress::Yes);
        // SumFold metadata
        size += self.sum_t.serialized_size(Compress::Yes);
        size += self.q_aux_info.serialized_size(Compress::Yes);
        size += self.v.serialized_size(Compress::Yes);
        // PCS commitments
        if let Some(ref commits) = self.selector_commits {
            size += commits.serialized_size(Compress::Yes);
        }
        if let Some(ref commits) = self.witness_commits {
            size += commits.serialized_size(Compress::Yes);
        }
        // PCS batch opening proof (serialize individual fields)
        if let Some(ref bp) = self.batch_openings {
            size += bp.sum_check_proof.serialized_size(Compress::Yes);
            size += bp.f_i_eval_at_point_i.serialized_size(Compress::Yes);
            size += bp.g_prime_proof.serialized_size(Compress::Yes);
        }
        size
    }
}

/// Proof produced by the SumFold protocol.
///
/// Packages the SumCheck proof over the folding variables together
/// with the derived values needed for verification and merging.
#[derive(Clone, Debug)]
pub struct SumFoldProof<F: PrimeField> {
    /// SumCheck proof over the folding variables (point + prover messages)
    pub proof: IOPProof<F>,
    /// Weighted sum: sum_t = Σᵢ eq(ρ, i) · sᵢ
    pub sum_t: F,
    /// Aux info for the folding SumCheck (max_degree, num_variables = log₂(m))
    pub q_aux_info: VPAuxInfo<F>,
    /// Claimed sum of the folded polynomial
    pub v: F,
}

impl<F: PrimeField> SumFoldProof<F> {
    pub fn new(proof: IOPProof<F>, sum_t: F, q_aux_info: VPAuxInfo<F>, v: F) -> Self {
        Self {
            proof,
            sum_t,
            q_aux_info,
            v,
        }
    }
}

/// Timing breakdown from `dist_prove`, for benchmark reporting.
///
/// All durations are in milliseconds (f64 for sub-ms precision).
#[derive(Clone, Debug, Default)]
pub struct BenchmarkTimings {
    /// SRS generation / loading + circuit preprocessing
    pub setup_ms: f64,
    /// Distributed proving: d_commit + SumFold + SumCheck + commitment folding + d_multi_open + assembly
    pub prover_ms: f64,
    /// Proof verification: SumCheck replay + gate check + PCS batch_verify
    pub verifier_ms: f64,
}

/// Network configuration for distributed proving.
#[derive(Clone, Debug, Deserialize)]
pub struct NetworkConfig {
    /// Path to hosts file (one HOST:PORT per line)
    pub hosts_file: String,
    /// This party's ID (0-indexed; overridden by CLI `--party` flag)
    #[serde(default)]
    pub party_id: usize,
}

impl NetworkConfig {
    /// Create a new NetworkConfig.
    pub fn new(hosts_file: impl Into<String>, party_id: usize) -> Self {
        Self {
            hosts_file: hosts_file.into(),
            party_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = Config::new(4, 10, GateType::Vanilla, 2);
        assert_eq!(config.num_instances(), 16);
        assert_eq!(config.log_num_instances, 4);
        assert_eq!(config.num_constraints(), 1024);
        assert_eq!(config.log_num_constraints, 10);
        assert_eq!(config.gate_type, GateType::Vanilla);
        assert_eq!(config.num_witness_columns(), 3);
        assert_eq!(config.num_selector_columns(), 5);
        assert_eq!(config.num_parties(), 4);
        assert_eq!(config.log_num_parties, 2);
    }

    #[test]
    fn test_gate_type_to_gate() {
        let gate = GateType::Vanilla.to_gate();
        assert_eq!(gate.num_witness_columns(), 3);
        assert_eq!(gate.num_selector_columns(), 5);
    }

    #[test]
    fn test_build_mock_circuit() {
        use ark_bn254::Fr;
        let config = Config::new(0, 8, GateType::Vanilla, 0);
        let circuit = config.build_mock_circuit::<Fr>();

        assert_eq!(circuit.index.params.num_constraints, 256);
        assert!(circuit.is_satisfied());
    }

    #[test]
    fn test_build_partitioned_circuits() {
        use ark_bn254::Fr;
        // ν=2 → M=4 instances, μ=10 → N=1024 constraints per instance, κ=2 → K=4 parties
        let config = Config::new(2, 10, GateType::Vanilla, 2);
        let circuits = config.build_partitioned_circuits::<Fr>();

        // M = 4 instances
        assert_eq!(circuits.len(), 4);
        // Each instance on this sub-prover: N/K = 1024/4 = 256 constraints
        for circuit in &circuits {
            assert_eq!(circuit.index.params.num_constraints, 256);
            assert!(circuit.is_satisfied());
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_build_partitioned_circuits_par() {
        use ark_bn254::Fr;
        let config = Config::new(2, 10, GateType::Vanilla, 2);
        let circuits = config.build_partitioned_circuits_par::<Fr>();

        assert_eq!(circuits.len(), 4);
        for circuit in &circuits {
            assert_eq!(circuit.index.params.num_constraints, 256);
            assert!(circuit.is_satisfied());
        }
    }

    #[test]
    fn bench_build_circuits_sequential_vs_parallel() {
        use ark_bn254::Fr;
        use ark_std::time::Instant;

        // M=8, N=2^16=65536, K=4 → each instance has 16384 constraints
        let config = Config::new(3, 16, GateType::Vanilla, 2);

        // Sequential
        let start = Instant::now();
        let circuits_seq = config.build_partitioned_circuits::<Fr>();
        let seq_duration = start.elapsed();
        println!("Sequential: M={} circuits, N/K={} constraints each, took {:?}",
            circuits_seq.len(), circuits_seq[0].index.params.num_constraints, seq_duration);

        // Parallel
        #[cfg(feature = "parallel")]
        {
            let start = Instant::now();
            let circuits_par = config.build_partitioned_circuits_par::<Fr>();
            let par_duration = start.elapsed();
            println!("Parallel:   M={} circuits, N/K={} constraints each, took {:?}",
                circuits_par.len(), circuits_par[0].index.params.num_constraints, par_duration);
            println!("Speedup: {:.2}x", seq_duration.as_secs_f64() / par_duration.as_secs_f64());
        }

        for circuit in &circuits_seq {
            assert!(circuit.is_satisfied());
        }
    }
}
