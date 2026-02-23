// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the HyperPlonk SNARK.

use std::time::Duration;

use crate::mock::MockCircuit;
use arithmetic::VirtualPolynomial;
use ark_ec::pairing::Pairing;
use errors::HyperPlonkErrors;
use subroutines::{
    pcs::prelude::PolynomialCommitmentScheme, poly_iop::prelude::PermutationCheck, BatchProof,
    IOPProof,
};
use transcript::IOPTranscript;
mod custom_gate;
mod errors;
mod mock;
pub mod prelude;
mod selectors;
mod snark;
pub mod structs;
pub mod utils;
mod witness;

/// A trait for HyperPlonk SNARKs.
/// A HyperPlonk is derived from ZeroChecks and PermutationChecks.
pub trait HyperPlonkSNARK<E, PCS>: PermutationCheck<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Index;
    type ProvingKey;
    type VerifyingKey;
    type Proof;

    /// Generate the preprocessed polynomials output by the indexer.
    ///
    /// Inputs:
    /// - `index`: HyperPlonk index
    /// - `pcs_srs`: Polynomial commitment structured reference string
    ///
    /// Outputs:
    /// - The HyperPlonk proving key, which includes the preprocessed
    ///   polynomials.
    /// - The HyperPlonk verifying key, which includes the preprocessed
    ///   polynomial commitments
    fn preprocess(
        index: &Self::Index,
        pcs_srs: &PCS::SRS,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey, Duration), HyperPlonkErrors>;

    /// Generate HyperPlonk SNARK proof.
    ///
    /// Inputs:
    /// - `pk`: circuit proving key
    /// - `pub_input`: online public input
    /// - `witness`: witness assignment
    ///
    /// Outputs:
    /// - The HyperPlonk SNARK proof.
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
    >;

    fn mul_prove(
        pks: &Self::ProvingKey,
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
    >;

    /// Verify the HyperPlonk proof.
    ///
    /// Inputs:
    /// - `vk`: verifying key
    /// - `pub_input`: online public input
    /// - `proof`: HyperPlonk SNARK proof challenges
    ///
    /// Outputs:
    /// - Return a boolean on whether the verification is successful
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
    ) -> Result<bool, HyperPlonkErrors>;
}
