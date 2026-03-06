//! deSnark - Distributed SNARK protocol implementation

pub mod d_sumfold;
pub mod errors;
pub mod snark;
pub mod structs;

pub use errors::DeSnarkError;
pub use snark::{
    circuits_to_sumcheck, dist_prove, dist_prove_sumcheck, make_circuit, prove_hyper_pianist,
    prove_sumfold, setup_for_testing, setup_from_srs, verify, HyperPlonkPCS,
};
#[cfg(any(test, feature = "test-srs"))]
pub use snark::setup;
pub use structs::{
    BenchmarkTimings, Config, GateType, MockCircuit, NetworkConfig, Proof, SumCheckInstance,
};
