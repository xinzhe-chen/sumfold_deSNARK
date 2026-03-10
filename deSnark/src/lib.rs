//! sumfold_deSNARK public entry points.
//!
//! This crate exposes the research prototype implementation of the distributed
//! prover, along with the configuration structures used by the benchmark and
//! demo binaries. The most relevant entry points are:
//!
//! - [`Config`] and [`NetworkConfig`] for loading benchmark/demo configuration.
//! - [`dist_prove`] for running the distributed proving pipeline.
//! - [`verify`] for standalone verification from a proof and verifying key.

#![allow(clippy::default_constructed_unit_structs)] // protocol structs use PhantomData-heavy archived code paths
#![allow(clippy::needless_range_loop)] // indexed loops make transcript and folding layouts explicit
#![allow(clippy::type_complexity)] // proof-key tuples are part of the artifact-facing API

pub mod d_sumfold;
pub mod errors;
pub mod snark;
pub mod structs;

pub use errors::DeSnarkError;
#[cfg(any(test, feature = "test-srs"))]
pub use snark::setup;
pub use snark::{
    circuits_to_sumcheck, dist_prove, dist_prove_sumcheck, make_circuit, prove_hyper_pianist,
    prove_sumfold, setup_for_testing, setup_from_srs, verify, HyperPlonkPCS,
};
pub use structs::{
    BenchmarkTimings, Config, GateType, MockCircuit, NetworkConfig, Proof, SumCheckInstance,
};
