//! Distributed Prove Demo (4 nodes)
//!
//! Demonstrates the `dist_prove` pipeline across 4 parties.
//! Currently runs through: network sync → SRS setup → circuit generation + preprocess.
//! SumFold and HyperPianist phases return errors (not yet implemented).
//!
//! Usage:
//! ```bash
//! dist_prove_demo --party <ID> <config.toml>
//! ```
//!
//! Run via script:
//! ```bash
//! ./scripts/run_desnark_demo.sh
//! ```

use ark_bls12_381::Bls12_381;
use deNetwork::{DeMultiNet as Net, DeNet};
use deSnark::snark::dist_prove;
use deSnark::structs::Config;
use std::env;
use std::time::Instant;
use tracing::{error, info};
use tracing_subscriber::{fmt, EnvFilter};

fn main() {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .init();

    let args: Vec<String> = env::args().collect();

    // Parse: dist_prove_demo --party <ID> <config.toml>
    let (party_id, config_path) = parse_args(&args);

    let (config, mut net_config) =
        Config::from_toml_file(&config_path).unwrap_or_else(|e| {
            eprintln!("Error loading {config_path}: {e}");
            std::process::exit(1);
        });
    net_config.party_id = party_id;

    info!(
        "[Party {}] Config: M={} instances, N={} constraints, K={} parties, N/K={} per party",
        party_id,
        config.num_instances(),
        config.num_constraints(),
        config.num_parties(),
        config.num_constraints() / config.num_parties()
    );

    // Initialize network before entering dist_prove
    Net::init_from_file(&net_config.hosts_file, net_config.party_id);
    info!(
        "[Party {}] Network initialized ({} parties)",
        party_id,
        Net::n_parties()
    );

    info!("[Party {}] Starting dist_prove pipeline...", party_id);
    let start = Instant::now();

    match dist_prove::<Bls12_381>(&config) {
        Ok((_vk, proof, _timings)) => {
            let role = if proof.is_some() { "master (has proof)" } else { "worker" };
            info!(
                "✅ [Party {}] dist_prove completed as {} in {:?}",
                party_id,
                role,
                start.elapsed()
            );
        }
        Err(e) => {
            error!(
                "❌[Party {}] dist_prove failed: {} (elapsed {:?})",
                party_id,
                e,
                start.elapsed()
            );
        }
    }

    // Deinitialize network after dist_prove
    Net::deinit();
    info!("✅ [Party {}] Exited successfully", party_id);
}

fn parse_args(args: &[String]) -> (usize, String) {
    let mut party_id = None;
    let mut config_path = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--party" => {
                i += 1;
                party_id = Some(args.get(i).expect("--party requires a value").parse::<usize>().expect("Invalid party ID"));
            }
            arg if !arg.starts_with('-') => {
                config_path = Some(arg.to_string());
            }
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    match (party_id, config_path) {
        (Some(p), Some(c)) => (p, c),
        _ => {
            eprintln!("Usage: {} --party <ID> <config.toml>", args[0]);
            std::process::exit(1);
        }
    }
}
