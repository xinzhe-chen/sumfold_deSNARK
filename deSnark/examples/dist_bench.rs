//! Distributed Benchmark Binary for sumfold_deSNARK
//!
//! Sweeps over a range of `nv` (log_num_constraints) values, calling `dist_prove`
//! for each and collecting performance metrics.
//!
//! ## Metrics collected (master only)
//!
//! | Column | Description |
//! |--------|-------------|
//! | `nv`   | log₂(N), total constraints per instance |
//! | `M`    | number of instances (2^log_num_instances) |
//! | `K`    | number of parties (2^log_num_parties) |
//! | `setup_ms` | SRS generation + circuit preprocessing |
//! | `prover_ms` | d_commit + SumFold + SumCheck + folding + d_multi_open + assembly |
//! | `verifier_ms` | SumCheck replay + gate check + PCS batch_verify |
//! | `proof_bytes` | compressed proof size |
//! | `comm_sent` | bytes sent (network) |
//! | `comm_recv` | bytes received (network) |
//!
//! ## Usage
//!
//! ```bash
//! dist_bench --party <ID> --nv-min 10 --nv-max 14 bench_config.toml
//! ```
//!
//! The TOML provides base parameters (log_num_instances, gate_type, log_num_parties,
//! srs_path, network). The `log_num_constraints` field is overridden per iteration
//! from the CLI range.

use ark_bn254::Bn254;
use deNetwork::{DeMultiNet as Net, DeNet};
use deSnark::snark::dist_prove;
use deSnark::structs::Config;
use std::env;
use std::time::Instant;
use tracing::error;
use tracing_subscriber::{fmt, EnvFilter};

fn main() {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .init();

    let args: Vec<String> = env::args().collect();
    let cli = parse_args(&args);

    let (base_config, mut net_config) =
        Config::from_toml_file(&cli.config_path).unwrap_or_else(|e| {
            eprintln!("Error loading {}: {e}", cli.config_path);
            std::process::exit(1);
        });
    net_config.party_id = cli.party_id;

    // Initialize network once — reused for all nv iterations
    Net::init_from_file(&net_config.hosts_file, net_config.party_id);

    if Net::am_master() {
        eprintln!(
            "# sumfold_deSNARK benchmark: M={}, K={}, nv={}..{}, reps={}",
            base_config.num_instances(),
            base_config.num_parties(),
            cli.nv_min,
            cli.nv_max,
            cli.repetitions,
        );
        // CSV header
        println!(
            "nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv"
        );
    }

    for nv in cli.nv_min..=cli.nv_max {
        // Create config for this nv
        let mut config = base_config.clone();
        config.log_num_constraints = nv;

        if Net::am_master() {
            eprintln!(
                "# nv={}: N={}, N/K={} per party, M={} instances",
                nv,
                config.num_constraints(),
                config.num_constraints() / config.num_parties(),
                config.num_instances(),
            );
        }

        // Accumulate timings across repetitions
        let mut total_setup_ms = 0.0_f64;
        let mut total_prover_ms = 0.0_f64;
        let mut total_verifier_ms = 0.0_f64;
        let mut total_proof_bytes = 0usize;
        let mut total_comm_sent = 0usize;
        let mut total_comm_recv = 0usize;

        for rep in 0..cli.repetitions {
            Net::reset_stats();
            let wall_start = Instant::now();

            match dist_prove::<Bn254>(&config) {
                Ok((_vk, proof, timings)) => {
                    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
                    let stats = Net::stats();

                    total_setup_ms += timings.setup_ms;
                    total_prover_ms += timings.prover_ms;
                    total_verifier_ms += timings.verifier_ms;
                    total_comm_sent += stats.bytes_sent;
                    total_comm_recv += stats.bytes_recv;

                    if let Some(ref p) = proof {
                        total_proof_bytes += p.proof_size_bytes();
                    }

                    if Net::am_master() {
                        eprintln!(
                            "#   rep {}/{}: wall={:.1}ms, prove={:.1}ms, verify={:.1}ms, proof={}B, sent={}B, recv={}B",
                            rep + 1,
                            cli.repetitions,
                            wall_ms,
                            timings.prover_ms,
                            timings.verifier_ms,
                            proof.as_ref().map_or(0, |p| p.proof_size_bytes()),
                            stats.bytes_sent,
                            stats.bytes_recv,
                        );
                    }
                }
                Err(e) => {
                    error!(
                        "❌ [Party {}] dist_prove failed at nv={}, rep={}: {}",
                        cli.party_id,
                        nv,
                        rep + 1,
                        e
                    );
                    // Skip remaining reps for this nv
                    break;
                }
            }
        }

        // Output averaged CSV line (master only)
        if Net::am_master() {
            let r = cli.repetitions as f64;
            println!(
                "{},{},{},{:.3},{:.3},{:.3},{},{},{}",
                nv,
                config.num_instances(),
                config.num_parties(),
                total_setup_ms / r,
                total_prover_ms / r,
                total_verifier_ms / r,
                total_proof_bytes / cli.repetitions,
                total_comm_sent / cli.repetitions,
                total_comm_recv / cli.repetitions,
            );
        }
    }

    Net::deinit();
    if Net::am_master() {
        eprintln!("# Benchmark complete.");
    }
}

// ─── CLI parsing ─────────────────────────────────────────────────

struct CliArgs {
    party_id: usize,
    nv_min: usize,
    nv_max: usize,
    repetitions: usize,
    config_path: String,
}

fn parse_args(args: &[String]) -> CliArgs {
    let mut party_id = None;
    let mut nv_min = None;
    let mut nv_max = None;
    let mut repetitions = 1usize;
    let mut config_path = None;
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--party" => {
                i += 1;
                party_id = Some(
                    args.get(i)
                        .expect("--party requires a value")
                        .parse::<usize>()
                        .expect("Invalid party ID"),
                );
            }
            "--nv-min" => {
                i += 1;
                nv_min = Some(
                    args.get(i)
                        .expect("--nv-min requires a value")
                        .parse::<usize>()
                        .expect("Invalid nv-min"),
                );
            }
            "--nv-max" => {
                i += 1;
                nv_max = Some(
                    args.get(i)
                        .expect("--nv-max requires a value")
                        .parse::<usize>()
                        .expect("Invalid nv-max"),
                );
            }
            "--reps" => {
                i += 1;
                repetitions = args
                    .get(i)
                    .expect("--reps requires a value")
                    .parse::<usize>()
                    .expect("Invalid repetitions");
            }
            arg if !arg.starts_with('-') => {
                config_path = Some(arg.to_string());
            }
            other => {
                eprintln!("Unknown option: {other}");
                print_usage(&args[0]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    match (party_id, nv_min, nv_max, config_path) {
        (Some(p), Some(lo), Some(hi), Some(c)) => {
            if lo > hi {
                eprintln!("Error: --nv-min ({lo}) must be <= --nv-max ({hi})");
                std::process::exit(1);
            }
            CliArgs {
                party_id: p,
                nv_min: lo,
                nv_max: hi,
                repetitions,
                config_path: c,
            }
        }
        _ => {
            print_usage(&args[0]);
            std::process::exit(1);
        }
    }
}

fn print_usage(prog: &str) {
    eprintln!(
        "Usage: {} --party <ID> --nv-min <N> --nv-max <N> [--reps <R>] <config.toml>",
        prog
    );
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --party <ID>     Party ID (0 = master)");
    eprintln!("  --nv-min <N>     Minimum log_num_constraints");
    eprintln!("  --nv-max <N>     Maximum log_num_constraints");
    eprintln!("  --reps <R>       Repetitions per nv (default: 1)");
}
