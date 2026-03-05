//! Distributed Benchmark Binary for sumfold_deSNARK
//!
//! Sweeps over a range of `nv` (log_num_constraints) values, calling
//! `dist_prove` for each and collecting performance metrics.
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
//! The TOML provides base parameters (log_num_instances, gate_type,
//! log_num_parties, srs_path, network). The `log_num_constraints` field is
//! overridden per iteration from the CLI range.

use ark_bn254::Bn254;
use deNetwork::{DeMultiNet as Net, DeNet};
use deSnark::{snark::dist_prove, setup, structs::Config};
use subroutines::pcs::prelude::MultilinearKzgPCS;
use std::{
    env,
    io::Write,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
use tracing::error;
use tracing_subscriber::{fmt, EnvFilter};

// ─── Peak RSS sampler (background thread, 5ms poll) ──────────────────────────

fn current_rss_mb() -> f64 {
    let mut sys = System::new();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        ProcessRefreshKind::new().with_memory(),
    );
    sys.process(pid)
        .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}

struct PeakSampler {
    stop_flag: Arc<AtomicBool>,
    peak_mb: Arc<Mutex<f64>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl PeakSampler {
    fn start(initial_rss: f64) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let peak_mb = Arc::new(Mutex::new(initial_rss));
        let flag = stop_flag.clone();
        let peak = peak_mb.clone();
        let pid = Pid::from_u32(std::process::id());
        let handle = thread::spawn(move || {
            let mut sys = System::new();
            while !flag.load(Ordering::Relaxed) {
                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]),
                    ProcessRefreshKind::new().with_memory(),
                );
                if let Some(p) = sys.process(pid) {
                    let mb = p.memory() as f64 / (1024.0 * 1024.0);
                    let mut g = peak.lock().unwrap();
                    if mb > *g {
                        *g = mb;
                    }
                }
                thread::sleep(Duration::from_millis(5));
            }
        });
        Self {
            stop_flag,
            peak_mb,
            handle: Some(handle),
        }
    }
    fn stop(mut self) -> f64 {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }
        (*self.peak_mb.lock().unwrap()).max(current_rss_mb())
    }
}

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
        eprintln!(
            "# Rayon threads: {} (RAYON_NUM_THREADS={})",
            rayon::current_num_threads(),
            std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string()),
        );
        // CSV header
        println!("nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb,d_commit_ms,sumfold_ms,sumcheck_ms,fold_ms,multi_open_ms");
        std::io::stdout().flush().ok();
    }

    // Pre-warm SRS cache at nv_max so all iterations can reuse it
    if cli.nv_min < cli.nv_max {
        let mut max_config = base_config.clone();
        max_config.log_num_constraints = cli.nv_max;
        if Net::am_master() {
            eprintln!("# Pre-generating SRS for nv_max={}...", cli.nv_max);
        }
        setup::<Bn254, MultilinearKzgPCS<Bn254>>(&max_config)
            .expect("SRS pre-generation failed");
        if Net::am_master() {
            eprintln!("# SRS pre-generation complete.");
        }
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

        // ─── Warmup run (not timed) ─────────────────────────────────────
        if Net::am_master() {
            eprintln!("#   warmup run...");
        }
        match dist_prove::<Bn254>(&config) {
            Ok(_) => {
                if Net::am_master() {
                    eprintln!("#   warmup complete");
                }
            },
            Err(e) => {
                error!(
                    "❌ [Party {}] warmup dist_prove failed at nv={}: {}",
                    cli.party_id, nv, e
                );
                continue; // skip this nv
            },
        }

        // ─── Timed repetitions ───────────────────────────────────────────
        let mut total_setup_ms = 0.0_f64;
        let mut total_prover_ms = 0.0_f64;
        let mut total_verifier_ms = 0.0_f64;
        let mut total_proof_bytes = 0usize;
        let mut total_comm_sent = 0usize;
        let mut total_comm_recv = 0usize;
        let mut total_cpu_ms = 0.0_f64;
        let mut total_wall_ms = 0.0_f64;
        let mut max_peak_rss_mb = 0.0_f64;
        let mut total_d_commit_ms = 0.0_f64;
        let mut total_sumfold_ms = 0.0_f64;
        let mut total_sumcheck_ms = 0.0_f64;
        let mut total_fold_ms = 0.0_f64;
        let mut total_multi_open_ms = 0.0_f64;
        let mut successful_reps = 0usize;

        for rep in 0..cli.repetitions {
            // Peak RSS sampled throughout dist_prove;
            // CPU is measured internally in snark.rs, scoped to proving phase only
            let sampler = PeakSampler::start(current_rss_mb());

            match dist_prove::<Bn254>(&config) {
                Ok((_vk, proof, timings)) => {
                    let peak_rss_mb = sampler.stop();

                    total_setup_ms += timings.setup_ms;
                    total_prover_ms += timings.prover_ms;
                    total_verifier_ms += timings.verifier_ms;
                    total_comm_sent += timings.comm_sent;
                    total_comm_recv += timings.comm_recv;
                    total_cpu_ms += timings.prove_cpu_ms;
                    total_wall_ms += timings.prove_wall_ms;
                    total_d_commit_ms += timings.d_commit_ms;
                    total_sumfold_ms += timings.sumfold_ms;
                    total_sumcheck_ms += timings.sumcheck_ms;
                    total_fold_ms += timings.fold_ms;
                    total_multi_open_ms += timings.multi_open_ms;
                    if peak_rss_mb > max_peak_rss_mb {
                        max_peak_rss_mb = peak_rss_mb;
                    }

                    if let Some(ref p) = proof {
                        total_proof_bytes += p.proof_size_bytes();
                    }

                    successful_reps += 1;

                    if Net::am_master() {
                        let avg_cpu = if timings.prove_wall_ms > 0.0 {
                            timings.prove_cpu_ms / timings.prove_wall_ms * 100.0
                        } else {
                            0.0
                        };
                        eprintln!(
                            "#   rep {}/{}: prove={:.1}ms, verify={:.1}ms, proof={}B, sent={}B, recv={}B, cpu={:.0}%, rss={:.1}MB",
                            rep + 1,
                            cli.repetitions,
                            timings.prover_ms,
                            timings.verifier_ms,
                            proof.as_ref().map_or(0, |p| p.proof_size_bytes()),
                            timings.comm_sent,
                            timings.comm_recv,
                            avg_cpu,
                            peak_rss_mb,
                        );
                    }
                },
                Err(e) => {
                    let _ = sampler.stop(); // stop background thread even on error
                    error!(
                        "❌ [Party {}] dist_prove failed at nv={}, rep={}: {}",
                        cli.party_id,
                        nv,
                        rep + 1,
                        e
                    );
                    // Skip remaining reps for this nv
                    break;
                },
            }
        }

        // Output averaged CSV line (master only); skip if all reps failed
        // prover_ms, verifier_ms, comm_sent/recv are all per-instance averages (÷ M ÷
        // reps) so they are directly comparable to HyperPianist (which always
        // proves 1 instance).
        if Net::am_master() && successful_reps > 0 {
            let r = successful_reps as f64;
            let m = config.num_instances() as f64;
            let m_usize = config.num_instances();
            let avg_cpu_pct = if total_wall_ms > 0.0 {
                total_cpu_ms / total_wall_ms * 100.0
            } else {
                0.0
            };
            println!(
                "{},{},{},{:.3},{:.3},{:.3},{},{},{},{:.1},{:.1},{:.3},{:.3},{:.3},{:.3},{:.3}",
                nv,
                config.num_instances(),
                config.num_parties(),
                total_setup_ms / r,
                total_prover_ms / r / m,
                total_verifier_ms / r / m,
                total_proof_bytes / successful_reps,
                total_comm_sent / successful_reps / m_usize,
                total_comm_recv / successful_reps / m_usize,
                avg_cpu_pct,
                max_peak_rss_mb,
                total_d_commit_ms / r / m,
                total_sumfold_ms / r / m,
                total_sumcheck_ms / r / m,
                total_fold_ms / r / m,
                total_multi_open_ms / r / m,
            );
            std::io::stdout().flush().ok();
        } else if Net::am_master() {
            eprintln!("# nv={}: all repetitions failed, skipping CSV line", nv);
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
            },
            "--nv-min" => {
                i += 1;
                nv_min = Some(
                    args.get(i)
                        .expect("--nv-min requires a value")
                        .parse::<usize>()
                        .expect("Invalid nv-min"),
                );
            },
            "--nv-max" => {
                i += 1;
                nv_max = Some(
                    args.get(i)
                        .expect("--nv-max requires a value")
                        .parse::<usize>()
                        .expect("Invalid nv-max"),
                );
            },
            "--reps" => {
                i += 1;
                repetitions = args
                    .get(i)
                    .expect("--reps requires a value")
                    .parse::<usize>()
                    .expect("Invalid repetitions");
            },
            arg if !arg.starts_with('-') => {
                config_path = Some(arg.to_string());
            },
            other => {
                eprintln!("Unknown option: {other}");
                print_usage(&args[0]);
                std::process::exit(1);
            },
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
        },
        _ => {
            print_usage(&args[0]);
            std::process::exit(1);
        },
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
