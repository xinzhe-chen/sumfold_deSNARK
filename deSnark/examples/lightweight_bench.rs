//! Lightweight single-machine benchmark — detailed per-phase resource report.
//!
//! Per phase measures:
//!   wall(ms)     — actual elapsed time
//!   avg_cpu%     — (user+sys CPU time) / wall × 100  via getrusage
//!                  >100% indicates multi-core parallelism
//!   peak_cpu%    — max instantaneous CPU% sampled every 20ms via sysinfo
//!   peak_rss(MB) — max RSS sampled every 20ms during the phase
//!   %RAM         — peak_rss as fraction of total system RAM
//!
//! Run:
//!   cargo run --release --example lightweight_bench -p deSnark

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use ark_bn254::{Bn254, Fr};
use deSnark::{
    snark::{circuits_to_sumcheck, make_circuit, prove_sumfold, setup},
    structs::{Config, GateType},
};
use subroutines::{
    pcs::prelude::MultilinearKzgPCS,
    poly_iop::prelude::{PolyIOP, SumCheck},
};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

type PCS = MultilinearKzgPCS<Bn254>;

// ─── CPU time via getrusage
// ───────────────────────────────────────────────────

/// Returns (user_ms, sys_ms) — cumulative CPU time for this process since
/// start.
fn get_cpu_ms() -> (f64, f64) {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let user = usage.ru_utime.tv_sec as f64 * 1e3 + usage.ru_utime.tv_usec as f64 * 1e-3;
    let sys = usage.ru_stime.tv_sec as f64 * 1e3 + usage.ru_stime.tv_usec as f64 * 1e-3;
    (user, sys)
}

// ─── System info helpers
// ──────────────────────────────────────────────────────

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

fn total_ram_mb() -> f64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.total_memory() as f64 / (1024.0 * 1024.0)
}

// ─── Background sampler: peak RSS + peak CPU% ────────────────────────────────

struct PeakSample {
    peak_rss_mb: f64,
    peak_cpu_pct: f32,
}

struct PeakSampler {
    stop_flag: Arc<AtomicBool>,
    result: Arc<Mutex<PeakSample>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl PeakSampler {
    fn start(initial_rss: f64) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let result = Arc::new(Mutex::new(PeakSample {
            peak_rss_mb: initial_rss,
            peak_cpu_pct: 0.0,
        }));

        let flag = stop_flag.clone();
        let res = result.clone();
        let pid = Pid::from_u32(std::process::id());

        let handle = thread::spawn(move || {
            let mut sys = System::new();
            // First refresh to establish CPU baseline (cpu_usage() needs ≥2 samples)
            sys.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[pid]),
                ProcessRefreshKind::new().with_memory().with_cpu(),
            );
            thread::sleep(Duration::from_millis(5));

            while !flag.load(Ordering::Relaxed) {
                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]),
                    ProcessRefreshKind::new().with_memory().with_cpu(),
                );
                if let Some(p) = sys.process(pid) {
                    let rss_mb = p.memory() as f64 / (1024.0 * 1024.0);
                    let cpu_pct = p.cpu_usage(); // % of 1 core; 1000% = all 10 cores
                    let mut guard = res.lock().unwrap();
                    if rss_mb > guard.peak_rss_mb {
                        guard.peak_rss_mb = rss_mb;
                    }
                    if cpu_pct > guard.peak_cpu_pct {
                        guard.peak_cpu_pct = cpu_pct;
                    }
                }
                thread::sleep(Duration::from_millis(5));
            }
        });

        Self {
            stop_flag,
            result,
            handle: Some(handle),
        }
    }

    fn stop(mut self) -> PeakSample {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }
        let g = self.result.lock().unwrap();
        PeakSample {
            peak_rss_mb: g.peak_rss_mb,
            peak_cpu_pct: g.peak_cpu_pct,
        }
    }
}

// ─── Per-phase stats
// ──────────────────────────────────────────────────────────

struct PhaseStats {
    wall_ms: f64,
    /// Average CPU utilization: (user+sys) / wall × 100.  >100% = multi-core.
    avg_cpu_pct: f64,
    /// Peak instantaneous CPU% sampled every ~20ms.  NaN if phase too short.
    peak_cpu_pct: f64,
    /// Peak RSS (MB) sampled every ~20ms during the phase.
    peak_rss_mb: f64,
}

fn timed<F, T>(f: F) -> (T, PhaseStats)
where
    F: FnOnce() -> T,
{
    let rss_before = current_rss_mb();
    let (cu_before, cs_before) = get_cpu_ms();
    let sampler = PeakSampler::start(rss_before);
    let wall = Instant::now();

    let result = f();

    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let (cu_after, cs_after) = get_cpu_ms();
    let sample = sampler.stop();
    let peak_rss_mb = sample.peak_rss_mb.max(current_rss_mb());

    let avg_cpu_pct = if wall_ms >= 1.0 {
        ((cu_after - cu_before) + (cs_after - cs_before)) / wall_ms * 100.0
    } else {
        f64::NAN
    };
    let peak_cpu_pct = if sample.peak_cpu_pct > 0.5 {
        sample.peak_cpu_pct as f64
    } else {
        f64::NAN
    };

    (
        result,
        PhaseStats {
            wall_ms,
            avg_cpu_pct,
            peak_cpu_pct,
            peak_rss_mb,
        },
    )
}

// ─── Per-config result
// ────────────────────────────────────────────────────────

struct ConfigResult {
    log_m: usize,
    log_nv: usize,
    log_k: usize,
    nv_eff: usize,
    m: usize,
    k: usize,
    phases: [(&'static str, PhaseStats); 4],
    total_ms: f64,
}

fn bench_config(log_m: usize, log_nv: usize, log_k: usize) -> ConfigResult {
    let config = Config::new(log_m, log_nv, GateType::Vanilla, log_k);
    let nv_eff = log_nv - log_k;
    let m = config.num_instances();
    let k = config.num_parties();
    let wall_start = Instant::now();

    let (srs, s0) = timed(|| setup::<Bn254, PCS>(&config).expect("setup"));
    let ((pk, _vk, circs), s1) =
        timed(|| make_circuit::<Bn254, PCS>(&config, &srs).expect("make_circuit"));
    let (instances, s2) = timed(|| circuits_to_sumcheck::<Bn254, PCS>(&pk, &circs).expect("c2sc"));
    let mut tr = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
    let (_, s3) = timed(|| prove_sumfold(instances, &mut tr).expect("prove_sumfold"));

    ConfigResult {
        log_m,
        log_nv,
        log_k,
        nv_eff,
        m,
        k,
        total_ms: wall_start.elapsed().as_secs_f64() * 1000.0,
        phases: [
            ("setup", s0),
            ("make_circuit", s1),
            ("c2sc", s2),
            ("prove_sumfold", s3),
        ],
    }
}

// ─── Formatting helpers
// ───────────────────────────────────────────────────────

fn fmt_pct(v: f64) -> String {
    if v.is_nan() {
        "  —  ".to_string()
    } else {
        format!("{:>6.0}%", v)
    }
}

fn fmt_ms(v: f64) -> String {
    if v < 1.0 {
        format!("{:>8.3}", v)
    } else {
        format!("{:>8.1}", v)
    }
}

// ─── Print results
// ────────────────────────────────────────────────────────────

fn print_results(results: &[ConfigResult], total_ram: f64) {
    let w = 72usize;
    let bar = "═".repeat(w);
    let sep = "─".repeat(w);
    let thin = "┄".repeat(w);

    for r in results {
        println!("╔{}╗", bar);
        println!(
            "║  Config ({},{},{})   M={:<3} K={:<3} nv_eff={:<3}   total = {:.1} ms{:>width$}║",
            r.log_m,
            r.log_nv,
            r.log_k,
            r.m,
            r.k,
            r.nv_eff,
            r.total_ms,
            "",
            width = w.saturating_sub(57 + format!("{:.1}", r.total_ms).len()),
        );
        println!("╠{}╣", bar);
        println!(
            "║  {:<15} {:>10}  {:>9}  {:>10}  {:>12}  {:>6}  ║",
            "Phase", "wall(ms)", "avg_cpu%", "peak_cpu%", "peak_rss(MB)", "%RAM"
        );
        println!("╠{}╣", thin);

        for (name, s) in &r.phases {
            println!(
                "║  {:<15} {}  {:>9}  {:>10}  {:>12.1}  {:>6}  ║",
                name,
                fmt_ms(s.wall_ms),
                fmt_pct(s.avg_cpu_pct),
                fmt_pct(s.peak_cpu_pct),
                s.peak_rss_mb,
                format!("{:.2}%", s.peak_rss_mb / total_ram * 100.0),
            );
        }
        println!("╚{}╝", bar);
        println!();
    }

    // ── Summary table ──────────────────────────────────────────────────────
    println!("╔{}╗", bar);
    println!(
        "║{:^width$}║",
        "  Summary — prove_sumfold  (core proving work)",
        width = w
    );
    println!("╠{}╣", bar);
    println!(
        "║  {:<18}  {:>4}  {:>4}  {:>6}  {:>10}  {:>9}  {:>10}  {:>10}  ║",
        "config", "M", "K", "nv_eff", "wall(ms)", "avg_cpu%", "peak_cpu%", "pk_rss(MB)"
    );
    println!("╠{}╣", sep);
    for r in results {
        let (_, s) = &r.phases[3]; // prove_sumfold
        println!(
            "║  {:<18}  {:>4}  {:>4}  {:>6}  {:>10.1}  {:>9}  {:>10}  {:>10.1}  ║",
            format!("({},{},{})", r.log_m, r.log_nv, r.log_k),
            r.m,
            r.k,
            r.nv_eff,
            s.wall_ms,
            fmt_pct(s.avg_cpu_pct),
            fmt_pct(s.peak_cpu_pct),
            s.peak_rss_mb,
        );
    }
    println!("╚{}╝", bar);
    println!();
    println!(
        "  System RAM   : {:.1} MB ({:.1} GB)",
        total_ram,
        total_ram / 1024.0
    );
    println!("  Rayon threads: {}", rayon::current_num_threads());
    println!("  avg_cpu%  = (user+sys CPU time) / wall_time × 100  (>100% = multi-core)");
    println!("  peak_cpu% = max instantaneous CPU% sampled every ~20ms");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // (log_M, log_nv, log_k)
    // nv_eff = log_nv - log_k  (per-party variable count)
    // log_M ≥ 1  (prove_sumfold requires M ≥ 2 and power-of-two)
    let configs: &[(usize, usize, usize)] = &[
        (1, 8, 1),  // nv_eff=7,  M=2,  K=2
        (1, 10, 1), // nv_eff=9,  M=2,  K=2
        (2, 10, 1), // nv_eff=9,  M=4,  K=2
        (2, 10, 2), // nv_eff=8,  M=4,  K=4
        (2, 12, 2), // nv_eff=10, M=4,  K=4
        (3, 12, 2), // nv_eff=10, M=8,  K=4
        (2, 14, 2), // nv_eff=12, M=4,  K=4
        (3, 14, 2), // nv_eff=12, M=8,  K=4
    ];

    let total_ram = total_ram_mb();
    eprintln!(
        "deSNARK lightweight bench  ({} Rayon threads, {:.1} GB RAM)",
        rayon::current_num_threads(),
        total_ram / 1024.0
    );
    eprintln!();

    let mut results = Vec::with_capacity(configs.len());
    for &(log_m, log_nv, log_k) in configs {
        eprintln!(
            "  running ({},{},{})  M={}  K={}  nv_eff={}...",
            log_m,
            log_nv,
            log_k,
            1 << log_m,
            1 << log_k,
            log_nv - log_k
        );
        results.push(bench_config(log_m, log_nv, log_k));
        eprintln!("    done  {:.1}ms\n", results.last().unwrap().total_ms);
    }

    print_results(&results, total_ram);
}
