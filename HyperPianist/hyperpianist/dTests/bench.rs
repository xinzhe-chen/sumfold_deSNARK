use std::{
    error::Error,
    marker::PhantomData,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use arithmetic::math::Math;
use ark_bn254::{Bn254, Fr};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet, Stats};
use hyperpianist::{
    prelude::{CustomizedGates, HyperPlonkErrors, MockCircuit},
    HyperPlonkSNARK,
};
use structopt::StructOpt;
use subroutines::{
    pcs::PolynomialCommitmentScheme,
    poly_iop::PolyIOP,
    BatchProof, DeMkzg, DeMkzgSRS, MultilinearProverParam, MultilinearVerifierParam,
};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

mod common;
use common::{test_rng, test_rng_deterministic};

// ─── CPU time via getrusage (consistent with sumfold_deSNARK) ────────────────

/// Returns cumulative (user_ms + sys_ms) CPU time for this process.
fn get_cpu_total_ms() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let user = usage.ru_utime.tv_sec as f64 * 1e3 + usage.ru_utime.tv_usec as f64 * 1e-3;
    let sys  = usage.ru_stime.tv_sec as f64 * 1e3 + usage.ru_stime.tv_usec as f64 * 1e-3;
    user + sys
}

// ─── Peak RSS sampler (background thread, 5ms poll) ──────────────────────────

fn current_rss_mb() -> f64 {
    let mut sys = System::new();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        ProcessRefreshKind::new().with_memory(),
    );
    sys.process(pid).map(|p| p.memory() as f64 / (1024.0 * 1024.0)).unwrap_or(0.0)
}

struct PeakSampler {
    stop_flag: Arc<AtomicBool>,
    peak_mb:   Arc<Mutex<f64>>,
    handle:    Option<thread::JoinHandle<()>>,
}

impl PeakSampler {
    fn start(initial_rss: f64) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let peak_mb   = Arc::new(Mutex::new(initial_rss));
        let flag = stop_flag.clone();
        let peak = peak_mb.clone();
        let pid  = Pid::from_u32(std::process::id());
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
                    if mb > *g { *g = mb; }
                }
                thread::sleep(Duration::from_millis(5));
            }
        });
        Self { stop_flag, peak_mb, handle: Some(handle) }
    }
    fn stop(mut self) -> f64 {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() { h.join().ok(); }
        (*self.peak_mb.lock().unwrap()).max(current_rss_mb())
    }
}

#[derive(Debug, StructOpt)]
#[structopt(name = "hyperpianist-bench", about = "HyperPianist DeMkzg benchmark")]
struct Opt {
    /// Party id (0 = master)
    id: usize,

    /// Hosts file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// Number of variables (total constraints = 2^num_vars)
    num_vars: usize,
}

fn main() {
    let opt = Opt::from_args();
    Net::init_from_file(opt.input.to_str().unwrap(), opt.id);

    if Net::am_master() {
        eprintln!(
            "# Rayon threads: {} (RAYON_NUM_THREADS={})",
            rayon::current_num_threads(),
            std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string()),
        );
    }

    let max_nv = opt.num_vars;

    // Generate or load DeMkzg SRS sized to the requested num_vars
    let pcs_srs = {
        match read_deMkzg_srs(max_nv) {
            Ok(p) => p,
            Err(_e) => {
                let mut srs_rng = test_rng_deterministic();
                let srs =
                    DeMkzg::<Bn254>::gen_srs_for_testing(&mut srs_rng, max_nv).unwrap();
                let (prover, verifier) =
                    DeMkzg::trim(&srs, None, Some(max_nv)).unwrap();
                write_deMkzg_srs(&prover, &verifier, max_nv);
                DeMkzgSRS::Processed((prover, verifier))
            },
        }
    };

    Helper::<DeMkzg<Bn254>>::bench_vanilla_plonk(&pcs_srs, opt.num_vars).unwrap();

    Net::deinit();
}

fn read_deMkzg_srs(max_nv: usize) -> Result<DeMkzgSRS<Bn254>, Box<dyn Error>> {
    let sub_prover_setup_filepath = format!(
        "deMkzg-SubProver{}-max{}.paras",
        Net::party_id(),
        max_nv
    );
    let verifier_setup_filepath = format!(
        "deMkzg-Verifier{}-max{}.paras",
        Net::party_id(),
        max_nv
    );
    let prover_setup = {
        let file = std::fs::File::open(sub_prover_setup_filepath)?;
        MultilinearProverParam::deserialize_uncompressed_unchecked(std::io::BufReader::new(file))?
    };

    let verifier_setup = {
        let file = std::fs::File::open(verifier_setup_filepath)?;
        MultilinearVerifierParam::deserialize_uncompressed_unchecked(std::io::BufReader::new(file))?
    };
    Ok(DeMkzgSRS::Processed((prover_setup, verifier_setup)))
}

fn write_deMkzg_srs(
    prover: &MultilinearProverParam<Bn254>,
    verifier: &MultilinearVerifierParam<Bn254>,
    max_nv: usize,
) {
    let sub_prover_setup_filepath = format!(
        "deMkzg-SubProver{}-max{}.paras",
        Net::party_id(),
        max_nv
    );
    let verifier_setup_filepath = format!(
        "deMkzg-Verifier{}-max{}.paras",
        Net::party_id(),
        max_nv
    );

    let file = std::fs::File::create(sub_prover_setup_filepath).unwrap();
    prover
        .serialize_uncompressed(std::io::BufWriter::new(file))
        .unwrap();

    let file = std::fs::File::create(verifier_setup_filepath).unwrap();
    verifier
        .serialize_uncompressed(std::io::BufWriter::new(file))
        .unwrap();
}

fn print_stats(before: &Stats, after: &Stats) {
    let to_master = after.to_master - before.to_master;
    let from_master = after.from_master - before.from_master;
    let bytes_sent = after.bytes_sent - before.bytes_sent;
    let bytes_recv = after.bytes_recv - before.bytes_recv;
    eprintln!("to_master: {to_master}, from_master: {from_master}, bytes_sent: {bytes_sent}, bytes_recv: {bytes_recv}");
}

struct Helper<PCS>(PhantomData<PCS>);

impl<PCS> Helper<PCS>
where
    PCS: PolynomialCommitmentScheme<
        Bn254,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
        BatchProof = BatchProof<Bn254, PCS>,
    >,
{
    fn bench_vanilla_plonk(
        pcs_srs: &PCS::SRS,
        nv: usize,
    ) -> Result<(), HyperPlonkErrors> {
        let vanilla_gate = CustomizedGates::vanilla_plonk_gate();
        Self::bench_mock_circuit_zkp_helper(nv, &vanilla_gate, pcs_srs)?;
        Ok(())
    }

    fn bench_mock_circuit_zkp_helper(
        nv: usize,
        gate: &CustomizedGates,
        pcs_srs: &PCS::SRS,
    ) -> Result<(), HyperPlonkErrors> {
        let nv = nv - Net::n_parties().log_2();
        let repetition = if nv <= 20 {
            20
        } else if nv <= 22 {
            10
        } else {
            5
        };

        // Peak RSS sampled throughout the entire function
        let sampler = PeakSampler::start(current_rss_mb());

        // Reset network stats for this run
        Net::reset_stats();

        let mut rng = test_rng();
        //==========================================================
        let circuit = MockCircuit::<Fr>::d_new(1 << nv, gate, &mut rng);
        assert!(circuit.is_satisfied());
        let index = circuit.index;
        //==========================================================
        // generate pk and vks
        let start = Instant::now();

        let stats_a = Net::stats();
        let (pk, vk) =
            <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::d_preprocess(&index, pcs_srs)?;
        let setup_us = start.elapsed().as_micros() as u128;
        eprintln!(
            "key extraction for {} variables: {} us",
            nv, setup_us,
        );
        let stats_b = Net::stats();
        print_stats(&stats_a, &stats_b);

        // Re-synchronize
        Net::recv_from_master_uniform(if Net::am_master() {
            Some(1usize)
        } else {
            None
        });

        //==========================================================
        // generate a proof (warmup — not timed)
        let proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::d_prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
            &(),
        )?;

        // Reset comm stats so they only cover the timed proving runs
        Net::reset_stats();

        // ─── CPU/wall measurement: scoped to timed proving only ──────────────
        let cpu_before = get_cpu_total_ms();
        let wall_start = Instant::now();

        let start = Instant::now();
        for _ in 0..repetition {
            let _proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::d_prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
                &(),
            )?;
        }
        let prover_us = start.elapsed().as_micros() / repetition as u128;
        // ─── CPU/wall measurement end (proving only) ─────────────────────────
        let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
        let cpu_ms = get_cpu_total_ms() - cpu_before;
        // Capture comm stats right after timed proving (before verify)
        let prove_stats = Net::stats();
        eprintln!(
            "proving for {} variables: {} us",
            nv, prover_us,
        );

        let mut bytes = Vec::with_capacity(CanonicalSerialize::compressed_size(&proof));
        CanonicalSerialize::serialize_compressed(&proof, &mut bytes).unwrap();
        let proof_bytes_compressed = bytes.len();
        eprintln!(
            "proof size for {} variables compressed: {} bytes",
            nv, proof_bytes_compressed,
        );

        let mut bytes = Vec::with_capacity(CanonicalSerialize::uncompressed_size(&proof));
        CanonicalSerialize::serialize_uncompressed(&proof, &mut bytes).unwrap();
        eprintln!(
            "proof size for {} variables uncompressed: {} bytes",
            nv,
            bytes.len()
        );

        let all_pi = Net::send_to_master(&circuit.public_inputs);

        let mut verifier_us = 0u128;
        if Net::am_master() {
            let vk = vk.unwrap();
            let pi = all_pi.unwrap().concat();
            let proof = proof.unwrap();
            //==========================================================
            // verify a proof
            let start = Instant::now();
            for _ in 0..(repetition * 5) {
                let verify =
                    <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::verify(&vk, &pi, &proof)?;
                assert!(verify);
            }
            verifier_us = start.elapsed().as_micros() / (repetition * 5) as u128;
            eprintln!(
                "verifying for {} variables: {} us",
                nv, verifier_us,
            );
        }

        // ─── Peak RSS capture ────────────────────────────────────────────────
        let peak_rss_mb = sampler.stop();

        // Use prove_stats (captured right after timed proving) for comm data,
        // averaged per proving call.
        let avg_bytes_sent = prove_stats.bytes_sent / repetition;
        let avg_bytes_recv = prove_stats.bytes_recv / repetition;

        // ─── Output CSV line on stdout (master only) ─────────────────────────
        // Format: setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,cpu_ms,wall_ms,peak_rss_mb
        // (nv, M, K, avg_cpu_pct are computed by the shell script)
        if Net::am_master() {
            println!(
                "{:.3},{:.3},{:.3},{},{},{},{:.3},{:.3},{:.1}",
                setup_us as f64 / 1000.0,
                prover_us as f64 / 1000.0,
                verifier_us as f64 / 1000.0,
                proof_bytes_compressed,
                avg_bytes_sent,
                avg_bytes_recv,
                cpu_ms,
                wall_ms,
                peak_rss_mb,
            );
        }

        Ok(())
    }
}
