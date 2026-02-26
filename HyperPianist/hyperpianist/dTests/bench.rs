use std::{marker::PhantomData, time::Instant};

use arithmetic::math::Math;
use ark_bn254::{Bn254, Fr};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet, Stats};
use hyperpianist::{
    prelude::{CustomizedGates, HyperPlonkErrors, MockCircuit},
    HyperPlonkSNARK,
};
use std::{error::Error, path::PathBuf, sync::Arc};
use structopt::StructOpt;
use subroutines::{
    pcs::PolynomialCommitmentScheme,
    poly_iop::PolyIOP,
    BatchProof, DeMkzg, DeMkzgSRS, MultilinearProverParam, MultilinearVerifierParam,
};

mod common;
use common::{test_rng, test_rng_deterministic};

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
    println!("to_master: {to_master}, from_master: {from_master}, bytes_sent: {bytes_sent}, bytes_recv: {bytes_recv}");
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
        println!(
            "key extraction for {} variables: {} us",
            nv,
            start.elapsed().as_micros() as u128
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
        // generate a proof
        let stats_a = Net::stats();
        let proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::d_prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
            &(),
        )?;
        let stats_b = Net::stats();
        print_stats(&stats_a, &stats_b);

        let start = Instant::now();
        for _ in 0..repetition {
            let _proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::d_prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
                &(),
            )?;
        }
        println!(
            "proving for {} variables: {} us",
            nv,
            start.elapsed().as_micros() / repetition as u128
        );

        let mut bytes = Vec::with_capacity(CanonicalSerialize::compressed_size(&proof));
        CanonicalSerialize::serialize_compressed(&proof, &mut bytes).unwrap();
        println!(
            "proof size for {} variables compressed: {} bytes",
            nv,
            bytes.len()
        );

        let mut bytes = Vec::with_capacity(CanonicalSerialize::uncompressed_size(&proof));
        CanonicalSerialize::serialize_uncompressed(&proof, &mut bytes).unwrap();
        println!(
            "proof size for {} variables uncompressed: {} bytes",
            nv,
            bytes.len()
        );

        let all_pi = Net::send_to_master(&circuit.public_inputs);

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
            println!(
                "verifying for {} variables: {} us",
                nv,
                start.elapsed().as_micros() / (repetition * 5) as u128
            );
        }
        Ok(())
    }
}
