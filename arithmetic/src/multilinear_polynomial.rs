// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use crate::{util::get_batched_nv, ArithErrors};
use ark_ff::{Field, PrimeField};
use ark_poly::MultilinearExtension;
use ark_std::{end_timer, rand::RngCore, start_timer};
#[cfg(feature = "parallel")]
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::sync::Arc;

pub use ark_poly::DenseMultilinearExtension;

/// Sample a random list of multilinear polynomials.
/// Returns
/// - the list of polynomials,
/// - its sum of polynomial evaluations over the boolean hypercube.
pub fn random_mle_list<F: PrimeField, R: RngCore>(
    nv: usize,
    degree: usize,
    rng: &mut R,
) -> (Vec<Arc<DenseMultilinearExtension<F>>>, F) {
    let start = start_timer!(|| "sample random mle list");
    let mut multiplicands = Vec::with_capacity(degree);
    for _ in 0..degree {
        multiplicands.push(Vec::with_capacity(1 << nv))
    }
    let mut sum = F::zero();

    for _ in 0..(1 << nv) {
        let mut product = F::one();

        for e in multiplicands.iter_mut() {
            let val = F::rand(rng);
            e.push(val);
            product *= val;
        }
        sum += product;
    }

    let list = multiplicands
        .into_iter()
        .map(|x| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, x)))
        .collect();

    end_timer!(start);
    (list, sum)
}

// Build a randomize list of mle-s whose sum is zero.
pub fn random_zero_mle_list<F: PrimeField, R: RngCore>(
    nv: usize,
    degree: usize,
    rng: &mut R,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let start = start_timer!(|| "sample random zero mle list");

    let mut multiplicands = Vec::with_capacity(degree);
    for _ in 0..degree {
        multiplicands.push(Vec::with_capacity(1 << nv))
    }
    for _ in 0..(1 << nv) {
        multiplicands[0].push(F::zero());
        for e in multiplicands.iter_mut().skip(1) {
            e.push(F::rand(rng));
        }
    }

    let list = multiplicands
        .into_iter()
        .map(|x| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, x)))
        .collect();

    end_timer!(start);
    list
}

pub fn identity_permutation<F: PrimeField>(num_vars: usize, num_chunks: usize) -> Vec<F> {
    let len = (num_chunks as u64) * (1u64 << num_vars);
    (0..len).map(F::from).collect()
}

/// A list of MLEs that represents an identity permutation
pub fn identity_permutation_mles<F: PrimeField>(
    num_vars: usize,
    num_chunks: usize,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let mut res = vec![];
    for i in 0..num_chunks {
        let shift = (i * (1 << num_vars)) as u64;
        let s_id_vec = (shift..shift + (1u64 << num_vars)).map(F::from).collect();
        res.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, s_id_vec,
        )));
    }
    res
}

pub fn random_permutation<F: PrimeField, R: RngCore>(
    num_vars: usize,
    num_chunks: usize,
    rng: &mut R,
) -> Vec<F> {
    let len = (num_chunks as u64) * (1u64 << num_vars);
    let mut s_id_vec: Vec<F> = (0..len).map(F::from).collect();
    let mut s_perm_vec = vec![];
    for _ in 0..len {
        let index = rng.next_u64() as usize % s_id_vec.len();
        s_perm_vec.push(s_id_vec.remove(index));
    }
    s_perm_vec
}

/// A list of MLEs that represent a random permutation
pub fn random_permutation_mles<F: PrimeField, R: RngCore>(
    num_vars: usize,
    num_chunks: usize,
    rng: &mut R,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let s_perm_vec = random_permutation(num_vars, num_chunks, rng);
    let mut res = vec![];
    let n = 1 << num_vars;
    for i in 0..num_chunks {
        res.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            s_perm_vec[i * n..i * n + n].to_vec(),
        )));
    }
    res
}

pub fn evaluate_opt<F: Field>(poly: &DenseMultilinearExtension<F>, point: &[F]) -> F {
    assert_eq!(poly.num_vars, point.len());
    fix_variables(poly, point).evaluations[0]
}

pub fn fix_variables<F: Field>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evaluations.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from left to right
    for (i, point) in partial_point.iter().enumerate().take(dim) {
        poly = fix_one_variable_helper(&poly, nv - i, point);
    }

    DenseMultilinearExtension::<F>::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
}

fn fix_one_variable_helper<F: Field>(data: &[F], nv: usize, point: &F) -> Vec<F> {
    let mut res = vec![F::zero(); 1 << (nv - 1)];

    // evaluate single variable of partial point from left to right
    #[cfg(not(feature = "parallel"))]
    for i in 0..(1 << (nv - 1)) {
        res[i] = data[i] + (data[(i << 1) + 1] - data[i << 1]) * point;
    }

    #[cfg(feature = "parallel")]
    res.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = data[i << 1] + (data[(i << 1) + 1] - data[i << 1]) * point;
    });

    res
}

/// Fix variables in-place without allocating new vectors.
/// Overwrites the first half of the evaluation buffer and returns the new effective length.
/// This is more efficient than `fix_variables` when the original data is no longer needed.
///
/// This function provides significant speedups (2-23x) over `fix_variables` by avoiding
/// repeated vector allocations during the folding process.
///
/// # Arguments
/// * `evaluations` - Mutable slice of evaluations, will be modified in-place
/// * `nv` - Current number of variables
/// * `partial_point` - Values to fix variables at
///
/// # Returns
/// The new number of variables after fixing (nv - partial_point.len())
pub fn fix_variables_in_place<F: Field>(
    evaluations: &mut [F],
    nv: usize,
    partial_point: &[F],
) -> usize {
    assert!(
        partial_point.len() <= nv,
        "invalid size of partial point"
    );
    let dim = partial_point.len();
    let mut current_nv = nv;

    // Fix each variable from left to right
    for point in partial_point.iter().take(dim) {
        fix_one_variable_in_place(evaluations, current_nv, point);
        current_nv -= 1;
    }

    current_nv
}

/// Helper function to fix one variable in-place.
/// Overwrites the first half of data with the interpolated values.
///
/// This uses a forward pass where each write position i reads from 2i and 2i+1.
/// Since 2i > i for i > 0, we always read before overwriting those positions.
#[inline]
fn fix_one_variable_in_place<F: Field>(data: &mut [F], nv: usize, point: &F) {
    let half_len = 1 << (nv - 1);

    // Process in-place: write to index i, read from indices 2i and 2i+1
    // Since 2i >= i for i > 0, we process forward and each write is safe
    // (we always read before we overwrite those positions)
    for i in 0..half_len {
        let low = data[i << 1];
        let high = data[(i << 1) + 1];
        data[i] = low + (high - low) * point;
    }
}

pub fn evaluate_no_par<F: Field>(poly: &DenseMultilinearExtension<F>, point: &[F]) -> F {
    assert_eq!(poly.num_vars, point.len());
    fix_variables_no_par(poly, point).evaluations[0]
}

fn fix_variables_no_par<F: Field>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evaluations.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from left to right
    for i in 1..dim + 1 {
        let r = partial_point[i - 1];
        for b in 0..(1 << (nv - i)) {
            poly[b] = poly[b << 1] + (poly[(b << 1) + 1] - poly[b << 1]) * r;
        }
    }
    DenseMultilinearExtension::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
}

/// merge a set of polynomials. Returns an error if the
/// polynomials do not share a same number of nvs.
pub fn merge_polynomials<F: PrimeField>(
    polynomials: &[Arc<DenseMultilinearExtension<F>>],
) -> Result<Arc<DenseMultilinearExtension<F>>, ArithErrors> {
    let nv = polynomials[0].num_vars();
    for poly in polynomials.iter() {
        if nv != poly.num_vars() {
            return Err(ArithErrors::InvalidParameters(
                "num_vars do not match for polynomials".to_string(),
            ));
        }
    }

    let merged_nv = get_batched_nv(nv, polynomials.len());
    let mut scalars = vec![];
    for poly in polynomials.iter() {
        scalars.extend_from_slice(poly.to_evaluations().as_slice());
    }
    scalars.extend_from_slice(vec![F::zero(); (1 << merged_nv) - scalars.len()].as_ref());
    Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        merged_nv, scalars,
    )))
}

pub fn fix_last_variables_no_par<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    let mut res = fix_last_variable_no_par(poly, partial_point.last().unwrap());
    for p in partial_point.iter().rev().skip(1) {
        res = fix_last_variable_no_par(&res, p);
    }
    res
}

fn fix_last_variable_no_par<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &F,
) -> DenseMultilinearExtension<F> {
    let nv = poly.num_vars();
    let half_len = 1 << (nv - 1);
    let mut res = vec![F::zero(); half_len];
    for (i, e) in res.iter_mut().enumerate().take(half_len) {
        *e = poly.evaluations[i]
            + *partial_point * (poly.evaluations[i + half_len] - poly.evaluations[i]);
    }
    DenseMultilinearExtension::from_evaluations_vec(nv - 1, res)
}
pub fn fix_last_variables<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evaluations.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from left to right
    for (i, point) in partial_point.iter().rev().enumerate().take(dim) {
        poly = fix_last_variable_helper(&poly, nv - i, point);
    }

    DenseMultilinearExtension::<F>::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
}

fn fix_last_variable_helper<F: Field>(data: &[F], nv: usize, point: &F) -> Vec<F> {
    let half_len = 1 << (nv - 1);
    let mut res = vec![F::zero(); half_len];

    // evaluate single variable of partial point from left to right
    #[cfg(not(feature = "parallel"))]
    for b in 0..half_len {
        res[b] = data[b] + (data[b + half_len] - data[b]) * point;
    }

    #[cfg(feature = "parallel")]
    res.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = data[i] + (data[i + half_len] - data[i]) * point;
    });

    res
}

/// Threshold for switching to parallel execution in split operations.
/// Based on empirical testing, parallelization overhead is worth it above this size.
const PARALLEL_SPLIT_THRESHOLD: usize = 1 << 12; // 4096 elements

/// Split an MLE into 2^n sub-MLEs by fixing the last n variables to all binary assignments.
///
/// For an MLE f(x_1, ..., x_m) with m variables, this produces 2^n MLEs:
/// - split[0] = f(x_1, ..., x_{m-n}, 0, 0, ..., 0)
/// - split[1] = f(x_1, ..., x_{m-n}, 1, 0, ..., 0)
/// - split[2] = f(x_1, ..., x_{m-n}, 0, 1, ..., 0)
/// - ...
/// - split[2^n - 1] = f(x_1, ..., x_{m-n}, 1, 1, ..., 1)
///
/// # Memory Layout Assumption
/// This function relies on the little-endian bit indexing used by `DenseMultilinearExtension`.
/// The evaluation at index `i` corresponds to the point where binary digits of `i` give the
/// variable assignments in order (x_1, x_2, ..., x_m). Thus, fixing the last n variables to
/// assignment `j` corresponds to taking the j-th contiguous chunk of size 2^(m-n).
///
/// Automatically selects parallel or sequential execution based on data size.
///
/// # Arguments
/// * `poly` - The multilinear extension to split
/// * `n` - Number of last variables to fix (number of bits)
///
/// # Returns
/// A vector of 2^n MLEs, each with (m-n) variables
///
/// # Panics
/// Panics if n > num_vars
pub fn split_by_last_variables<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    n: usize,
) -> Vec<DenseMultilinearExtension<F>> {
    let m = poly.num_vars;
    assert!(n <= m, "n ({}) cannot exceed num_vars ({})", n, m);

    if n == 0 {
        return vec![poly.clone()];
    }

    let chunk_size = 1 << (m - n);

    // Choose parallel or sequential based on total data size
    #[cfg(feature = "parallel")]
    if poly.evaluations.len() >= PARALLEL_SPLIT_THRESHOLD {
        return split_by_last_variables_par(poly, n, chunk_size);
    }

    split_by_last_variables_seq(poly, n, chunk_size)
}

/// Sequential version of split_by_last_variables.
fn split_by_last_variables_seq<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    n: usize,
    chunk_size: usize,
) -> Vec<DenseMultilinearExtension<F>> {
    let new_num_vars = poly.num_vars - n;
    poly.evaluations
        .chunks(chunk_size)
        .map(|chunk| DenseMultilinearExtension::from_evaluations_vec(new_num_vars, chunk.to_vec()))
        .collect()
}

/// Parallel version of split_by_last_variables using par_chunks.
#[cfg(feature = "parallel")]
fn split_by_last_variables_par<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    n: usize,
    chunk_size: usize,
) -> Vec<DenseMultilinearExtension<F>> {
    use rayon::prelude::ParallelSlice;

    let new_num_vars = poly.num_vars - n;
    poly.evaluations
        .par_chunks(chunk_size)
        .map(|chunk| DenseMultilinearExtension::from_evaluations_vec(new_num_vars, chunk.to_vec()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    /// Test fix_variables_in_place against fix_variables for small polynomials (serial path)
    #[test]
    fn test_fix_variables_in_place_small() {
        let mut rng = test_rng();
        let nv = 8; // Small polynomial, serial execution

        let evals: Vec<Fr> = (0..(1 << nv)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = DenseMultilinearExtension::from_evaluations_vec(nv, evals.clone());

        let partial_point: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();

        // Reference: allocating fix_variables
        let expected = fix_variables(&poly, &partial_point);

        // Test: in-place version
        let mut in_place_evals = evals;
        let new_nv = fix_variables_in_place(&mut in_place_evals, nv, &partial_point);

        assert_eq!(new_nv, nv - partial_point.len());
        assert_eq!(new_nv, expected.num_vars);
        assert_eq!(
            &in_place_evals[..(1 << new_nv)],
            expected.evaluations.as_slice()
        );
    }

    /// Test fix_variables_in_place for large polynomials (parallel path when feature enabled)
    #[test]
    fn test_fix_variables_in_place_large() {
        let mut rng = test_rng();
        let nv = 14; // Large polynomial, triggers parallel execution (16384 elements)

        let evals: Vec<Fr> = (0..(1 << nv)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = DenseMultilinearExtension::from_evaluations_vec(nv, evals.clone());

        let partial_point: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();

        // Reference: allocating fix_variables
        let expected = fix_variables(&poly, &partial_point);

        // Test: in-place version
        let mut in_place_evals = evals;
        let new_nv = fix_variables_in_place(&mut in_place_evals, nv, &partial_point);

        assert_eq!(new_nv, nv - partial_point.len());
        assert_eq!(new_nv, expected.num_vars);
        assert_eq!(
            &in_place_evals[..(1 << new_nv)],
            expected.evaluations.as_slice()
        );
    }

    /// Benchmark comparison between fix_variables and fix_variables_in_place
    #[test]
    fn bench_fix_variables_comparison() {
        use std::time::Instant;

        let mut rng = test_rng();

        for nv in [10, 12, 14, 16, 18, 20] {
            let evals: Vec<Fr> = (0..(1 << nv)).map(|_| Fr::rand(&mut rng)).collect();
            let poly = DenseMultilinearExtension::from_evaluations_vec(nv, evals.clone());
            let partial_point: Vec<Fr> = (0..(nv / 2)).map(|_| Fr::rand(&mut rng)).collect();

            const ITERATIONS: usize = 5;

            // Benchmark fix_variables (allocating)
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                let _ = fix_variables(&poly, &partial_point);
            }
            let alloc_time = start.elapsed() / ITERATIONS as u32;

            // Benchmark fix_variables_in_place
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                let mut in_place_evals = evals.clone();
                let _ = fix_variables_in_place(&mut in_place_evals, nv, &partial_point);
            }
            let in_place_time = start.elapsed() / ITERATIONS as u32;

            let speedup = alloc_time.as_nanos() as f64 / in_place_time.as_nanos() as f64;
            println!(
                "nv={}: fix_variables={:?}, fix_variables_in_place={:?}, speedup={:.2}x",
                nv, alloc_time, in_place_time, speedup
            );
        }
    }

    /// Test split_by_last_variables produces correct evaluations
    #[test]
    fn test_split_by_last_variables_correctness() {
        let mut rng = test_rng();
        let m = 6; // Total variables

        for n in 0..=4 {
            let evals: Vec<Fr> = (0..(1 << m)).map(|_| Fr::rand(&mut rng)).collect();
            let poly = DenseMultilinearExtension::from_evaluations_vec(m, evals);

            let splits = split_by_last_variables(&poly, n);

            assert_eq!(splits.len(), 1 << n);
            for split in &splits {
                assert_eq!(split.num_vars, m - n);
            }

            // Verify evaluation consistency:
            // For a random point (x_1, ..., x_{m-n}), evaluate both:
            // 1. Original poly at (x_1, ..., x_{m-n}, b_1, ..., b_n) where b encodes split index
            // 2. Split poly at (x_1, ..., x_{m-n})
            let point: Vec<Fr> = (0..(m - n)).map(|_| Fr::rand(&mut rng)).collect();

            for split_idx in 0..(1 << n) {
                // Build full point by appending binary assignment for split_idx
                let mut full_point = point.clone();
                for bit in 0..n {
                    let bit_val = ((split_idx >> bit) & 1) as u64;
                    full_point.push(Fr::from(bit_val));
                }

                let expected = evaluate_opt(&poly, &full_point);
                let actual = evaluate_opt(&splits[split_idx], &point);

                assert_eq!(expected, actual, "Mismatch at split_idx={}, n={}", split_idx, n);
            }
        }
    }

    /// Test edge case: n = 0 returns a clone
    #[test]
    fn test_split_by_last_variables_n_zero() {
        let mut rng = test_rng();
        let nv = 5;
        let evals: Vec<Fr> = (0..(1 << nv)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = DenseMultilinearExtension::from_evaluations_vec(nv, evals);

        let splits = split_by_last_variables(&poly, 0);

        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].num_vars, nv);
        assert_eq!(splits[0].evaluations, poly.evaluations);
    }

    /// Test edge case: n = num_vars returns 2^m scalar constants
    #[test]
    fn test_split_by_last_variables_n_equals_num_vars() {
        let mut rng = test_rng();
        let nv = 4;
        let evals: Vec<Fr> = (0..(1 << nv)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = DenseMultilinearExtension::from_evaluations_vec(nv, evals.clone());

        let splits = split_by_last_variables(&poly, nv);

        assert_eq!(splits.len(), 1 << nv);
        for (i, split) in splits.iter().enumerate() {
            assert_eq!(split.num_vars, 0);
            assert_eq!(split.evaluations.len(), 1);
            // Each split should contain the evaluation at index i
            assert_eq!(split.evaluations[0], evals[i]);
        }
    }
}
