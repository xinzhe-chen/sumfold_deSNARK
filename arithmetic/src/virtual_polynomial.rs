// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module defines our main mathematical object `VirtualPolynomial`; and
//! various functions associated with it.

use crate::{errors::ArithErrors, multilinear_polynomial::random_zero_mle_list, random_mle_list};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::CanonicalSerialize;
use ark_std::{
    rand::{Rng, RngCore},
};
use rayon::prelude::*;
use std::{cmp::max, collections::HashMap, marker::PhantomData, ops::Add, sync::Arc};

#[rustfmt::skip]
/// A virtual polynomial is a sum of products of multilinear polynomials;
/// where the multilinear polynomials are stored via their multilinear
/// extensions:  `(coefficient, DenseMultilinearExtension)`
///
/// * Number of products n = `polynomial.products.len()`,
/// * Number of multiplicands of ith product m_i =
///   `polynomial.products[i].1.len()`,
/// * Coefficient of ith product c_i = `polynomial.products[i].0`
///
/// The resulting polynomial is
///
/// $$ \sum_{i=0}^{n} c_i \cdot \prod_{j=0}^{m_i} P_{ij} $$
///
/// Example:
///  f = c0 * f0 * f1 * f2 + c1 * f3 * f4
/// where f0 ... f4 are multilinear polynomials
///
/// - flattened_ml_extensions stores the multilinear extension representation of
///   f0, f1, f2, f3 and f4
/// - products is
///   \[
///   (c0, \[0, 1, 2\]),
///   (c1, \[3, 4\])
///   \]
/// - raw_pointers_lookup_table maps fi to i
///
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VirtualPolynomial<F: PrimeField> {
    /// Aux information about the multilinear polynomial
    pub aux_info: VPAuxInfo<F>,
    /// list of reference to products (as usize) of multilinear extension
    pub products: Vec<(F, Vec<usize>)>,
    /// Stores multilinear extensions in which product multiplicand can refer
    /// to.
    pub flattened_ml_extensions: Vec<Arc<DenseMultilinearExtension<F>>>,
    /// Pointers to the above poly extensions
    pub raw_pointers_lookup_table: HashMap<*const DenseMultilinearExtension<F>, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, CanonicalSerialize)]
/// Auxiliary information about the multilinear polynomial
pub struct VPAuxInfo<F: PrimeField> {
    /// max number of multiplicands in each product
    pub max_degree: usize,
    /// number of variables of the polynomial
    pub num_variables: usize,
    /// Associated field
    #[doc(hidden)]
    pub phantom: PhantomData<F>,
}

impl<F: PrimeField> Add for &VirtualPolynomial<F> {
    type Output = VirtualPolynomial<F>;
    fn add(self, other: &VirtualPolynomial<F>) -> Self::Output {
        let mut res = self.clone();
        for products in other.products.iter() {
            let cur: Vec<Arc<DenseMultilinearExtension<F>>> = products
                .1
                .iter()
                .map(|&x| other.flattened_ml_extensions[x].clone())
                .collect();

            res.add_mle_list(cur, products.0)
                .expect("add product failed");
        }

        res
    }
}

// TODO: convert this into a trait
impl<F: PrimeField> VirtualPolynomial<F> {
    /// Creates an empty virtual polynomial with `num_variables`.
    pub fn new(num_variables: usize) -> Self {
        VirtualPolynomial {
            aux_info: VPAuxInfo {
                max_degree: 0,
                num_variables,
                phantom: PhantomData,
            },
            products: Vec::new(),
            flattened_ml_extensions: Vec::new(),
            raw_pointers_lookup_table: HashMap::new(),
        }
    }

    /// Creates an new virtual polynomial from a MLE and its coefficient.
    pub fn new_from_mle(mle: &Arc<DenseMultilinearExtension<F>>, coefficient: F) -> Self {
        let mle_ptr: *const DenseMultilinearExtension<F> = Arc::as_ptr(mle);
        let mut hm = HashMap::new();
        hm.insert(mle_ptr, 0);

        VirtualPolynomial {
            aux_info: VPAuxInfo {
                // The max degree is the max degree of any individual variable
                max_degree: 1,
                num_variables: mle.num_vars,
                phantom: PhantomData,
            },
            // here `0` points to the first polynomial of `flattened_ml_extensions`
            products: vec![(coefficient, vec![0])],
            flattened_ml_extensions: vec![mle.clone()],
            raw_pointers_lookup_table: hm,
        }
    }

    /// Add a product of list of multilinear extensions to self
    /// Returns an error if the list is empty, or the MLE has a different
    /// `num_vars` from self.
    ///
    /// The MLEs will be multiplied together, and then multiplied by the scalar
    /// `coefficient`.
    pub fn add_mle_list(
        &mut self,
        mle_list: impl IntoIterator<Item = Arc<DenseMultilinearExtension<F>>>,
        coefficient: F,
    ) -> Result<(), ArithErrors> {
        let mle_list: Vec<Arc<DenseMultilinearExtension<F>>> = mle_list.into_iter().collect();
        let mut indexed_product = Vec::with_capacity(mle_list.len());

        if mle_list.is_empty() {
            return Err(ArithErrors::InvalidParameters(
                "input mle_list is empty".to_string(),
            ));
        }

        self.aux_info.max_degree = max(self.aux_info.max_degree, mle_list.len());

        for mle in mle_list {
            if mle.num_vars != self.aux_info.num_variables {
                return Err(ArithErrors::InvalidParameters(format!(
                    "product has a multiplicand with wrong number of variables {} vs {}",
                    mle.num_vars, self.aux_info.num_variables
                )));
            }

            let mle_ptr: *const DenseMultilinearExtension<F> = Arc::as_ptr(&mle);
            if let Some(index) = self.raw_pointers_lookup_table.get(&mle_ptr) {
                indexed_product.push(*index)
            } else {
                let curr_index = self.flattened_ml_extensions.len();
                self.flattened_ml_extensions.push(mle.clone());
                self.raw_pointers_lookup_table.insert(mle_ptr, curr_index);
                indexed_product.push(curr_index);
            }
        }
        self.products.push((coefficient, indexed_product));
        Ok(())
    }

    /// Replace all MLEs in this VirtualPolynomial with a new set of MLEs.
    /// Rebuilds the raw_pointers_lookup_table accordingly.
    ///
    /// Ported from HyperPianist: .agent/HyperPianist/arithmetic/src/virtual_polynomial.rs
    pub fn replace_mles(&mut self, new_mle_list: Vec<Arc<DenseMultilinearExtension<F>>>) {
        self.flattened_ml_extensions = new_mle_list;
        self.raw_pointers_lookup_table = HashMap::new();
        for (index, mle) in self.flattened_ml_extensions.iter().enumerate() {
            let mle_ptr = Arc::as_ptr(mle);
            self.raw_pointers_lookup_table.insert(mle_ptr, index);
        }
    }

    /// Multiple the current VirtualPolynomial by an MLE:
    /// - add the MLE to the MLE list;
    /// - multiple each product by MLE and its coefficient.
    ///
    /// Returns an error if the MLE has a different `num_vars` from self.
    pub fn mul_by_mle(
        &mut self,
        mle: Arc<DenseMultilinearExtension<F>>,
        coefficient: F,
    ) -> Result<(), ArithErrors> {

        if mle.num_vars != self.aux_info.num_variables {
            return Err(ArithErrors::InvalidParameters(format!(
                "product has a multiplicand with wrong number of variables {} vs {}",
                mle.num_vars, self.aux_info.num_variables
            )));
        }

        let mle_ptr: *const DenseMultilinearExtension<F> = Arc::as_ptr(&mle);

        // check if this mle already exists in the virtual polynomial
        let mle_index = match self.raw_pointers_lookup_table.get(&mle_ptr) {
            Some(&p) => p,
            None => {
                self.raw_pointers_lookup_table
                    .insert(mle_ptr, self.flattened_ml_extensions.len());
                self.flattened_ml_extensions.push(mle);
                self.flattened_ml_extensions.len() - 1
            },
        };

        for (prod_coef, indices) in self.products.iter_mut() {
            // - add the MLE to the MLE list;
            // - multiple each product by MLE and its coefficient.
            indices.push(mle_index);
            *prod_coef *= coefficient;
        }

        // increase the max degree by one as the MLE has degree 1.
        self.aux_info.max_degree += 1;
        Ok(())
    }

    /// Evaluate the virtual polynomial at point `point`.
    /// Returns an error is point.len() does not match `num_variables`.
    pub fn evaluate(&self, point: &[F]) -> Result<F, ArithErrors> {

        if self.aux_info.num_variables != point.len() {
            return Err(ArithErrors::InvalidParameters(format!(
                "wrong number of variables {} vs {}",
                self.aux_info.num_variables,
                point.len()
            )));
        }

        let evals: Vec<F> = self
            .flattened_ml_extensions
            .iter()
            .map(|x| {
                x.evaluate(point).unwrap() // safe unwrap here since we have
                                           // already checked that num_var
                                           // matches
            })
            .collect();

        let res = self
            .products
            .iter()
            .map(|(c, p)| *c * p.iter().map(|&i| evals[i]).product::<F>())
            .sum();
        Ok(res)
    }

    /// Sample a random virtual polynomial, return the polynomial and its sum.
    pub fn rand<R: RngCore>(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
        rng: &mut R,
    ) -> Result<(Self, F), ArithErrors> {

        let mut sum = F::zero();
        let mut poly = VirtualPolynomial::new(nv);
        for _ in 0..num_products {
            let num_multiplicands =
                rng.gen_range(num_multiplicands_range.0..num_multiplicands_range.1);
            let (product, product_sum) = random_mle_list(nv, num_multiplicands, rng);
            let coefficient = F::rand(rng);
            poly.add_mle_list(product.into_iter(), coefficient)?;
            sum += product_sum * coefficient;
        }

        Ok((poly, sum))
    }

    pub fn compute_sums<Fr: PrimeField>(
        f_hats: &[VirtualPolynomial<Fr>],
    ) -> Result<Vec<Fr>, ArithErrors> {
        let mut sums = Vec::with_capacity(f_hats.len());

        for poly in f_hats {
            let mut poly_sum = Fr::zero();

            for (coefficient, mle_indices) in poly.products.iter() {
                let mut product_sum = Fr::one();

                for &index in mle_indices {
                    let mle = &poly.flattened_ml_extensions[index];
                    let mle_sum = mle.evaluations.iter().sum::<Fr>();
                    product_sum *= mle_sum;
                }
                poly_sum += product_sum * coefficient;
            }
            sums.push(poly_sum);
        }
        Ok(sums)
    }

    pub fn deep_copy(&self) -> Self {
        let mut copy = VirtualPolynomial {
            aux_info: self.aux_info.clone(),
            products: self.products.clone(),
            flattened_ml_extensions: self
                .flattened_ml_extensions
                .iter()
                .map(|x| Arc::new(DenseMultilinearExtension::clone(x)))
                .collect(),
            raw_pointers_lookup_table: HashMap::new(),
        };
        for (idx, mle) in copy.flattened_ml_extensions.iter().enumerate() {
            let mle_ptr: *const DenseMultilinearExtension<F> = Arc::as_ptr(mle);
            copy.raw_pointers_lookup_table.insert(mle_ptr, idx);
        }
        copy
    }
    
    /// Multiply every product coefficient by `scalar`, effectively scaling
    /// the entire polynomial: `P(x) → scalar · P(x)`.
    pub fn scale_by_scalar(&mut self, scalar: &F) {
        for (coeff, _) in &mut self.products {
            *coeff *= scalar;
        }
    }

    /// Sample a random virtual polynomial that evaluates to zero everywhere
    /// over the boolean hypercube.
    pub fn rand_zero<R: RngCore>(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
        rng: &mut R,
    ) -> Result<Self, ArithErrors> {
        let mut poly = VirtualPolynomial::new(nv);
        for _ in 0..num_products {
            let num_multiplicands =
                rng.gen_range(num_multiplicands_range.0..num_multiplicands_range.1);
            let product = random_zero_mle_list(nv, num_multiplicands, rng);
            let coefficient = F::rand(rng);
            poly.add_mle_list(product.into_iter(), coefficient)?;
        }

        Ok(poly)
    }

    // Input poly f(x) and a random vector r, output
    //      \hat f(x) = \sum_{x_i \in eval_x} f(x_i) eq(x, r)
    // where
    //      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
    //
    // This function is used in ZeroCheck.
    pub fn build_f_hat(&self, r: &[F]) -> Result<Self, ArithErrors> {

        if self.aux_info.num_variables != r.len() {
            return Err(ArithErrors::InvalidParameters(format!(
                "r.len() is different from number of variables: {} vs {}",
                r.len(),
                self.aux_info.num_variables
            )));
        }

        let eq_x_r = build_eq_x_r(r)?;
        let mut res = self.clone();
        res.mul_by_mle(eq_x_r, F::one())?;

        Ok(res)
    }

    /// Print out the evaluation map for testing. Panic if the num_vars > 5.
    pub fn print_evals(&self) {
        if self.aux_info.num_variables > 5 {
            panic!("this function is used for testing only. cannot print more than 5 num_vars")
        }
        for i in 0..1 << self.aux_info.num_variables {
            let point = bit_decompose(i, self.aux_info.num_variables);
            let point_fr: Vec<F> = point.iter().map(|&x| F::from(x)).collect();
            println!("{} {}", i, self.evaluate(point_fr.as_ref()).unwrap())
        }
        println!()
    }

    /// Split this VirtualPolynomial into 2^n sub-polynomials by fixing the last n variables
    /// of each MLE to all binary assignments.
    ///
    /// For a VirtualPolynomial with m variables, this produces 2^n VirtualPolynomials:
    /// - split[0]: all MLEs evaluated at f(x_1, ..., x_{m-n}, 0, 0, ..., 0)
    /// - split[1]: all MLEs evaluated at f(x_1, ..., x_{m-n}, 1, 0, ..., 0)
    /// - split[2]: all MLEs evaluated at f(x_1, ..., x_{m-n}, 0, 1, ..., 0)
    /// - ...
    /// - split[2^n - 1]: all MLEs evaluated at f(x_1, ..., x_{m-n}, 1, 1, ..., 1)
    ///
    /// # Memory Layout Assumption
    /// This function relies on the little-endian bit indexing used by `DenseMultilinearExtension`.
    /// The evaluation at index `i` corresponds to the point where binary digits of `i` give the
    /// variable assignments in order (x_1, x_2, ..., x_m). Thus, fixing the last n variables to
    /// assignment `j` corresponds to taking the j-th contiguous chunk of size 2^(m-n) from each MLE.
    ///
    /// Automatically selects parallel or sequential execution based on data size.
    ///
    /// # Arguments
    /// * `n` - Number of last variables to fix (number of bits)
    ///
    /// # Returns
    /// A vector of 2^n VirtualPolynomials, each with (m-n) variables
    ///
    /// # Panics
    /// Panics if n > num_variables
    pub fn split_by_last_variables(&self, n: usize) -> Vec<Self> {
        use crate::split_by_last_variables;

        let m = self.aux_info.num_variables;
        assert!(n <= m, "n ({}) cannot exceed num_variables ({})", n, m);

        if n == 0 {
            return vec![self.clone()];
        }

        let num_splits = 1 << n;
        let new_num_vars = m - n;

        // Threshold for parallel MLE iteration (based on number of MLEs and their size)
        const PARALLEL_MLE_THRESHOLD: usize = 4;

        // Split each MLE into 2^n chunks
        // Use parallel iteration over MLEs if there are enough of them
        let split_mles: Vec<Vec<Arc<DenseMultilinearExtension<F>>>> =
            if self.flattened_ml_extensions.len() >= PARALLEL_MLE_THRESHOLD {
                self.flattened_ml_extensions
                    .par_iter()
                    .map(|mle| {
                        split_by_last_variables(mle, n)
                            .into_iter()
                            .map(Arc::new)
                            .collect()
                    })
                    .collect()
            } else {
                self.flattened_ml_extensions
                    .iter()
                    .map(|mle| {
                        split_by_last_variables(mle, n)
                            .into_iter()
                            .map(Arc::new)
                            .collect()
                    })
                    .collect()
            };

        // Build 2^n VirtualPolynomials
        // Note: Sequential iteration here because VirtualPolynomial contains
        // raw pointers in its HashMap which are not Send-safe.
        (0..num_splits)
            .map(|split_idx| {
                // Collect the split_idx-th chunk of each MLE
                let new_mles: Vec<Arc<DenseMultilinearExtension<F>>> = split_mles
                    .iter()
                    .map(|mle_splits| mle_splits[split_idx].clone())
                    .collect();

                // Build lookup table for new Arc pointers
                let mut lookup_table = HashMap::new();
                for (idx, mle) in new_mles.iter().enumerate() {
                    let ptr: *const DenseMultilinearExtension<F> = Arc::as_ptr(mle);
                    lookup_table.insert(ptr, idx);
                }

                VirtualPolynomial {
                    aux_info: VPAuxInfo {
                        max_degree: self.aux_info.max_degree,
                        num_variables: new_num_vars,
                        phantom: PhantomData,
                    },
                    products: self.products.clone(),
                    flattened_ml_extensions: new_mles,
                    raw_pointers_lookup_table: lookup_table,
                }
            })
            .collect()
    }
}

/// Evaluate eq polynomial.
pub fn eq_eval<F: PrimeField>(x: &[F], y: &[F]) -> Result<F, ArithErrors> {
    if x.len() != y.len() {
        return Err(ArithErrors::InvalidParameters(
            "x and y have different length".to_string(),
        ));
    }
    let mut res = F::one();
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let xi_yi = xi * yi;
        res *= xi_yi + xi_yi - xi - yi + F::one();
    }
    Ok(res)
}

/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r<F: PrimeField>(
    r: &[F],
) -> Result<Arc<DenseMultilinearExtension<F>>, ArithErrors> {
    let evals = build_eq_x_r_vec(r)?;
    let mle = DenseMultilinearExtension::from_evaluations_vec(r.len(), evals);

    Ok(Arc::new(mle))
}

/// Build eq(x, r) polynomial scaled by a coefficient, as an MLE.
///
/// Returns coeff * eq(x, r) as a multilinear extension.
pub fn build_eq_x_r_with_coeff<F: PrimeField>(
    r: &[F],
    coeff: &F,
) -> Result<Arc<DenseMultilinearExtension<F>>, ArithErrors> {
    let evals = build_eq_x_r_vec(r)?;
    let scaled_evals: Vec<F> = evals.into_iter().map(|e| e * coeff).collect();
    let mle = DenseMultilinearExtension::from_evaluations_vec(r.len(), scaled_evals);
    Ok(Arc::new(mle))
}

/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r_vec<F: PrimeField>(r: &[F]) -> Result<Vec<F>, ArithErrors> {
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations

    let mut eval = Vec::new();
    build_eq_x_r_helper(r, &mut eval)?;

    Ok(eval)
}

/// A helper function to build eq(x, r) recursively.
/// This function takes `r.len()` steps, and for each step it requires a maximum
/// `r.len()-1` multiplications.
fn build_eq_x_r_helper<F: PrimeField>(r: &[F], buf: &mut Vec<F>) -> Result<(), ArithErrors> {
    if r.is_empty() {
        return Err(ArithErrors::InvalidParameters("r length is 0".to_string()));
    } else if r.len() == 1 {
        // initializing the buffer with [1-r_0, r_0]
        buf.push(F::one() - r[0]);
        buf.push(r[0]);
    } else {
        build_eq_x_r_helper(&r[1..], buf)?;

        // suppose at the previous step we received [b_1, ..., b_k]
        // for the current step we will need
        // if x_0 = 0:   (1-r0) * [b_1, ..., b_k]
        // if x_0 = 1:   r0 * [b_1, ..., b_k]
        // let mut res = vec![];
        // for &b_i in buf.iter() {
        //     let tmp = r[0] * b_i;
        //     res.push(b_i - tmp);
        //     res.push(tmp);
        // }
        // *buf = res;

        let mut res = vec![F::zero(); buf.len() << 1];
        res.par_iter_mut().enumerate().for_each(|(i, val)| {
            let bi = buf[i >> 1];
            let tmp = r[0] * bi;
            if i & 1 == 0 {
                *val = bi - tmp;
            } else {
                *val = tmp;
            }
        });
        *buf = res;
    }

    Ok(())
}

/// Decompose an integer into a binary vector in little endian.
pub fn bit_decompose(input: u64, num_var: usize) -> Vec<bool> {
    let mut res = Vec::with_capacity(num_var);
    let mut i = input;
    for _ in 0..num_var {
        res.push(i & 1 == 1);
        i >>= 1;
    }
    res
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_virtual_polynomial_additions() -> Result<(), ArithErrors> {
        let mut rng = test_rng();
        for nv in 2..5 {
            for num_products in 2..5 {
                let base: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

                let (a, _a_sum) =
                    VirtualPolynomial::<Fr>::rand(nv, (2, 3), num_products, &mut rng)?;
                let (b, _b_sum) =
                    VirtualPolynomial::<Fr>::rand(nv, (2, 3), num_products, &mut rng)?;
                let c = &a + &b;

                assert_eq!(
                    a.evaluate(base.as_ref())? + b.evaluate(base.as_ref())?,
                    c.evaluate(base.as_ref())?
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_virtual_polynomial_mul_by_mle() -> Result<(), ArithErrors> {
        let mut rng = test_rng();
        for nv in 2..5 {
            for num_products in 2..5 {
                let base: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

                let (a, _a_sum) =
                    VirtualPolynomial::<Fr>::rand(nv, (2, 3), num_products, &mut rng)?;
                let (b, _b_sum) = random_mle_list(nv, 1, &mut rng);
                let b_mle = b[0].clone();
                let coeff = Fr::rand(&mut rng);
                let b_vp = VirtualPolynomial::new_from_mle(&b_mle, coeff);

                let mut c = a.clone();

                c.mul_by_mle(b_mle, coeff)?;

                assert_eq!(
                    a.evaluate(base.as_ref())? * b_vp.evaluate(base.as_ref())?,
                    c.evaluate(base.as_ref())?
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_eq_xr() {
        let mut rng = test_rng();
        for nv in 4..10 {
            let r: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
            let eq_x_r = build_eq_x_r(r.as_ref()).unwrap();
            let eq_x_r2 = build_eq_x_r_for_test(r.as_ref());
            assert_eq!(eq_x_r, eq_x_r2);
        }
    }

    /// Naive method to build eq(x, r).
    /// Only used for testing purpose.
    // Evaluate
    //      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
    // over r, which is
    //      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
    fn build_eq_x_r_for_test<F: PrimeField>(r: &[F]) -> Arc<DenseMultilinearExtension<F>> {
        // we build eq(x,r) from its evaluations
        // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
        // for example, with num_vars = 4, x is a binary vector of 4, then
        //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
        //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
        //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
        //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
        //  ....
        //  1 1 1 1 -> r0       * r1        * r2        * r3
        // we will need 2^num_var evaluations

        // First, we build array for {1 - r_i}
        let one_minus_r: Vec<F> = r.iter().map(|ri| F::one() - ri).collect();

        let num_var = r.len();
        let mut eval = vec![];

        for i in 0..1 << num_var {
            let mut current_eval = F::one();
            let bit_sequence = bit_decompose(i, num_var);

            for (&bit, (ri, one_minus_ri)) in
                bit_sequence.iter().zip(r.iter().zip(one_minus_r.iter()))
            {
                current_eval *= if bit { *ri } else { *one_minus_ri };
            }
            eval.push(current_eval);
        }

        let mle = DenseMultilinearExtension::from_evaluations_vec(num_var, eval);

        Arc::new(mle)
    }

    /// Test split_by_last_variables produces correct evaluations for VirtualPolynomial
    #[test]
    fn test_virtual_polynomial_split_by_last_variables() -> Result<(), ArithErrors> {
        let mut rng = test_rng();

        for m in 4..7 {
            for n in 0..=3 {
                for num_products in 1..3 {
                    let (vp, _sum) =
                        VirtualPolynomial::<Fr>::rand(m, (1, 3), num_products, &mut rng)?;

                    let splits = vp.split_by_last_variables(n);

                    assert_eq!(splits.len(), 1 << n);
                    for split in &splits {
                        assert_eq!(split.aux_info.num_variables, m - n);
                        assert_eq!(split.aux_info.max_degree, vp.aux_info.max_degree);
                    }

                    // Verify evaluation consistency
                    let point: Vec<Fr> = (0..(m - n)).map(|_| Fr::rand(&mut rng)).collect();

                    for split_idx in 0..(1 << n) {
                        // Build full point by appending binary assignment for split_idx
                        let mut full_point = point.clone();
                        for bit in 0..n {
                            let bit_val = ((split_idx >> bit) & 1) as u64;
                            full_point.push(Fr::from(bit_val));
                        }

                        let expected = vp.evaluate(&full_point)?;
                        let actual = splits[split_idx].evaluate(&point)?;

                        assert_eq!(
                            expected, actual,
                            "Mismatch at m={}, n={}, split_idx={}",
                            m, n, split_idx
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Test edge case: n = 0 returns a clone
    #[test]
    fn test_virtual_polynomial_split_n_zero() -> Result<(), ArithErrors> {
        let mut rng = test_rng();
        let (vp, _sum) = VirtualPolynomial::<Fr>::rand(5, (2, 3), 2, &mut rng)?;

        let splits = vp.split_by_last_variables(0);

        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].aux_info.num_variables, 5);

        // Verify same evaluation at random point
        let point: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
        assert_eq!(vp.evaluate(&point)?, splits[0].evaluate(&point)?);

        Ok(())
    }

    /// Test edge case: n = num_variables returns 2^m constant polynomials
    #[test]
    fn test_virtual_polynomial_split_n_equals_num_vars() -> Result<(), ArithErrors> {
        let mut rng = test_rng();
        let nv = 4;
        let (vp, _sum) = VirtualPolynomial::<Fr>::rand(nv, (1, 2), 2, &mut rng)?;

        let splits = vp.split_by_last_variables(nv);

        assert_eq!(splits.len(), 1 << nv);
        for split in &splits {
            assert_eq!(split.aux_info.num_variables, 0);
        }

        // Verify each split evaluates to the original at the corresponding binary point
        for split_idx in 0..(1 << nv) {
            let mut binary_point = Vec::with_capacity(nv);
            for bit in 0..nv {
                let bit_val = ((split_idx >> bit) & 1) as u64;
                binary_point.push(Fr::from(bit_val));
            }

            let expected = vp.evaluate(&binary_point)?;
            let actual = splits[split_idx].evaluate(&[])?;

            assert_eq!(expected, actual, "Mismatch at split_idx={}", split_idx);
        }

        Ok(())
    }
}
