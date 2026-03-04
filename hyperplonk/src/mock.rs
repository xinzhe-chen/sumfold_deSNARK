// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use arithmetic::identity_permutation;
use ark_ff::PrimeField;
use ark_std::{log2, rand::RngCore, test_rng};

use crate::{
    custom_gate::CustomizedGates,
    selectors::SelectorColumn,
    structs::{HyperPlonkIndex, HyperPlonkParams},
    witness::WitnessColumn,
};

#[derive(Clone)]
pub struct MockCircuit<F: PrimeField> {
    pub public_inputs: Vec<F>,
    pub witnesses: Vec<WitnessColumn<F>>,
    pub index: HyperPlonkIndex<F>,
}

impl<F: PrimeField> MockCircuit<F> {
    /// Number of variables in a multilinear system
    pub fn num_variables(&self) -> usize {
        self.index.num_variables()
    }

    /// number of selector columns
    pub fn num_selector_columns(&self) -> usize {
        self.index.num_selector_columns()
    }

    /// number of witness columns
    pub fn num_witness_columns(&self) -> usize {
        self.index.num_witness_columns()
    }
}

impl<F: PrimeField> MockCircuit<F> {
    /// Generate a mock plonk circuit for the input constraint size.
    pub fn new(num_constraints: usize, gate: &CustomizedGates) -> MockCircuit<F> {
        Self::new_with_rng(num_constraints, gate, &mut test_rng())
    }

    /// Generate a mock plonk circuit using the provided RNG.
    pub fn new_with_rng(
        num_constraints: usize,
        gate: &CustomizedGates,
        rng: &mut impl RngCore,
    ) -> MockCircuit<F> {
        let num_selectors = gate.num_selector_columns();
        let num_witnesses = gate.num_witness_columns();

        let mut selectors: Vec<SelectorColumn<F>> = vec![SelectorColumn::default(); num_selectors];
        let mut witnesses: Vec<WitnessColumn<F>> = vec![WitnessColumn::default(); num_witnesses];

        // Witness generation matching HyperPianist's MockCircuit::d_new:
        // First 1/4 of constraints get fully random witnesses;
        // remaining 3/4 reuse witnesses from the first quarter.
        let portion_len = num_constraints / 4;

        for cs_counter in 0..num_constraints {
            let mut cur_selectors: Vec<F> =
                (0..(num_selectors - 1)).map(|_| F::rand(rng)).collect();
            let cur_witness: Vec<F> = if cs_counter < portion_len {
                (0..num_witnesses).map(|_| F::rand(rng)).collect()
            } else {
                let row = cs_counter % portion_len;
                (0..num_witnesses)
                    .map(|i| witnesses[i].0[row])
                    .collect()
            };
            let mut last_selector = F::zero();
            for (index, (coeff, q, wit)) in gate.gates.iter().enumerate() {
                if index != num_selectors - 1 {
                    let mut cur_monomial = if *coeff < 0 {
                        -F::from((-coeff) as u64)
                    } else {
                        F::from(*coeff as u64)
                    };
                    cur_monomial = match q {
                        Some(p) => cur_monomial * cur_selectors[*p],
                        None => cur_monomial,
                    };
                    for wit_index in wit.iter() {
                        cur_monomial *= cur_witness[*wit_index];
                    }
                    last_selector += cur_monomial;
                } else {
                    let mut cur_monomial = if *coeff < 0 {
                        -F::from((-coeff) as u64)
                    } else {
                        F::from(*coeff as u64)
                    };
                    for wit_index in wit.iter() {
                        cur_monomial *= cur_witness[*wit_index];
                    }
                    last_selector /= -cur_monomial;
                }
            }
            cur_selectors.push(last_selector);
            for i in 0..num_selectors {
                selectors[i].append(cur_selectors[i]);
            }
            for i in 0..num_witnesses {
                witnesses[i].append(cur_witness[i]);
            }
        }
        let pub_input_len = ark_std::cmp::min(4, num_constraints);
        let public_inputs = witnesses[0].0[0..pub_input_len].to_vec();

        let params = HyperPlonkParams {
            num_constraints,
            num_pub_input: public_inputs.len(),
            gate_func: gate.clone(),
        };

        // Identity permutation: sumfold bypasses HyperPlonk's permutation
        // argument (using SumFold for gate constraints only), and HyperPlonk's
        // standard path requires the permutation to be valid w.r.t. the copy
        // constraint structure. Identity is always valid.
        let permutation = identity_permutation(
            (log2(num_constraints) + log2(num_witnesses)) as usize,
            1,
        );

        let index = HyperPlonkIndex {
            params,
            permutation,
            selectors,
        };

        Self {
            public_inputs,
            witnesses,
            index,
        }
    }

    /// Generate a mock circuit that **reuses pre-built selectors** and only
    /// generates fresh witnesses. The output wire (last witness column used in
    /// a term with a single witness and a selector) is computed as the dependent
    /// variable so the constraint is satisfied.
    ///
    /// For vanilla plonk (`q_L w_1 + q_R w_2 + q_O w_3 + q_M w_1 w_2 + q_C = 0`):
    /// `w_3 = -(q_L w_1 + q_R w_2 + q_M w_1 w_2 + q_C) / q_O`.
    pub fn new_with_shared_selectors(
        num_constraints: usize,
        gate: &CustomizedGates,
        shared_selectors: &[SelectorColumn<F>],
        rng: &mut impl RngCore,
    ) -> MockCircuit<F> {
        let num_selectors = gate.num_selector_columns();
        let num_witnesses = gate.num_witness_columns();
        assert_eq!(shared_selectors.len(), num_selectors);

        // Find the "output" term: a gate term with exactly one witness and a
        // selector.  For vanilla plonk this is term 2: (1, Some(2), [2])
        // i.e. q_O * w_3.
        // We'll solve for that witness column.
        let (out_term_idx, out_sel_idx, out_wit_idx) = gate
            .gates
            .iter()
            .enumerate()
            .find_map(|(idx, (_coeff, q, ws))| {
                if ws.len() == 1 && q.is_some() {
                    // Check that this witness column does NOT appear in any
                    // higher-degree term (to guarantee it's linear and solvable).
                    let wid = ws[0];
                    let appears_elsewhere = gate.gates.iter().enumerate().any(|(i2, (_, _, ws2))| {
                        i2 != idx && ws2.contains(&wid)
                    });
                    if !appears_elsewhere {
                        return Some((idx, q.unwrap(), wid));
                    }
                }
                None
            })
            .expect("gate must have a solvable output term (single-wire, unique column)");

        let mut witnesses: Vec<WitnessColumn<F>> = vec![WitnessColumn::default(); num_witnesses];
        let portion_len = num_constraints / 4;

        for row in 0..num_constraints {
            // Generate random input witnesses (all columns except out_wit_idx)
            let mut cur_witness: Vec<F> = Vec::with_capacity(num_witnesses);
            if row < portion_len {
                for _ in 0..num_witnesses {
                    cur_witness.push(F::rand(rng));
                }
            } else {
                let src_row = row % portion_len;
                for i in 0..num_witnesses {
                    cur_witness.push(witnesses[i].0[src_row]);
                }
            }

            // Accumulate all terms EXCEPT the output term
            let mut rest_sum = F::zero();
            for (idx, (coeff, q, ws)) in gate.gates.iter().enumerate() {
                if idx == out_term_idx {
                    continue;
                }
                let mut term = if *coeff < 0 {
                    -F::from((-coeff) as u64)
                } else {
                    F::from(*coeff as u64)
                };
                if let Some(s) = q {
                    term *= shared_selectors[*s].0[row];
                }
                for &wi in ws {
                    term *= cur_witness[wi];
                }
                rest_sum += term;
            }

            // Solve: coeff_out * q_out[row] * w_out = -rest_sum
            let (out_coeff, _, _) = &gate.gates[out_term_idx];
            let c = if *out_coeff < 0 {
                -F::from((-out_coeff) as u64)
            } else {
                F::from(*out_coeff as u64)
            };
            let q_val = shared_selectors[out_sel_idx].0[row];
            let denom = c * q_val;
            // If denom is zero, output witness is set to zero (term vanishes)
            cur_witness[out_wit_idx] = if denom.is_zero() {
                F::zero()
            } else {
                -rest_sum / denom
            };

            for i in 0..num_witnesses {
                witnesses[i].append(cur_witness[i]);
            }
        }

        let pub_input_len = ark_std::cmp::min(4, num_constraints);
        let public_inputs = witnesses[0].0[0..pub_input_len].to_vec();

        let params = HyperPlonkParams {
            num_constraints,
            num_pub_input: public_inputs.len(),
            gate_func: gate.clone(),
        };

        let permutation = identity_permutation(
            (log2(num_constraints) + log2(num_witnesses)) as usize,
            1,
        );

        let index = HyperPlonkIndex {
            params,
            permutation,
            selectors: shared_selectors.to_vec(),
        };

        Self {
            public_inputs,
            witnesses,
            index,
        }
    }

    pub fn partition_circuit<R: RngCore>(
        num_constraints: usize,
        gate: &CustomizedGates,
        num_partitions: usize,
    ) -> Vec<MockCircuit<F>> {
        assert!(
            num_constraints % num_partitions == 0,
            "约束数量必须能被分割数量整除"
        );

        let constraints_per_partition = num_constraints / num_partitions;

        let sub_circuit = Self::new(constraints_per_partition, gate);

        assert!(sub_circuit.is_satisfied(), "电路不满足约束");

        let sub_circuits: Vec<MockCircuit<F>> =
            (0..num_partitions).map(|_| sub_circuit.clone()).collect();
        sub_circuits
    }

    pub fn is_satisfied(&self) -> bool {
        for current_row in 0..self.num_variables() {
            let mut cur = F::zero();
            for (coeff, q, wit) in self.index.params.gate_func.gates.iter() {
                let mut cur_monomial = if *coeff < 0 {
                    -F::from((-coeff) as u64)
                } else {
                    F::from(*coeff as u64)
                };
                cur_monomial = match q {
                    Some(p) => cur_monomial * self.index.selectors[*p].0[current_row],
                    None => cur_monomial,
                };
                for wit_index in wit.iter() {
                    cur_monomial *= self.witnesses[*wit_index].0[current_row];
                }
                cur += cur_monomial;
            }
            if !cur.is_zero() {
                return false;
            }
        }

        true
    }
}
