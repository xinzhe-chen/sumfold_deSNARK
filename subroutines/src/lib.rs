// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

#![allow(clippy::non_canonical_clone_impl)] // using `derivative`
#![allow(clippy::default_constructed_unit_structs)] // protocol structs use PhantomData in many archived code paths
#![allow(clippy::needless_range_loop)] // indexed loops keep the algebraic layout explicit
#![allow(clippy::type_complexity)] // proof and transcript tuples are part of the protocol surface
#![allow(clippy::too_many_arguments)] // some prover helpers mirror protocol state directly
#![allow(non_snake_case)]

pub mod pcs;
pub mod poly_iop;

pub use pcs::prelude::*;
pub use poly_iop::prelude::*;
