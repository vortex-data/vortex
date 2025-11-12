// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::AllOr;
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, BoolVTable};
use crate::compute::{SumKernel, SumKernelAdapter};
use crate::register_kernel;

impl SumKernel for BoolVTable {
    fn sum(&self, array: &BoolArray, accumulator: &Scalar) -> VortexResult<Scalar> {
        let true_count: Option<u64> = match array.validity_mask().bit_buffer() {
            AllOr::All => {
                // All-valid
                Some(array.bit_buffer().true_count() as u64)
            }
            AllOr::None => {
                // All-invalid
                unreachable!("All-invalid boolean array should have been handled by entry-point")
            }
            AllOr::Some(validity_mask) => {
                Some(array.bit_buffer().bitand(validity_mask).true_count() as u64)
            }
        };

        let accumulator = accumulator
            .as_primitive()
            .as_::<u64>()
            .vortex_expect("cannot be null");
        Ok(Scalar::from(
            true_count.and_then(|tc| accumulator.checked_add(tc)),
        ))
    }
}

register_kernel!(SumKernelAdapter(BoolVTable).lift());
