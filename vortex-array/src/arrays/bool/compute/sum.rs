// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::compute::SumKernel;
use crate::compute::SumKernelAdapter;
use crate::dtype::Nullability;
use crate::register_kernel;
use crate::scalar::Scalar;

impl SumKernel for BoolVTable {
    fn sum(&self, array: &BoolArray, accumulator: &Scalar) -> VortexResult<Scalar> {
        let true_count: Option<u64> = match array.validity_mask()?.bit_buffer() {
            AllOr::All => {
                // All-valid
                Some(array.to_bit_buffer().true_count() as u64)
            }
            AllOr::None => {
                // All-invalid
                unreachable!("All-invalid boolean array should have been handled by entry-point")
            }
            AllOr::Some(validity_mask) => {
                Some(array.to_bit_buffer().bitand(validity_mask).true_count() as u64)
            }
        };

        let acc_value = accumulator
            .as_primitive()
            .as_::<u64>()
            .vortex_expect("cannot be null");
        let result = true_count.and_then(|tc| acc_value.checked_add(tc));
        Ok(match result {
            Some(v) => Scalar::primitive(v, Nullability::Nullable),
            None => Scalar::null_native::<u64>(),
        })
    }
}

register_kernel!(SumKernelAdapter(BoolVTable).lift());
