use std::ops::BitAnd;

use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::{SumKernel, SumKernelAdapter};
use crate::{Array, register_kernel};

impl SumKernel for BoolEncoding {
    fn sum(&self, array: &BoolArray) -> VortexResult<Scalar> {
        let true_count: Option<u64> = match array.validity_mask()?.boolean_buffer() {
            AllOr::All => {
                // All-valid
                Some(array.boolean_buffer().count_set_bits() as u64)
            }
            AllOr::None => {
                // All-invalid
                unreachable!("All-invalid boolean array should have been handled by entry-point")
            }
            AllOr::Some(validity_mask) => Some(
                array
                    .boolean_buffer()
                    .bitand(validity_mask)
                    .count_set_bits() as u64,
            ),
        };
        Ok(Scalar::from(true_count))
    }
}

register_kernel!(SumKernelAdapter(BoolEncoding).lift());
