use vortex_array::compute::{BetweenKernel, BetweenKernelAdapter, BetweenOptions, between};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;

use crate::{BitPackedArray, BitPackedVTable};

impl BetweenKernel for BitPackedVTable {
    fn between(
        &self,
        array: &BitPackedArray,
        lower: &dyn Array,
        upper: &dyn Array,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        if !lower.is_constant() || !upper.is_constant() {
            return Ok(None);
        };

        between(
            &array.clone().to_canonical()?.into_array(),
            lower,
            upper,
            options,
        )
        .map(Some)
    }
}

register_kernel!(BetweenKernelAdapter(BitPackedVTable).lift());
