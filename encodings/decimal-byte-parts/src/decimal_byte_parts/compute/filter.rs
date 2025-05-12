use itertools::Itertools;
use vortex_array::compute::{FilterKernel, FilterKernelAdapter, filter};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{DecimalBytePartsArray, DecimalBytePartsEncoding};

impl FilterKernel for DecimalBytePartsEncoding {
    fn filter(&self, array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef> {
        DecimalBytePartsArray::try_new(
            filter(&array.msp, mask)?,
            array
                .lower_parts
                .iter()
                .map(|p| filter(p, mask))
                .try_collect()?,
            *array.decimal_dtype(),
        )
        .map(|d| d.to_array())
    }
}

register_kernel!(FilterKernelAdapter(DecimalBytePartsEncoding).lift());
