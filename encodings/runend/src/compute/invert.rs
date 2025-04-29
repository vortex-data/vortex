use vortex_array::compute::{InvertKernel, InvertKernelAdapter, invert};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

impl InvertKernel for RunEndEncoding {
    fn invert(&self, array: &RunEndArray) -> VortexResult<ArrayRef> {
        RunEndArray::with_offset_and_length(
            array.ends().clone(),
            invert(array.values())?,
            array.len(),
            array.offset(),
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(InvertKernelAdapter(RunEndEncoding).lift());
