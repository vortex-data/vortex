use vortex_array::compute::{InvertKernel, InvertKernelAdapter, invert};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndVTable};

impl InvertKernel for RunEndVTable {
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

register_kernel!(InvertKernelAdapter(RunEndVTable).lift());
