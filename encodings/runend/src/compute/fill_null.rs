use vortex_array::compute::{FillNullKernel, FillNullKernelAdapter, fill_null};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{RunEndArray, RunEndVTable};

impl FillNullKernel for RunEndVTable {
    fn fill_null(&self, array: &RunEndArray, fill_value: &Scalar) -> VortexResult<ArrayRef> {
        Ok(RunEndArray::with_offset_and_length(
            array.ends().clone(),
            fill_null(array.values(), fill_value)?,
            array.offset(),
            array.len(),
        )?
        .into_array())
    }
}

register_kernel!(FillNullKernelAdapter(RunEndVTable).lift());
