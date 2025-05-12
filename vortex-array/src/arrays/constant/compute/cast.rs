use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::compute::{CastKernel, CastKernelAdapter};
use crate::{ArrayRef, IntoArray, register_kernel};

impl CastKernel for ConstantVTable {
    fn cast(&self, array: &ConstantArray, dtype: &DType) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(array.scalar().cast(dtype)?, array.len()).into_array())
    }
}

register_kernel!(CastKernelAdapter(ConstantVTable).lift());
