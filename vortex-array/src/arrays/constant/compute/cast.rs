use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{ConstantArray, ConstantEncoding};
use crate::compute::{CastKernel, CastKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl CastKernel for ConstantEncoding {
    fn cast(&self, array: &ConstantArray, dtype: &DType) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(array.scalar().cast(dtype)?, array.len()).into_array())
    }
}

register_kernel!(CastKernelAdapter(ConstantEncoding).lift());
