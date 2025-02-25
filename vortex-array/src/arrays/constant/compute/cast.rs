use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{ConstantArray, ConstantEncoding};
use crate::compute::CastFn;
use crate::{Array, ArrayRef};

impl CastFn<&ConstantArray> for ConstantEncoding {
    fn cast(&self, array: &ConstantArray, dtype: &DType) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(array.scalar().cast(dtype)?, array.len()).into_array())
    }
}
