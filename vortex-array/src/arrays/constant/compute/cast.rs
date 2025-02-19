use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::array::{ConstantArray, ConstantEncoding};
use crate::compute::CastFn;
use crate::{Array, IntoArray};

impl CastFn<ConstantArray> for ConstantEncoding {
    fn cast(&self, array: &ConstantArray, dtype: &DType) -> VortexResult<Array> {
        Ok(ConstantArray::new(array.scalar().cast(dtype)?, array.len()).into_array())
    }
}
