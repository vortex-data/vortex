use vortex_array::compute::IsConstantFn;
use vortex_error::VortexResult;

use crate::{ALPRDArray, ALPRDEncoding};

impl IsConstantFn<&ALPRDArray> for ALPRDEncoding {
    fn is_constant(&self, _array: &ALPRDArray) -> VortexResult<Option<bool>> {
        Ok(None)
    }
}
