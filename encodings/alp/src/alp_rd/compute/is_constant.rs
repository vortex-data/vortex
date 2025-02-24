use vortex_array::compute::IsConstantFn;

use crate::ALPEncoding;

impl IsConstantFn<&ALPArray> for ALPEncoding {
    fn is_constant(&self, array: &ALPArray) -> VortexResult<Option<bool>> {
        Ok(None)
    }
}
