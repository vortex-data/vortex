use vortex_array::compute::IsConstantFn;
use vortex_error::VortexResult;

use crate::{BitPackedArray, BitPackedEncoding};

impl IsConstantFn<&BitPackedArray> for BitPackedEncoding {
    fn is_constant(&self, _array: &BitPackedArray) -> VortexResult<Option<bool>> {
        Ok(None)
    }
}
