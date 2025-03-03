use vortex_error::VortexResult;

use crate::arrays::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::UncompressedSizeFn;

impl UncompressedSizeFn<&PrimitiveArray> for PrimitiveEncoding {
    fn uncompressed_size(&self, array: &PrimitiveArray) -> VortexResult<usize> {
        Ok(array.buffer.len() + array.validity().uncompressed_size())
    }
}
