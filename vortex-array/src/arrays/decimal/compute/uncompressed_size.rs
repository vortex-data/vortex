use vortex_error::VortexResult;

use crate::arrays::{DecimalArray, DecimalEncoding};
use crate::compute::UncompressedSizeFn;

impl UncompressedSizeFn<&DecimalArray> for DecimalEncoding {
    fn uncompressed_size(&self, array: &DecimalArray) -> VortexResult<usize> {
        Ok(array.byte_buffer().len() + array.validity().uncompressed_size())
    }
}
