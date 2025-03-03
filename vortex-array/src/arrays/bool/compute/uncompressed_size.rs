use vortex_error::VortexResult;

use crate::Array;
use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::UncompressedSizeFn;

impl UncompressedSizeFn<&BoolArray> for BoolEncoding {
    fn uncompressed_size(&self, array: &BoolArray) -> VortexResult<usize> {
        Ok(array.len().div_ceil(8) + array.validity().uncompressed_size())
    }
}
