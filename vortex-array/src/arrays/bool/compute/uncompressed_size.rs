use vortex_error::VortexResult;

use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::UncompressedSizeFn;
use crate::nbytes::NBytes;

impl UncompressedSizeFn<&BoolArray> for BoolEncoding {
    fn uncompressed_size(&self, array: &BoolArray) -> VortexResult<usize> {
        Ok(array.nbytes())
    }
}
