use vortex_error::VortexResult;

use crate::arrays::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::UncompressedSizeFn;
use crate::nbytes::NBytes;

impl UncompressedSizeFn<&PrimitiveArray> for PrimitiveEncoding {
    fn uncompressed_size(&self, array: &PrimitiveArray) -> VortexResult<usize> {
        Ok(array.nbytes())
    }
}
