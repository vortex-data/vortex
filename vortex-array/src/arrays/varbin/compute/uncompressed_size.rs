use vortex_error::VortexResult;

use crate::arrays::{VarBinArray, VarBinEncoding};
use crate::compute::{UncompressedSizeFn, uncompressed_size};

impl UncompressedSizeFn<&VarBinArray> for VarBinEncoding {
    fn uncompressed_size(&self, array: &VarBinArray) -> VortexResult<usize> {
        let offsets = uncompressed_size(array.offsets().as_ref())?;

        Ok(offsets + array.bytes().len() + array.validity().uncompressed_size())
    }
}
