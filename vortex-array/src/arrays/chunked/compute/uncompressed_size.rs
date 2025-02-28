use std::mem::size_of_val;

use vortex_error::VortexResult;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{UncompressedSizeFn, uncompressed_size};

impl UncompressedSizeFn<&ChunkedArray> for ChunkedEncoding {
    fn uncompressed_size(&self, array: &ChunkedArray) -> VortexResult<usize> {
        let mut sum = size_of_val(array.chunk_offsets());

        for chunk in array.chunks().iter() {
            sum += uncompressed_size(chunk)?;
        }

        Ok(sum)
    }
}
