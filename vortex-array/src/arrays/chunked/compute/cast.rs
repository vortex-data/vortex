use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{CastKernel, CastKernelAdapter, cast};
use crate::{Array, ArrayRef, register_kernel};

impl CastKernel for ChunkedEncoding {
    fn cast(&self, array: &ChunkedArray, dtype: &DType) -> VortexResult<ArrayRef> {
        let mut cast_chunks = Vec::new();
        for chunk in array.chunks() {
            cast_chunks.push(cast(chunk, dtype)?);
        }

        Ok(ChunkedArray::new_unchecked(cast_chunks, dtype.clone()).into_array())
    }
}

register_kernel!(CastKernelAdapter(ChunkedEncoding).lift());
