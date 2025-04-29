use itertools::Itertools;
use vortex_error::VortexResult;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{InvertKernel, InvertKernelAdapter, invert};
use crate::{Array, ArrayRef, register_kernel};

impl InvertKernel for ChunkedEncoding {
    fn invert(&self, array: &ChunkedArray) -> VortexResult<ArrayRef> {
        let chunks = array.chunks().iter().map(|c| invert(c)).try_collect()?;
        Ok(ChunkedArray::new_unchecked(chunks, array.dtype().clone()).into_array())
    }
}

register_kernel!(InvertKernelAdapter(ChunkedEncoding).lift());
