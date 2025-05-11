use itertools::Itertools;
use vortex_error::VortexResult;

use crate::arrays::{ChunkedArray, ChunkedVTable};
use crate::compute::{invert, InvertKernel, InvertKernelAdapter};
use crate::{register_kernel, ArrayRef, IntoArray};

impl InvertKernel for ChunkedVTable {
    fn invert(&self, array: &ChunkedArray) -> VortexResult<ArrayRef> {
        let chunks = array.chunks().iter().map(|c| invert(c)).try_collect()?;
        Ok(ChunkedArray::new_unchecked(chunks, array.dtype().clone()).into_array())
    }
}

register_kernel!(InvertKernelAdapter(ChunkedVTable).lift());
