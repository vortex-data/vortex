use vortex_error::VortexResult;
use vortex_scalar::NumericOperator;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{NumericKernel, NumericKernelAdapter, numeric, slice};
use crate::{Array, ArrayRef, register_kernel};

impl NumericKernel for ChunkedEncoding {
    fn numeric(
        &self,
        array: &ChunkedArray,
        rhs: &dyn Array,
        op: NumericOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        let mut start = 0;

        let mut new_chunks = Vec::with_capacity(array.nchunks());
        for chunk in array.non_empty_chunks() {
            let end = start + chunk.len();
            new_chunks.push(numeric(chunk, &slice(rhs, start, end)?, op)?);
            start = end;
        }

        ChunkedArray::try_new(new_chunks, array.dtype().clone())
            .map(|c| c.into_array())
            .map(Some)
    }
}

register_kernel!(NumericKernelAdapter(ChunkedEncoding).lift());
