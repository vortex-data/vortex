use vortex_error::VortexResult;
use vortex_scalar::BinaryNumericOperator;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{BinaryNumericFn, binary_numeric, slice};
use crate::{Array, ArrayRef};

impl BinaryNumericFn<&ChunkedArray> for ChunkedEncoding {
    fn binary_numeric(
        &self,
        array: &ChunkedArray,
        rhs: &dyn Array,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        let mut start = 0;

        let mut new_chunks = Vec::with_capacity(array.nchunks());
        for chunk in array.non_empty_chunks() {
            let end = start + chunk.len();
            new_chunks.push(binary_numeric(chunk, &slice(rhs, start, end)?, op)?);
            start = end;
        }

        ChunkedArray::try_new(new_chunks, array.dtype().clone())
            .map(|c| c.into_array())
            .map(Some)
    }
}
