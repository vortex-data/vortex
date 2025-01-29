use vortex_error::VortexResult;
use vortex_scalar::BinaryNumericOperator;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::{binary_numeric, slice, BinaryNumericFn};
use crate::{ArrayData, IntoArrayData};

impl BinaryNumericFn<ChunkedArray> for ChunkedEncoding {
    fn binary_numeric(
        &self,
        array: &ChunkedArray,
        rhs: &ArrayData,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<ArrayData>> {
        let mut start = 0;

        let mut new_chunks = Vec::with_capacity(array.nchunks());
        for chunk in array.chunks() {
            let end = start + chunk.len();
            new_chunks.push(binary_numeric(&chunk, &slice(rhs, start, end)?, op)?);
            start = end;
        }

        ChunkedArray::try_new(new_chunks, array.dtype().clone())
            .map(IntoArrayData::into_array)
            .map(Some)
    }
}
