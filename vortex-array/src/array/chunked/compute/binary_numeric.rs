use vortex_error::VortexResult;
use vortex_scalar::BinaryNumericOperator;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::builders::{ArrayBuilder, BoolBuilder};
use crate::compute::{binary_numeric, slice, BinaryNumericFn};
use crate::Array;

impl BinaryNumericFn<ChunkedArray> for ChunkedEncoding {
    fn binary_numeric(
        &self,
        array: &ChunkedArray,
        rhs: &Array,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<Array>> {
        let mut start = 0;

        let mut builder = BoolBuilder::with_capacity(array.dtype().nullability(), array.len());

        for chunk in array.non_empty_chunks() {
            let end = start + chunk.len();
            builder.extend_from_array(binary_numeric(&chunk, &slice(rhs, start, end)?, op)?)?;
            start = end;
        }

        builder.finish().map(Some)
    }
}
