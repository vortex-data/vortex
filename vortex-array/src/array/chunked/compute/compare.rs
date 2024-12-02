use vortex_dtype::{DType, Nullability};
use vortex_error::VortexResult;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::{compare, slice, CompareFn, Operator};
use crate::{ArrayData, IntoArrayData};

impl CompareFn<ChunkedArray> for ChunkedEncoding {
    fn compare(
        &self,
        lhs: &ChunkedArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        let mut idx = 0;
        let mut compare_chunks = Vec::with_capacity(lhs.nchunks());

        for chunk in lhs.chunks() {
            let sliced = slice(rhs, idx, idx + chunk.len())?;
            let cmp_result = compare(&chunk, &sliced, operator)?;
            compare_chunks.push(cmp_result);

            idx += chunk.len();
        }

        Ok(Some(
            ChunkedArray::try_new(compare_chunks, DType::Bool(Nullability::Nullable))?.into_array(),
        ))
    }
}
