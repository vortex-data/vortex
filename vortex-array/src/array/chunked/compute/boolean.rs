use vortex_dtype::{DType, Nullability};
use vortex_error::VortexResult;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::{and, and_kleene, or, or_kleene, slice, BinaryBooleanFn, BinaryOperator};
use crate::{ArrayData, IntoArrayData};

impl BinaryBooleanFn<ChunkedArray> for ChunkedEncoding {
    fn binary_boolean(
        &self,
        lhs: &ChunkedArray,
        rhs: &ArrayData,
        op: BinaryOperator,
    ) -> VortexResult<Option<ArrayData>> {
        let mut idx = 0;
        let mut chunks = Vec::with_capacity(lhs.nchunks());

        for chunk in lhs.chunks() {
            let sliced = slice(rhs, idx, idx + chunk.len())?;
            let result = match op {
                BinaryOperator::And => and(&chunk, &sliced),
                BinaryOperator::AndKleene => and_kleene(&chunk, &sliced),
                BinaryOperator::Or => or(&chunk, &sliced),
                BinaryOperator::OrKleene => or_kleene(&chunk, &sliced),
            };
            chunks.push(result?);
            idx += chunk.len();
        }

        Ok(Some(
            ChunkedArray::try_new(chunks, DType::Bool(Nullability::Nullable))?.into_array(),
        ))
    }
}
