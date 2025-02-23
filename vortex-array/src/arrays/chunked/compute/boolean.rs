use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{binary_boolean, slice, BinaryBooleanFn, BinaryOperator};
use crate::{Array, ArrayRef, IntoArray};

impl BinaryBooleanFn<&ChunkedArray> for ChunkedEncoding {
    fn binary_boolean(
        &self,
        lhs: &ChunkedArray,
        rhs: &dyn Array,
        op: BinaryOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        let mut idx = 0;
        let mut chunks = Vec::with_capacity(lhs.nchunks());

        for chunk in lhs.non_empty_chunks() {
            let sliced = slice(rhs, idx, idx + chunk.len())?;
            let result = binary_boolean(chunk, &sliced, op)?;
            chunks.push(result);
            idx += chunk.len();
        }

        let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
        let dtype = DType::Bool(nullable.into());
        Ok(Some(ChunkedArray::try_new(chunks, dtype)?.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability};

    use crate::array::Array;
    use crate::arrays::{BoolArray, BooleanBuffer, ChunkedArray};
    use crate::canonical::ToCanonical;
    use crate::compute::{binary_boolean, BinaryOperator};
    use crate::IntoArray;

    #[test]
    fn test_bin_bool_chunked() {
        let arr0 = BoolArray::from_iter(vec![true, false]).to_array();
        let arr1 = BoolArray::from_iter(vec![false, false, true]).to_array();
        let chunked1 =
            ChunkedArray::try_new(vec![arr0, arr1], DType::Bool(Nullability::NonNullable)).unwrap();

        let arr2 = BoolArray::from_iter(vec![Some(false), Some(true)]).to_array();
        let arr3 = BoolArray::from_iter(vec![Some(false), None, Some(false)]).to_array();
        let chunked2 =
            ChunkedArray::try_new(vec![arr2, arr3], DType::Bool(Nullability::Nullable)).unwrap();

        let result = binary_boolean(&chunked1, &chunked2, BinaryOperator::Or)
            .unwrap()
            .to_bool()
            .unwrap();
        assert_eq!(
            result.boolean_buffer(),
            &BooleanBuffer::from_iter([true, true, false, false, true])
        );
    }
}
