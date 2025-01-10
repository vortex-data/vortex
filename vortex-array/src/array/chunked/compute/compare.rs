use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::{compare, slice, CompareFn, Operator};
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl CompareFn<ChunkedArray> for ChunkedEncoding {
    fn compare(
        &self,
        lhs: &ChunkedArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        let mut idx = 0;
        let mut compare_chunks = Vec::with_capacity(lhs.nchunks());

        for chunk in lhs.chunks().filter(|c| !c.is_empty()) {
            let sliced = slice(rhs, idx, idx + chunk.len())?;
            let cmp_result = compare(&chunk, &sliced, operator)?;

            compare_chunks.push(cmp_result);
            idx += chunk.len();
        }

        Ok(Some(
            ChunkedArray::try_new(
                compare_chunks,
                DType::Bool((lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into()),
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::array::StructArray;
    use crate::validity::Validity;

    #[test]
    fn empty_compare() {
        let base = StructArray::try_new([].into(), [].into(), 0, Validity::NonNullable)
            .unwrap()
            .into_array();
        let chunked =
            ChunkedArray::try_new(vec![base.clone(), base.clone()], base.dtype().clone()).unwrap();
        let chunked_empty = ChunkedArray::try_new(vec![], base.dtype().clone()).unwrap();

        let r = compare(&chunked, &chunked_empty, Operator::Eq).unwrap();
        assert!(r.is_empty());
    }
}
