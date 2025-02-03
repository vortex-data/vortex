use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::{compare, slice, CompareFn, Operator};
use crate::{Array, IntoArray};

impl CompareFn<ChunkedArray> for ChunkedEncoding {
    fn compare(
        &self,
        lhs: &ChunkedArray,
        rhs: &Array,
        operator: Operator,
    ) -> VortexResult<Option<Array>> {
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
    use crate::array::PrimitiveArray;

    #[test]
    fn empty_compare() {
        let base = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        let chunked =
            ChunkedArray::try_new(vec![base.clone(), base.clone()], base.dtype().clone()).unwrap();
        let chunked_empty = ChunkedArray::try_new(vec![], base.dtype().clone()).unwrap();

        let r = compare(&chunked, &chunked_empty, Operator::Eq).unwrap();

        assert!(r.is_empty());
    }
}
