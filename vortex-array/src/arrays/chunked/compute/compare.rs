// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::builders::ArrayBuilder;
use crate::builders::BoolBuilder;
use crate::compute::CompareKernel;
use crate::compute::CompareKernelAdapter;
use crate::compute::Operator;
use crate::compute::compare;
use crate::register_kernel;

impl CompareKernel for ChunkedVTable {
    fn compare(
        &self,
        lhs: &ChunkedArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        let mut idx = 0;

        let mut bool_builder = BoolBuilder::with_capacity(
            // nullable <= non-nullable
            (lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into(),
            lhs.len(),
        );

        for chunk in lhs.non_empty_chunks() {
            let sliced = rhs.slice(idx..idx + chunk.len())?;
            let cmp_result = compare(chunk, &sliced, operator)?;

            bool_builder.extend_from_array(&cmp_result);
            idx += chunk.len();
        }

        Ok(Some(bool_builder.finish()))
    }
}

register_kernel!(CompareKernelAdapter(ChunkedVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;

    use super::*;
    use crate::IntoArray;

    #[test]
    fn empty_compare() {
        let base = Buffer::<u32>::empty().into_array();
        let chunked =
            ChunkedArray::try_new(vec![base.clone(), base.clone()], base.dtype().clone()).unwrap();
        let chunked_empty = ChunkedArray::try_new(vec![], base.dtype().clone()).unwrap();

        let r = compare(chunked.as_ref(), chunked_empty.as_ref(), Operator::Eq).unwrap();

        assert!(r.is_empty());
    }
}
