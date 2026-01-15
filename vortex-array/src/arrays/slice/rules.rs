// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::slice::SliceArray;
use crate::arrays::slice::SliceVTable;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ReduceRuleSet;

pub(super) const RULES: ReduceRuleSet<SliceVTable> = ReduceRuleSet::new(&[
    &SliceSliceRule,
    // Try the generic VTable::slice for all encodings
    &SliceVTableRule,
]);

/// Reduce rule for Slice(Slice(child)) -> Slice(child) with combined ranges
#[derive(Debug)]
struct SliceSliceRule;

impl ArrayReduceRule<SliceVTable> for SliceSliceRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(inner_slice) = array.child().as_opt::<SliceVTable>() else {
            return Ok(None);
        };

        // Combine the ranges: outer range is relative to inner slice
        let outer_range = array.slice_range();
        let inner_range = inner_slice.slice_range();

        let combined_start = inner_range.start + outer_range.start;
        let combined_end = inner_range.start + outer_range.end;

        Ok(Some(
            SliceArray::new(inner_slice.child().clone(), combined_start..combined_end).into_array(),
        ))
    }
}

/// Generic reduce rule that calls VTable::slice on the child.
/// This allows all encodings to implement their own slice logic.
#[derive(Debug)]
struct SliceVTableRule;

impl ArrayReduceRule<SliceVTable> for SliceVTableRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        // Try the child's VTable::slice implementation
        array
            .child()
            .encoding()
            .slice(array.child(), array.slice_range().clone())
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::SliceSliceRule;
    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::SliceArray;
    use crate::arrays::SliceVTable;
    use crate::assert_arrays_eq;
    use crate::optimizer::rules::ArrayReduceRule;

    #[test]
    fn test_slice_slice() -> VortexResult<()> {
        // Slice(1..4, Slice(2..8, base)) combines to Slice(3..6, base)
        let arr = PrimitiveArray::from_iter(0i32..10).into_array();
        let inner_slice = SliceArray::new(arr, 2..8).into_array();
        let outer_slice = SliceArray::new(inner_slice, 1..4);

        let result = SliceSliceRule.reduce(&outer_slice)?;
        assert!(result.is_some());

        let reduced = result.unwrap();
        assert_eq!(reduced.as_::<SliceVTable>().slice_range(), &(3..6));
        assert_arrays_eq!(reduced, PrimitiveArray::from_iter([3i32, 4, 5]));

        Ok(())
    }
}
