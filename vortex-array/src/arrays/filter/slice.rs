// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Filter;
use crate::arrays::filter::FilterArraySlotsExt;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Filter {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let mask = array.filter_mask();
        // The slice addresses surviving rows. Translate its bounds using the already
        // materialized mask so separate slices do not each execute the whole filtered child.
        let start = mask.rank(range.start);
        let end = mask.rank(range.end - 1) + 1;
        Ok(Some(
            array
                .child()
                .slice(start..end)?
                .filter(mask.slice(start..end))?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Filter;
    use crate::arrays::FilterArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::filter::FilterArraySlotsExt;
    use crate::assert_arrays_eq;

    #[rstest]
    #[case(0..2, vec![Some(1), None], 4)]
    #[case(1..3, vec![None, Some(7)], 4)]
    #[case(2..4, vec![Some(7), Some(9)], 3)]
    fn slice_filter_limits_child(
        #[case] range: Range<usize>,
        #[case] expected: Vec<Option<i32>>,
        #[case] child_len: usize,
        #[values(false, true)] cache_indices: bool,
    ) -> VortexResult<()> {
        let values =
            PrimitiveArray::from_option_iter((0..11).map(|value| (value != 4).then_some(value)))
                .into_array();
        let mask = Mask::from_iter((0..11).map(|index| matches!(index, 1 | 4 | 7 | 9)));
        if cache_indices {
            let _ = mask.indices();
        }
        let filtered = FilterArray::try_new(values, mask)?.into_array();
        let sliced = filtered.slice(range)?;

        assert_eq!(sliced.as_::<Filter>().child().len(), child_len);
        assert_arrays_eq!(
            sliced,
            PrimitiveArray::from_option_iter(expected),
            &mut VortexSession::empty().create_execution_ctx()
        );
        Ok(())
    }

    #[rstest]
    #[case(0..0, vec![])]
    #[case(4..4, vec![])]
    #[case(0..4, vec![1, 4, 7, 9])]
    #[case(1..2, vec![4])]
    fn slice_filter_boundaries(
        #[case] range: Range<usize>,
        #[case] expected: Vec<i32>,
    ) -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0..11).into_array();
        let filtered =
            FilterArray::try_new(values, Mask::from_indices(11, vec![1, 4, 7, 9]))?.into_array();

        assert_arrays_eq!(
            filtered.slice(range)?,
            PrimitiveArray::from_iter(expected),
            &mut VortexSession::empty().create_execution_ctx()
        );
        Ok(())
    }
}
