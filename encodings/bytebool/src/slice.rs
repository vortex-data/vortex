// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::filter::FilterReduce;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ByteBool;
use crate::ByteBoolData;

impl SliceReduce for ByteBool {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ByteBoolData::new(
                array.buffer().slice(range.clone()),
                array.validity().slice(range)?,
            )
            .into_array(),
        ))
    }
}

impl FilterReduce for ByteBool {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let ranges = match mask.slices() {
            AllOr::Some(slices) => slices,
            // Precondition: FilterReduce only runs for non-trivial masks.
            AllOr::All | AllOr::None => {
                unreachable!("precondition violated: expected a Mask::Values slice list")
            }
        };
        let ranges: Vec<Range<usize>> = ranges.iter().map(|&(s, e)| s..e).collect();
        Ok(Some(
            ByteBoolData::new(
                array.buffer().filter_typed::<u8>(&ranges)?,
                array.validity().filter(mask)?,
            )
            .into_array(),
        ))
    }
}
