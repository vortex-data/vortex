// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::filter::FilterReduce;
use crate::arrays::slice::SliceReduce;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;

impl SliceReduce for Primitive {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let result = match_each_native_ptype!(array.ptype(), |T| {
            PrimitiveArray::from_buffer_handle(
                array.buffer_handle().slice_typed::<T>(range.clone()),
                T::PTYPE,
                array.validity()?.slice(range)?,
            )
            .into_array()
        });
        Ok(Some(result))
    }
}

impl FilterReduce for Primitive {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let ranges = match mask.slices() {
            AllOr::Some(slices) => slices,
            // Precondition: FilterReduce only runs for non-trivial masks.
            AllOr::All | AllOr::None => {
                unreachable!("precondition violated: expected a Mask::Values slice list")
            }
        };
        let ranges: Vec<Range<usize>> = ranges.iter().map(|&(s, e)| s..e).collect();
        let result = match_each_native_ptype!(array.ptype(), |T| {
            PrimitiveArray::from_buffer_handle(
                array.buffer_handle().filter_typed::<T>(&ranges)?,
                T::PTYPE,
                array.validity()?.filter(mask)?,
            )
            .into_array()
        });
        Ok(Some(result))
    }
}
