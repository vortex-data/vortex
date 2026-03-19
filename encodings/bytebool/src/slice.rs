// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::filter::FilterReduce;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ByteBool;
use crate::ByteBoolArray;

impl SliceReduce for ByteBool {
    fn slice(array: &ByteBoolArray, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ByteBoolArray::new(
                array.buffer().slice(range.clone()),
                array.validity().slice(range)?,
            )
            .into_array(),
        ))
    }
}

impl FilterReduce for ByteBool {
    fn filter(array: &ByteBoolArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let ranges: Vec<Range<usize>> = mask
            .slices()
            .unwrap_or_else(|| unreachable!(), || unreachable!())
            .iter()
            .map(|&(s, e)| s..e)
            .collect();
        Ok(Some(
            ByteBoolArray::new(
                array.buffer().filter_typed::<u8>(&ranges)?,
                array.validity().filter(mask)?,
            )
            .into_array(),
        ))
    }
}
