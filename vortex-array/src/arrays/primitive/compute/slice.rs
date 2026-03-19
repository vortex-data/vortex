// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::filter::FilterReduce;
use crate::arrays::slice::SliceReduce;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::vtable::ValidityHelper;

impl SliceReduce for Primitive {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let result = match_each_native_ptype!(array.ptype(), |T| {
            PrimitiveArray::from_buffer_handle(
                array.buffer_handle().slice_typed::<T>(range.clone()),
                T::PTYPE,
                array.validity().slice(range)?,
            )
            .into_array()
        });
        Ok(Some(result))
    }
}

impl FilterReduce for Primitive {
    fn filter(array: &PrimitiveArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let ranges: Vec<Range<usize>> = mask
            .slices()
            .unwrap_or_else(|| unreachable!(), || unreachable!())
            .iter()
            .map(|&(s, e)| s..e)
            .collect();
        let result = match_each_native_ptype!(array.ptype(), |T| {
            PrimitiveArray::from_buffer_handle(
                array.buffer_handle().filter_typed::<T>(&ranges)?,
                T::PTYPE,
                array.validity().filter(mask)?,
            )
            .into_array()
        });
        Ok(Some(result))
    }
}
