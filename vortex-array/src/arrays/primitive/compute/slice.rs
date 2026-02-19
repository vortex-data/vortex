// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::arrays::SliceReduce;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::vtable::ValidityHelper;

impl SliceReduce for PrimitiveVTable {
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
