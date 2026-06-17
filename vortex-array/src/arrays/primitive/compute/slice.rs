// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Primitive {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let byte_width = array.ptype().byte_width();
        let byte_range = range.start * byte_width..range.end * byte_width;
        let values = array.buffer_handle().slice(byte_range);
        let validity = array.validity()?.slice(range)?;

        // SAFETY:
        // slicing an existing PrimitiveArray on element boundaries preserves the buffer
        // alignment, ptype, length, and validity invariants.
        let array = unsafe {
            PrimitiveArray::new_unchecked_from_handle(values, array.ptype(), validity).into_array()
        };

        Ok(Some(array))
    }
}
