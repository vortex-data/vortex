// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Bool {
    fn slice(array: ArrayView<'_, Bool>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let bit_buffer = array.to_bit_buffer().slice(range.clone());
        let validity = array.validity()?.slice(range)?;

        // Safety:
        // range is verified in the callers and is the same for both bits and validity.
        let array = unsafe { BoolArray::new_unchecked(bit_buffer, validity).into_array() };

        Ok(Some(array))
    }
}
