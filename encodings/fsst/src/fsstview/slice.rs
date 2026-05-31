// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use super::array::FSSTView;
use super::array::FSSTViewArrayExt;
use super::array::FSSTViewArraySlotsExt;

impl SliceReduce for FSSTView {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // Slicing leaves the symbol table and compressed byte heap intact; we only slice the
        // addressing arrays.
        Ok(Some(
            unsafe {
                FSSTView::new_unchecked(
                    array.dtype().clone(),
                    array.symbols().clone(),
                    array.symbol_lengths().clone(),
                    array.codes_bytes_handle().clone(),
                    array.codes_offsets().slice(range.clone())?,
                    array.codes_ends().slice(range.clone())?,
                    array.uncompressed_lengths().slice(range.clone())?,
                    array.fsstview_validity().slice(range)?,
                )
            }
            .into_array(),
        ))
    }
}
