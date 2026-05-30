// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Slicing an `OnPairViewArray` reuses the dictionary blob, the full `codes`
//! child and the full `dict_offsets` child. Only the per-row children change,
//! and unlike [`OnPair`](crate::OnPair) there is no `+ 1` on `codes_offsets`:
//! `codes_offsets`, `codes_sizes`, `uncompressed_lengths` and validity are all
//! narrowed to the same `[start, end)` window. No decode, no re-training.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::OnPairView;
use crate::OnPairViewArrayExt;
use crate::OnPairViewArraySlotsExt;

impl SliceReduce for OnPairView {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let codes_offsets = array.codes_offsets().slice(range.clone())?;
        let codes_sizes = array.codes_sizes().slice(range.clone())?;
        let uncompressed_lengths = array.uncompressed_lengths().slice(range.clone())?;
        let validity = array.array_validity().slice(range)?;
        Ok(Some(
            unsafe {
                OnPairView::new_unchecked(
                    array.dtype().clone(),
                    array.dict_bytes_handle().clone(),
                    array.dict_offsets().clone(),
                    array.codes().clone(),
                    codes_offsets,
                    codes_sizes,
                    uncompressed_lengths,
                    validity,
                    array.bits(),
                )
            }
            .into_array(),
        ))
    }
}
