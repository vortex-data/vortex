// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Slicing an `OnPairArray` reuses the same dictionary blob and shares the
//! `codes` child; we only narrow the `codes_offsets` and `uncompressed_lengths`
//! slices and adjust the validity child. No decode, no re-training.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArrayExt;

impl SliceReduce for OnPair {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let codes_offsets = array.codes_offsets().slice(range.start..range.end + 1)?;
        let uncompressed_lengths = array.uncompressed_lengths().slice(range.clone())?;
        let validity = array.array_validity().slice(range)?;
        Ok(Some(
            unsafe {
                OnPair::new_unchecked(
                    array.dtype().clone(),
                    array.dict_bytes_handle().clone(),
                    array.dict_offsets().clone(),
                    array.codes().clone(),
                    codes_offsets,
                    uncompressed_lengths,
                    validity,
                    array.bits(),
                )
            }
            .into_array(),
        ))
    }
}
