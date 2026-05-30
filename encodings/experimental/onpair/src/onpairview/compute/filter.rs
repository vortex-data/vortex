// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Filter at **ListView speed** — metadata only.
//!
//! [`OnPair`](crate::OnPair)'s filter rebuilds the surviving `codes` token
//! stream. `OnPairView` instead filters just the per-row `codes_offsets` and
//! `codes_sizes` (and `uncompressed_lengths` + validity) with the same mask and
//! **reuses the shared `codes` buffer and dictionary verbatim** — exactly the
//! `ListView` filter, but applied directly to our children so we skip building a
//! `ListViewArray` and re-dispatching through canonicalisation.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::OnPairView;
use crate::OnPairViewArrayExt;
use crate::OnPairViewArraySlotsExt;

impl FilterKernel for OnPairView {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Filter the per-row children directly. `codes` and the dictionary are
        // untouched, so this never reads the token payload.
        let codes_offsets = array.codes_offsets().clone().filter(mask.clone())?;
        let codes_sizes = array.codes_sizes().clone().filter(mask.clone())?;
        let uncompressed_lengths = array.uncompressed_lengths().clone().filter(mask.clone())?;
        let validity = array.array_validity().filter(mask)?;

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
