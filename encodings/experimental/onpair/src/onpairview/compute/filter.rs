// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Filter at **ListView speed**.
//!
//! [`OnPair`](crate::OnPair)'s filter wraps `codes` + `codes_offsets` in a
//! [`ListArray`](vortex_array::arrays::ListArray) and delegates to the `List`
//! filter, which **rebuilds the surviving `codes` token stream** into a fresh
//! contiguous buffer. `OnPairView` instead wraps `codes` + `codes_offsets` +
//! `codes_sizes` in a [`ListViewArray`](vortex_array::arrays::ListViewArray) and
//! delegates to the `ListView` filter, which only filters the per-row
//! `offsets`/`sizes` and **reuses the `codes` buffer verbatim**. The shared
//! dictionary blob and `dict_offsets` are likewise untouched. The only arrays
//! materialised are the tiny per-row children.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::OnPairView;
use crate::OnPairViewArrayExt;
use crate::OnPairViewArraySlotsExt;

impl FilterKernel for OnPairView {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // View the per-row token windows as a ListView and let the metadata-only
        // ListView filter do the work — it shares the `codes` (`elements`)
        // buffer and only filters the per-row `offsets`/`sizes`.
        let list_view = unsafe {
            ListViewArray::new_unchecked(
                array.codes().clone(),
                array.codes_offsets().clone(),
                array.codes_sizes().clone(),
                Validity::NonNullable,
            )
        };
        let filtered = list_view
            .into_array()
            .filter(mask.clone())?
            .execute::<ListViewArray>(ctx)?;

        // `uncompressed_lengths` + validity are short integer/bit arrays, so the
        // primitive filter cost is negligible next to the (avoided) codes copy.
        let uncompressed_lengths = array.uncompressed_lengths().clone().filter(mask.clone())?;
        let validity = array.array_validity().filter(mask)?;

        Ok(Some(
            unsafe {
                OnPairView::new_unchecked(
                    array.dtype().clone(),
                    array.dict_bytes_handle().clone(),
                    array.dict_offsets().clone(),
                    // `elements` is the *same* shared `codes` buffer.
                    filtered.elements().clone(),
                    filtered.offsets().clone(),
                    filtered.sizes().clone(),
                    uncompressed_lengths,
                    validity,
                    array.bits(),
                )
            }
            .into_array(),
        ))
    }
}
