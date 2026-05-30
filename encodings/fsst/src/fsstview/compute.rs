// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metadata-only `filter` and `take` for [`FSSTView`].
//!
//! Both operations rewrite only the small `offsets`/`sizes`/`uncompressed_lengths`/`validity`
//! arrays and reuse the compressed byte heap (and symbol table) untouched. This is the core
//! "ListView speed" win over plain [`FSST`][crate::FSST], whose `filter`/`take` delegate to
//! `VarBin` and therefore rewrite the entire compressed heap.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::array::FSSTView;
use super::array::FSSTViewArrayExt;
use super::array::FSSTViewArraySlotsExt;

impl FilterKernel for FSSTView {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Filter only the addressing arrays; the byte heap and symbol table are reused as-is.
        let validity = array.fsstview_validity().filter(mask)?;
        let codes_offsets = array.codes_offsets().filter(mask.clone())?;
        let codes_sizes = array.codes_sizes().filter(mask.clone())?;
        let uncompressed_lengths = array.uncompressed_lengths().filter(mask.clone())?;

        // SAFETY: filter preserves all `FSSTView` invariants — offsets/sizes/lengths stay
        // non-nullable and equal-length, and validity tracks nullness separately.
        Ok(Some(
            unsafe {
                FSSTView::new_unchecked(
                    array.dtype().clone(),
                    array.symbols().clone(),
                    array.symbol_lengths().clone(),
                    array.codes_bytes_handle().clone(),
                    codes_offsets,
                    codes_sizes,
                    uncompressed_lengths,
                    validity,
                )
            }
            .into_array(),
        ))
    }
}

impl TakeExecute for FSSTView {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let dtype = array
            .dtype()
            .clone()
            .union_nullability(indices.dtype().nullability());

        let validity = array.fsstview_validity().take(indices)?;

        // `take` of a non-nullable child with non-nullable indices stays non-nullable, so the
        // `fill_null` (and the `cast`/`optimize` it pulls in) is pure overhead in the common case.
        // Only when the indices are nullable can a null index introduce a null we must fill with
        // zero — nullness itself is tracked separately by `validity`.
        let fill = indices.dtype().is_nullable();
        let codes_offsets = take_child(array.codes_offsets(), indices, fill)?;
        let codes_sizes = take_child(array.codes_sizes(), indices, fill)?;
        let uncompressed_lengths = take_child(array.uncompressed_lengths(), indices, fill)?;

        // SAFETY: take preserves all `FSSTView` invariants (see `filter`).
        Ok(Some(
            unsafe {
                FSSTView::new_unchecked(
                    dtype,
                    array.symbols().clone(),
                    array.symbol_lengths().clone(),
                    array.codes_bytes_handle().clone(),
                    codes_offsets,
                    codes_sizes,
                    uncompressed_lengths,
                    validity,
                )
            }
            .into_array(),
        ))
    }
}

/// Take a non-nullable integer child by `indices`, only filling nulls with zero when the indices
/// are nullable (and so could have introduced nulls). The child is always non-nullable on input.
fn take_child(child: &ArrayRef, indices: &ArrayRef, fill: bool) -> VortexResult<ArrayRef> {
    let taken = child.take(indices.clone())?;
    if fill {
        taken.fill_null(Scalar::zero_value(child.dtype()))
    } else {
        Ok(taken)
    }
}
