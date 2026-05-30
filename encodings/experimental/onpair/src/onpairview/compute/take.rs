// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Take at **ListView speed** — metadata only.
//!
//! A reordering/duplicating take gathers the per-row `codes_offsets`/`codes_sizes`
//! (and `uncompressed_lengths` + validity) at the requested indices and reuses
//! the shared `codes` buffer and dictionary verbatim. We take the children
//! directly rather than round-tripping through a `ListViewArray`. `take` returns
//! nullable arrays (null where an index is null); we refill the integer children
//! with zero and keep them non-nullable — the outer validity tracks nullness.

use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::dict::TakeReduce;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::OnPairView;
use crate::OnPairViewArray;
use crate::OnPairViewArrayExt;
use crate::OnPairViewArraySlotsExt;

/// Metadata-only take for [`OnPairView`].
impl TakeReduce for OnPairView {
    fn take(
        array: ArrayView<'_, OnPairView>,
        indices: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(apply_take(array, indices)?.into_array()))
    }
}

/// Execution-path take for [`OnPairView`].
impl TakeExecute for OnPairView {
    fn take(
        array: ArrayView<'_, OnPairView>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(apply_take(array, indices)?.into_array()))
    }
}

/// Take a non-nullable integer child and refill any nulls with zero.
fn take_int_filled(child: &ArrayRef, indices: &ArrayRef) -> VortexResult<ArrayRef> {
    let taken = child.clone().take(indices.clone())?;
    Ok(match_each_integer_ptype!(taken.dtype().as_ptype(), |P| {
        taken.fill_null(Scalar::primitive(P::zero(), Nullability::NonNullable))?
    }))
}

fn apply_take(
    array: ArrayView<'_, OnPairView>,
    indices: &ArrayRef,
) -> VortexResult<OnPairViewArray> {
    let codes_offsets = take_int_filled(array.codes_offsets(), indices)?;
    let codes_sizes = take_int_filled(array.codes_sizes(), indices)?;
    let uncompressed_lengths = take_int_filled(array.uncompressed_lengths(), indices)?;
    let validity = array.array_validity().take(indices)?;

    Ok(unsafe {
        OnPairView::new_unchecked(
            array.dtype().clone(),
            array.dict_bytes_handle().clone(),
            array.dict_offsets().clone(),
            // `codes` is the shared token buffer, reused as-is.
            array.codes().clone(),
            codes_offsets,
            codes_sizes,
            uncompressed_lengths,
            validity,
            array.bits(),
        )
    })
}
