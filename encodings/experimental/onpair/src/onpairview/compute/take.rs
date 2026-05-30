// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Take at **ListView speed**.
//!
//! As with [`filter`](super::filter), a reordering/duplicating take only needs
//! to gather the per-row `codes_offsets`/`codes_sizes` (and `uncompressed_lengths`
//! and validity) at the requested indices — the shared `codes` buffer and the
//! dictionary are reused verbatim. We express this by wrapping the per-row
//! children in a [`ListViewArray`](vortex_array::arrays::ListViewArray) and
//! delegating to its metadata-only take, which is exactly what produces the new
//! `offsets`/`sizes` (possibly out-of-order or overlapping — which `OnPairView`
//! tolerates by construction).

use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::dict::TakeReduce;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
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

fn apply_take(
    array: ArrayView<'_, OnPairView>,
    indices: &ArrayRef,
) -> VortexResult<OnPairViewArray> {
    // Reuse the ListView metadata-only take for the per-row windows.
    let list_view = unsafe {
        ListViewArray::new_unchecked(
            array.codes().clone(),
            array.codes_offsets().clone(),
            array.codes_sizes().clone(),
            Validity::NonNullable,
        )
    };
    let taken = list_view
        .into_array()
        .take(indices.clone())?
        .execute::<ListViewArray>(&mut vortex_array::LEGACY_SESSION.create_execution_ctx())?;

    // `take` returns a nullable array; refill nulls with zero and drop the
    // nullability so the `uncompressed_lengths` child stays a non-nullable
    // integer (the outer validity tracks nullness separately).
    let nullable_lengths = array.uncompressed_lengths().clone().take(indices.clone())?;
    let uncompressed_lengths =
        match_each_integer_ptype!(nullable_lengths.dtype().as_ptype(), |L| {
            nullable_lengths.fill_null(Scalar::primitive(L::zero(), Nullability::NonNullable))?
        });
    let validity = array.array_validity().take(indices)?;

    Ok(unsafe {
        OnPairView::new_unchecked(
            array.dtype().clone(),
            array.dict_bytes_handle().clone(),
            array.dict_offsets().clone(),
            // `elements` is the *same* shared `codes` buffer.
            taken.elements().clone(),
            taken.offsets().clone(),
            taken.sizes().clone(),
            uncompressed_lengths,
            validity,
            array.bits(),
        )
    })
}
