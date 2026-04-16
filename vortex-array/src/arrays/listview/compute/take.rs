// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::Zero;
use vortex_error::VortexResult;

use super::REBUILD_DENSITY_THRESHOLD;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::dict::TakeReduce;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;

/// Metadata-only take for [`ListViewArray`].
impl TakeReduce for ListView {
    fn take(array: ArrayView<'_, ListView>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // Approximate element density by the fraction of list rows retained. Assumes roughly
        // uniform list sizes; good enough to decide whether dragging along the full `elements`
        // buffer is worth avoiding a rebuild.
        let kept_row_fraction = indices.len() as f32 / array.sizes().len() as f32;
        if kept_row_fraction < REBUILD_DENSITY_THRESHOLD {
            return Ok(None);
        }

        Ok(Some(apply_take(array, indices)?.into_array()))
    }
}

/// Execution-path take for [`ListViewArray`].
///
/// This does the same metadata-only take as [`TakeReduce`], but also rebuilds the array if the
/// resulting array will be less dense than `REBUILD_DENSITY_THRESHOLD`.
impl TakeExecute for ListView {
    fn take(
        array: ArrayView<'_, ListView>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let kept_row_fraction = indices.len() as f32 / array.sizes().len() as f32;
        let taken = apply_take(array, indices)?;

        if kept_row_fraction < REBUILD_DENSITY_THRESHOLD {
            // TODO(connor)[ListView]: Ideally, we would only rebuild after all `take`s and `filter`
            // compute functions have run, at the "top" of the operator tree. However, we cannot do
            // this right now, so we will just rebuild every time (similar to `ListArray`).
            Ok(Some(
                taken
                    .rebuild(ListViewRebuildMode::MakeZeroCopyToList)?
                    .into_array(),
            ))
        } else {
            Ok(Some(taken.into_array()))
        }
    }
}

/// Shared metadata-only take: take `offsets`, `sizes` and `validity` at `indices` while reusing
/// the original `elements` buffer as-is.
fn apply_take(array: ArrayView<'_, ListView>, indices: &ArrayRef) -> VortexResult<ListViewArray> {
    let elements = array.elements();
    let offsets = array.offsets();
    let sizes = array.sizes();

    // Combine the array's validity with the indices' validity.
    let new_validity = array.validity()?.take(indices)?;

    // Take can reorder offsets, create gaps, and may introduce overlaps if `indices` contain
    // duplicates.
    let nullable_new_offsets = offsets.take(indices.clone())?;
    let nullable_new_sizes = sizes.take(indices.clone())?;

    // `take` returns nullable arrays; cast back to non-nullable (filling with zeros to represent
    // the null lists — the validity mask tracks nullness separately).
    let new_offsets = match_each_integer_ptype!(nullable_new_offsets.dtype().as_ptype(), |O| {
        nullable_new_offsets.fill_null(Scalar::primitive(O::zero(), Nullability::NonNullable))?
    });
    let new_sizes = match_each_integer_ptype!(nullable_new_sizes.dtype().as_ptype(), |S| {
        nullable_new_sizes.fill_null(Scalar::primitive(S::zero(), Nullability::NonNullable))?
    });

    // SAFETY: Take operation maintains all `ListViewArray` invariants:
    // - `new_offsets` and `new_sizes` are derived from existing valid child arrays.
    // - `new_offsets` and `new_sizes` are non-nullable.
    // - `new_offsets` and `new_sizes` have the same length (both taken with the same `indices`).
    // - Validity correctly reflects the combination of array and indices validity.
    Ok(unsafe {
        ListViewArray::new_unchecked(elements.clone(), new_offsets, new_sizes, new_validity)
    })
}
