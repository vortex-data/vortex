// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::Zero;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;

/// The density threshold at which we skip rebuilding the underlying `elements` buffer.
///
/// Rebuilding `elements` can be expensive, so we avoid it when the selection keeps most of
/// the list rows. However, we also don't want to drag around a large amount of garbage data
/// when the selection is sparse — below this fraction of list rows retained, the rebuild is
/// worth it.
const REBUILD_DENSITY_THRESHOLD: f32 = 0.1;

/// [`ListViewArray`] take implementation.
///
/// We always take the `offsets` and `sizes` at the requested indices. Whether we also rebuild
/// the `elements` buffer depends on the selection density:
///
/// - **Dense selections** (above `REBUILD_DENSITY_THRESHOLD`): reuse the original `elements`
///   buffer as-is. This is the cheap, read-optimized path. We may keep some unreferenced
///   elements in memory, but this is acceptable since we're not copying the data.
/// - **Sparse selections**: rebuild via [`ListViewRebuildMode::MakeZeroCopyToList`] so the
///   result does not carry a large amount of garbage data.
///
/// This works because `ListView` (unlike `List`) allows non-contiguous and out-of-order lists,
/// so the metadata-only take is valid without touching `elements`.
impl TakeReduce for ListView {
    fn take(array: ArrayView<'_, ListView>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let elements = array.elements();
        let offsets = array.offsets();
        let sizes = array.sizes();

        // Combine the array's validity with the indices' validity.
        let new_validity = array.validity()?.take(indices)?;

        // Take can reorder offsets, create gaps, and may introduce overlaps if `indices` contain
        // duplicates.
        let nullable_new_offsets = offsets.take(indices.clone())?;
        let nullable_new_sizes = sizes.take(indices.clone())?;

        // `take` returns nullable arrays; cast back to non-nullable (filling with zeros to
        // represent the null lists — the validity mask tracks nullness separately).
        let new_offsets = match_each_integer_ptype!(nullable_new_offsets.dtype().as_ptype(), |O| {
            nullable_new_offsets
                .fill_null(Scalar::primitive(O::zero(), Nullability::NonNullable))?
        });
        let new_sizes = match_each_integer_ptype!(nullable_new_sizes.dtype().as_ptype(), |S| {
            nullable_new_sizes.fill_null(Scalar::primitive(S::zero(), Nullability::NonNullable))?
        });

        // SAFETY: Take operation maintains all `ListViewArray` invariants:
        // - `new_offsets` and `new_sizes` are derived from existing valid child arrays.
        // - `new_offsets` and `new_sizes` are non-nullable.
        // - `new_offsets` and `new_sizes` have the same length (both taken with the same
        //   `indices`).
        // - Validity correctly reflects the combination of array and indices validity.
        let new_array = unsafe {
            ListViewArray::new_unchecked(elements.clone(), new_offsets, new_sizes, new_validity)
        };

        // Approximate element density by the fraction of list rows retained. Assumes roughly
        // uniform list sizes; good enough to decide whether dragging along the full `elements`
        // buffer is worth avoiding a rebuild.
        let kept_row_fraction = indices.len() as f32 / array.sizes().len() as f32;
        if kept_row_fraction >= REBUILD_DENSITY_THRESHOLD {
            return Ok(Some(new_array.into_array()));
        }

        // TODO(connor)[ListView]: Ideally, we would only rebuild after all `take`s and `filter`
        // compute functions have run, at the "top" of the operator tree. However, we cannot do
        // this right now, so we rebuild here for sparse selections.
        Ok(Some(
            new_array
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList)?
                .into_array(),
        ))
    }
}
