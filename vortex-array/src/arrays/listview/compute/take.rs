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
use crate::arrays::listview::ListViewRebuildMode;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;

// TODO(connor)[ListView]: Make use of this threshold after we start migrating operators.
/// The threshold for triggering a rebuild of the [`ListViewArray`].
///
/// By default, we will not touch the underlying `elements` array of the [`ListViewArray`] since it
/// can be potentially expensive to reorganize the array based on what views we have into it.
///
/// However, we also do not want to carry around a large amount of garbage data. Below this
/// threshold of the density of the selection mask, we will rebuild the [`ListViewArray`], removing
/// any garbage data.
#[allow(unused)]
const REBUILD_DENSITY_THRESHOLD: f64 = 0.1;

/// [`ListViewArray`] take implementation.
///
/// This implementation is deliberately simple and read-optimized. We just take the `offsets` and
/// `sizes` at the requested indices and reuse the original `elements` array. This works because
/// `ListView` (unlike `List`) allows non-contiguous and out-of-order lists.
///
/// We don't slice the `elements` array because it would require computing min/max offsets and
/// adjusting all offsets accordingly, which is not really worth the small potential memory we would
/// be able to get back.
///
/// The trade-off is that we may keep unreferenced elements in memory, but this is acceptable since
/// we're optimizing for read performance and the data isn't being copied.
impl TakeReduce for ListView {
    fn take(array: ArrayView<'_, ListView>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let elements = array.elements();
        let offsets = array.offsets();
        let sizes = array.sizes();

        // Compute the new validity by combining the array's validity with the indices' validity.
        let new_validity = array.validity().take(indices)?;

        // Take the offsets and sizes arrays at the requested indices.
        // Take can reorder offsets, create gaps, and may introduce overlaps if the `indices`
        // contain duplicates.
        let nullable_new_offsets = offsets.take(indices.clone())?;
        let nullable_new_sizes = sizes.take(indices.clone())?;

        // Since `take` returns nullable arrays, we simply cast it back to non-nullable (filled with
        // zeros to represent null lists).
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

        // TODO(connor)[ListView]: Ideally, we would only rebuild after all `take`s and `filter`
        // compute functions have run, at the "top" of the operator tree. However, we cannot do this
        // right now, so we will just rebuild every time (similar to `ListArray`).

        Ok(Some(
            new_array
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList)?
                .into_array(),
        ))
    }
}
