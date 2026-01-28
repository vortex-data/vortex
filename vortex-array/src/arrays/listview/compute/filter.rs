// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewRebuildMode;
use crate::arrays::ListViewVTable;
use crate::compute::FilterKernel;
use crate::compute::FilterKernelAdapter;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

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

/// [`ListViewArray`] filter implementation.
///
/// This implementation is deliberately simple and read-optimized. We just filter the `offsets` and
/// `sizes` arrays and reuse the original `elements` array. This works because `ListView` (unlike
/// `List`) allows non-contiguous and out-of-order lists.
///
/// We don't slice the `elements` array because it would require computing min/max offsets and
/// adjusting all offsets accordingly, which is not really worth the small potential memory we would
/// be able to get back.
///
/// The trade-off is that we may keep unreferenced elements in memory, but this is acceptable since
/// we're optimizing for read performance and the data isn't being copied.
impl FilterKernel for ListViewVTable {
    fn filter(&self, array: &ListViewArray, selection_mask: &Mask) -> VortexResult<ArrayRef> {
        let elements = array.elements();
        let offsets = array.offsets();
        let sizes = array.sizes();

        let new_validity = array.validity().filter(selection_mask)?;
        debug_assert!(
            new_validity
                .maybe_len()
                .is_none_or(|len| len == selection_mask.true_count())
        );

        // Simply filter the offsets and sizes arrays.
        let new_offsets = offsets.filter(selection_mask.clone())?;
        let new_sizes = sizes.filter(selection_mask.clone())?;

        // SAFETY: Filter operation maintains all `ListViewArray` invariants:
        // - Offsets and sizes are derived from existing valid child arrays.
        // - Offsets and sizes have the same length (both filtered by `selection_mask`).
        // - Validity matches the filtered array's nullability.
        let new_array = unsafe {
            ListViewArray::new_unchecked(elements.clone(), new_offsets, new_sizes, new_validity)
        };

        // TODO(connor)[ListView]: Ideally, we would only rebuild after all `take`s and `filter`
        // compute functions have run, at the "top" of the operator tree. However, we cannot do this
        // right now, so we will just rebuild every time (similar to `ListArray`).

        Ok(new_array
            .rebuild(ListViewRebuildMode::MakeZeroCopyToList)?
            .into_array())
    }
}

register_kernel!(FilterKernelAdapter(ListViewVTable).lift());
