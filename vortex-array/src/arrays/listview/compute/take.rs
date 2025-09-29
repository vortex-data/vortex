// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{ListViewArray, ListViewRebuildMode, ListViewVTable};
use crate::compute::{self, TakeKernel, TakeKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, register_kernel};

/// The threshold for triggering a rebuild of the [`ListViewArray`].
///
/// By default, we will not touch the underlying `elements` array of the [`ListViewArray`] since it
/// can be potentially expensive to reorganize the array based on what views we have into it.
///
/// However, we also do not want to carry around a large amount of garbage data. Below this
/// threshold of the density of the selection mask, we will rebuild the [`ListViewArray`], removing
/// any garbage data.
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
impl TakeKernel for ListViewVTable {
    fn take(&self, array: &ListViewArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let elements = array.elements();
        let offsets = array.offsets();
        let sizes = array.sizes();

        // Compute the new validity by combining the array's validity with the indices' validity.
        let new_validity = array.validity().take(indices)?;

        // Take the offsets and sizes arrays at the requested indices.
        let new_offsets = compute::take(offsets.as_ref(), indices)?;
        let new_sizes = compute::take(sizes.as_ref(), indices)?;

        // SAFETY: Take operation maintains all `ListViewArray` invariants:
        // - Offsets and sizes are derived from existing valid child arrays.
        // - Offsets and sizes have the same length (both taken with same `indices`).
        // - Validity correctly reflects the combination of array and indices validity.
        let mut new_array = unsafe {
            ListViewArray::new_unchecked(elements.clone(), new_offsets, new_sizes, new_validity)
        };

        #[allow(clippy::cast_possible_truncation)]
        let rebuild_threshold = (array.len() as f64 * REBUILD_DENSITY_THRESHOLD) as usize;

        // TODO(connor)[ListView]: Figure out a better heuristic.
        if indices.len() <= rebuild_threshold {
            new_array = new_array.rebuild(ListViewRebuildMode::RemoveGaps);
        }

        Ok(new_array.into_array())
    }
}

register_kernel!(TakeKernelAdapter(ListViewVTable).lift());
