// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::compute::{self, FilterKernel, FilterKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

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

        // Filter the offsets and sizes arrays.
        let new_offsets = compute::filter(offsets.as_ref(), selection_mask)?;
        let new_sizes = compute::filter(sizes.as_ref(), selection_mask)?;

        // SAFETY: Filter operation maintains all `ListViewArray` invariants:
        // - Offsets and sizes are derived from existing valid child arrays.
        // - Offsets and sizes have the same length (both filtered by `selection_mask`).
        // - Validity matches the filtered array's nullability.
        let new_array = unsafe {
            ListViewArray::new_unchecked(elements.clone(), new_offsets, new_sizes, new_validity)
        };

        // TODO(connor)[ListView]: IsZeroCopyToList optimization.
        // TODO(connor)[ListView]: Rebuild if the threshold is too low.

        Ok(new_array.into_array())
    }
}

register_kernel!(FilterKernelAdapter(ListViewVTable).lift());
