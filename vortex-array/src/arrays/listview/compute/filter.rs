// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::compute::{self, FilterKernel, FilterKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

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
        Ok(unsafe {
            ListViewArray::new_unchecked(elements.clone(), new_offsets, new_sizes, new_validity)
        }
        .into_array())
    }
}

register_kernel!(FilterKernelAdapter(ListViewVTable).lift());
