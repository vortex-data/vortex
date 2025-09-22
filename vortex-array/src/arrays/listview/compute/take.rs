// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::compute::{self, TakeKernel, TakeKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, register_kernel};

/// [`ListViewArray`] take implementation.
///
/// This implementation is deliberately simple and read-optimized. We just take the `offsets` and
/// `sizes` at the requested indices and reuse the original `elements` array. This works because
/// `ListView` (unlike `List`) allows non-contiguous and out-of-order lists.
///
/// We don't slice the `elements` array because: (1) it would require computing min/max offsets
/// and adjusting all offsets accordingly, adding CPU overhead; (2) the `elements` array is already
/// `Arc`'d, so keeping the full reference has virtually no memory overhead; and (3) it keeps the
/// implementation simple and maintainable.
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
        Ok(unsafe {
            ListViewArray::new_unchecked(elements.clone(), new_offsets, new_sizes, new_validity)
        }
        .into_array())
    }
}

register_kernel!(TakeKernelAdapter(ListViewVTable).lift());
