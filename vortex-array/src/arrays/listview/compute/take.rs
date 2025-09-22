// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::compute::{self, TakeKernel, TakeKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, Canonical, IntoArray, register_kernel};

impl TakeKernel for ListViewVTable {
    fn take(&self, array: &ListViewArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let elements = array.elements();
        let offsets = array.offsets();
        let sizes = array.sizes();

        // Compute the new validity by combining the array's validity with the indices' validity.
        let new_validity = array.validity().take(indices)?;

        let (new_elements, new_offsets, new_sizes) = compute_taken_elements_and_arrays(
            elements.as_ref(),
            offsets.as_ref(),
            sizes.as_ref(),
            indices,
        )?;

        // SAFETY: Take operation maintains all `ListViewArray` invariants:
        // - Offsets and sizes are derived from existing valid child arrays.
        // - Offsets and sizes have the same length (both taken with same `indices`).
        // - Validity correctly reflects the combination of array and indices validity.
        Ok(unsafe {
            ListViewArray::new_unchecked(new_elements, new_offsets, new_sizes, new_validity)
        }
        .into_array())
    }
}

register_kernel!(TakeKernelAdapter(ListViewVTable).lift());

/// Takes a [`ListViewArray`] by pushing down the `take` indices into the `offsets` and `sizes`.
///
/// This implementation optimizes for read performance by simply reusing the existing `elements`
/// array. Unlike `ListArray` and `FixedSizeListArray` (which must maintain contiguous, in-order
/// elements), `ListView` allows non-contiguous and out-of-order lists, enabling us to simply slice
/// the elements and adjust offsets.
///
/// # Example
///
/// ```text
/// Input:
///   elements = [X, X, X, a, b, Z, Z, Z, c, d, e, f, g, h, i, j, k, Y, Y]
///   Lists: #0=[i,j,k] (offset=11, size=3)
///          #1=[a,b] (offset=3, size=2)
///          #2=[X,X,X] (offset=0, size=3) - will not be taken
///          #3=[Z,Z,Z] (offset=5, size=3) - will not be taken
///          #4=[Y,Y] (offset=17, size=2) - will not be taken
///          #5=[c,d,e,f] (offset=8, size=4)
///          #6=[g,h] (offset=12, size=2) - will not be taken
///
/// Take indices: [0, 1, 5] (keep lists #0, #1, #5)
///
/// Result:
///   elements = [a, b, Z, Z, Z, c, d, e, f, g, h, i, j, k] (slice from 3..14)
///   offsets = [8, 0, 5] (adjusted: 11-3=8, 3-3=0, 8-3=5)
///   sizes = [3, 2, 4] (unchanged)
///
/// Note: Elements Z,Z,Z at indices 5,6,7 are kept even though they're not referenced,
/// because they fall within the range [3..14] between the first and last used elements.
/// This tradeoff avoids the cost of rebuilding the elements array.
/// ```
fn compute_taken_elements_and_arrays(
    elements: &dyn Array,
    offsets: &dyn Array,
    sizes: &dyn Array,
    indices: &dyn Array,
) -> VortexResult<(ArrayRef, ArrayRef, ArrayRef)> {
    // Step 1: Take the offsets and sizes arrays at the requested indices.
    let taken_offsets = compute::take(offsets, indices)?;
    let taken_sizes = compute::take(sizes, indices)?;

    // Step 2: Find the range of elements used by the taken lists.
    // If there are no taken offsets, return empty arrays.
    if taken_offsets.is_empty() {
        debug_assert!(taken_sizes.is_empty());
        let empty_elements = Canonical::empty(elements.dtype()).into_array();
        return Ok((empty_elements, taken_offsets, taken_sizes));
    }

    // From here on, we are guaranteed that the taken `offsets` and `sizes` are both non-nullable
    // and also non-empty.

    // TODO(connor)[ListView]: Figure out if truncating the `elements` is worth it.

    // Get min offset and maximum end index from the taken offsets + sizes.
    let min_offset_scalar = compute::min_max(&taken_offsets)?
        .vortex_expect("offsets cannot be null or empty here")
        .min;

    let list_ends = compute::add(&taken_offsets, &taken_sizes)
        .vortex_expect("`offset + size` somehow overflowed");
    let max_end_scalar = compute::min_max(&list_ends)?
        .vortex_expect("offsets cannot be null or empty here")
        .max;

    // Step 3: Slice elements array to only include the used range.
    let min_offset = min_offset_scalar
        .as_primitive()
        .as_::<usize>()
        .vortex_expect("offset cannot be null");
    let max_end = max_end_scalar
        .as_primitive()
        .as_::<usize>()
        .vortex_expect("end index cannot be null");

    let sliced_elements = elements.slice(min_offset..max_end);

    // Step 4: Adjust offsets by subtracting min_offset if necessary.
    let adjusted_offsets = if min_offset > 0 {
        compute::sub_scalar(&taken_offsets, min_offset_scalar)?
    } else {
        taken_offsets
    };

    Ok((sliced_elements, adjusted_offsets, taken_sizes))
}
