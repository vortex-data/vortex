// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::compute::{self, FilterKernel, FilterKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, Canonical, IntoArray, register_kernel};

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

        let (new_elements, new_offsets, new_sizes) = compute_filtered_elements_and_arrays(
            elements.as_ref(),
            offsets.as_ref(),
            sizes.as_ref(),
            selection_mask,
        )?;

        // SAFETY: Filter operation maintains all `ListViewArray` invariants:
        // - Offsets and sizes are derived from existing valid child arrays.
        // - Offsets and sizes have the same length (both filtered by `selection_mask`).
        // - Validity matches the filtered array's nullability.
        Ok(unsafe {
            ListViewArray::new_unchecked(new_elements, new_offsets, new_sizes, new_validity)
        }
        .into_array())
    }
}

register_kernel!(FilterKernelAdapter(ListViewVTable).lift());

/// Filters a [`ListViewArray`] by pushing down the filter into the `offsets` and `sizes`.
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
///          #2=[X,X,X] (offset=0, size=3) - unused data at start
///          #3=[Z,Z,Z] (offset=5, size=3) - unused data in middle
///          #4=[Y,Y] (offset=17, size=2) - unused data at end
///          #5=[c,d,e,f] (offset=8, size=4)
///          #6=[g,h] (offset=12, size=2) - overlaps with #0's range
///
/// Filter: [true, true, false, false, false, true, false] (keep #0, #1, #5; skip others)
///
/// Result:
///   elements = [a, b, Z, Z, Z, c, d, e, f, g, h, i, j, k] (slice from 3..17, trimming X's and Y's)
///   offsets = [8, 0, 5] (adjusted: 11-3=8, 3-3=0, 8-3=5)
///   sizes = [3, 2, 4] (unchanged)
///
/// Note: Elements at indices 0,1,2 (X,X,X) and 17,18 (Y,Y) are trimmed from ends.
/// Elements Z,Z,Z at indices 5,6,7 are kept even though list #3 is not selected,
/// because they fall within the range [3..17] between the first and last used elements.
/// This shows the slice trims unused ends but preserves gaps in the middle for performance.
/// ```
fn compute_filtered_elements_and_arrays(
    elements: &dyn Array,
    offsets: &dyn Array,
    sizes: &dyn Array,
    selection_mask: &Mask,
) -> VortexResult<(ArrayRef, ArrayRef, ArrayRef)> {
    // Step 1: Filter the offsets and sizes arrays.
    let filtered_offsets = compute::filter(offsets, selection_mask)?;
    let filtered_sizes = compute::filter(sizes, selection_mask)?;

    // Step 2: Find the range of elements used by selected lists.
    // If there are no filtered offsets, return empty arrays.
    if filtered_offsets.is_empty() {
        debug_assert!(filtered_sizes.is_empty());
        let empty_elements = Canonical::empty(elements.dtype()).into_array();
        return Ok((empty_elements, filtered_offsets, filtered_sizes));
    }

    // From here on, we are guaranteed that the filtered `offsets` and `sizes` are both non-nullable
    // and also non-empty.

    // TODO(connor)[ListView]: Figure out if truncating the `elements` is worth it.

    // Get min offset and maximum end index from the filtered offsets + sizes.
    let min_offset_scalar = compute::min_max(&filtered_offsets)?
        .vortex_expect("offsets cannot be null or empty here")
        .min;

    let list_ends = compute::add(&filtered_offsets, &filtered_sizes)
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
        compute::sub_scalar(&filtered_offsets, min_offset_scalar)?
    } else {
        filtered_offsets
    };

    Ok((sliced_elements, adjusted_offsets, filtered_sizes))
}
