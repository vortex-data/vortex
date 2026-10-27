// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_mask::MaskValuesRef;

use crate::arrays::ListViewArray;
use crate::arrays::filter::execute::filter_validity;
use crate::arrays::listview::ListViewArraySlotsExt;

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
pub fn filter_listview(array: &ListViewArray, selection_mask: &MaskValuesRef) -> ListViewArray {
    let elements = array.elements();
    let offsets = array.offsets();
    let sizes = array.sizes();

    let new_validity = filter_validity(
        array
            .validity()
            .vortex_expect("validity is derivable for a valid ListViewArray"),
        selection_mask,
    );
    debug_assert!(
        new_validity
            .maybe_len()
            .is_none_or(|len| len == selection_mask.true_count())
    );

    let mask_for_filter = Mask::Values(MaskValuesRef::clone(selection_mask));
    let new_offsets = offsets
        .filter(mask_for_filter.clone())
        .vortex_expect("ListViewArray offsets are guaranteed to support filter");
    let new_sizes = sizes
        .filter(mask_for_filter)
        .vortex_expect("ListViewArray sizes are guaranteed to support filter");

    // SAFETY:
    // - Offsets and sizes are derived from existing valid child arrays.
    // - Filtering offsets, sizes, and validity with `selection_mask` preserves their row alignment.
    // - The retained elements buffer keeps every referenced element in bounds.
    unsafe { ListViewArray::new_unchecked(elements.clone(), new_offsets, new_sizes, new_validity) }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_filter_listview_conformance() {
        let mut ctx = SESSION.create_execution_ctx();
        // 3 lists: [1,2], [3,4], [5,6]
        let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
        let offsets = buffer![0u32, 2, 4].into_array();
        let sizes = buffer![2u32, 2, 2].into_array();
        let array = ListViewArray::new(
            elements.into_array(),
            offsets,
            sizes,
            Validity::NonNullable,
            &mut ctx,
        );
        test_filter_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_filter_listview_with_nulls_conformance() {
        let mut ctx = SESSION.create_execution_ctx();
        // 3 lists with nullability: [1,2], null, [5,6]
        let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
        let offsets = buffer![0u32, 2, 4].into_array();
        let sizes = buffer![2u32, 2, 2].into_array();
        let validity = Validity::from_iter([true, false, true]);
        let array = ListViewArray::new(elements.into_array(), offsets, sizes, validity, &mut ctx);
        test_filter_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
