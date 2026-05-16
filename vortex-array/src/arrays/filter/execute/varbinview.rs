// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_mask::MaskIter;
use vortex_mask::MaskValues;

use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::VarBinViewArrayExt;
use crate::buffer::BufferHandle;

pub fn filter_varbinview(array: &VarBinViewArray, mask: &Arc<MaskValues>) -> VarBinViewArray {
    let filter_mask = Mask::Values(Arc::clone(mask));
    let views = filter_views(array.views(), mask);
    let validity = array
        .varbinview_validity()
        .filter(&filter_mask)
        .vortex_expect("filtering VarBinView validity should not fail");

    // SAFETY: filtering views and validity by the same mask preserves all view invariants. The data
    // buffers are immutable and remain referenced by the copied views.
    unsafe {
        VarBinViewArray::new_handle_unchecked(
            BufferHandle::new_host(views.into_byte_buffer()),
            Arc::clone(array.data_buffers()),
            array.dtype().clone(),
            validity,
        )
    }
}

fn filter_views(views: &[BinaryView], mask: &MaskValues) -> Buffer<BinaryView> {
    match mask.threshold_iter(0.5) {
        MaskIter::Indices(indices) => {
            Buffer::from_trusted_len_iter(indices.iter().map(|idx| views[*idx]))
        }
        MaskIter::Slices(slices) => {
            let mut filtered = BufferMut::with_capacity(mask.true_count());
            for (start, end) in slices.iter().copied() {
                for view in &views[start..end] {
                    filtered.push(*view);
                }
            }
            filtered.freeze()
        }
    }
}

#[cfg(test)]
mod test {
    use crate::IntoArray;
    use crate::arrays::VarBinViewArray;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn test_filter_varbinview_conformance() {
        test_filter_conformance(
            &VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]).into_array(),
        );

        test_filter_conformance(
            &VarBinViewArray::from_iter_nullable_str([
                Some("one"),
                None,
                Some("three"),
                Some("four"),
                Some("five"),
            ])
            .into_array(),
        );
    }
}
