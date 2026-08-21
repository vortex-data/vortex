// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_mask::MaskValuesRef;

use crate::arrays::VarBinViewArray;
use crate::arrays::filter::execute::buffer;
use crate::arrays::filter::execute::filter_validity;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::VarBinViewArrayExt;
use crate::buffer::BufferHandle;

pub fn filter_varbinview(array: &VarBinViewArray, mask: &MaskValuesRef) -> VarBinViewArray {
    let filtered_validity = filter_validity(array.varbinview_validity(), mask);

    let views = Buffer::<BinaryView>::from_byte_buffer(array.views_handle().as_host().clone());
    let filtered_views = buffer::filter_buffer(views, mask);

    // SAFETY: the filtered views are a subset of the original views and reference the same data
    // buffers, and the validity is filtered by the same mask so lengths stay aligned.
    unsafe {
        VarBinViewArray::new_handle_unchecked(
            BufferHandle::new_host(filtered_views.into_byte_buffer()),
            Arc::clone(array.data_buffers()),
            array.dtype().clone(),
            filtered_validity,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::VarBinViewArray;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn test_filter_varbinview_conformance() {
        test_filter_conformance(
            &VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]).into_array(),
            &mut array_session().create_execution_ctx(),
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
            &mut array_session().create_execution_ctx(),
        );
    }
}
