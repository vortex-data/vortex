// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_vector::listview::ListViewVector;

use crate::ArrayRef;
use crate::arrays::{ListViewArray, ListViewVTable};
use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::vtable::{OperatorVTable, ValidityHelper};

impl OperatorVTable<ListViewVTable> for ListViewVTable {
    fn bind(
        array: &ListViewArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        // Create selection kernels over just the views rather than calculate the exact elements we
        // need from the child `elements` array.
        let offsets_kernel = ctx.bind(array.offsets(), selection)?;
        let sizes_kernel = ctx.bind(array.sizes(), selection)?;

        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        // TODO There is definitely a smarter way we can do this...
        let elements_kernel = ctx.bind(array.elements(), None)?;

        Ok(kernel(move || {
            let offsets = offsets_kernel.execute()?.into_primitive();
            let sizes = sizes_kernel.execute()?.into_primitive();

            let validity_mask = validity.execute()?;

            // TODO There is definitely a smarter way we can do this...
            let elements = elements_kernel.execute()?;

            Ok(ListViewVector::try_new(Arc::new(elements), offsets, sizes, validity_mask)?.into())
        }))
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::PTypeDowncast;
    use vortex_mask::Mask;
    use vortex_vector::VectorOps;

    use crate::arrays::listview::tests::common::create_basic_listview;

    #[test]
    fn test_listview_operator_basic() {
        // Test basic ListView execution without selection.
        // ListView: [[0,1,2], [3,4], [5,6], [7,8,9]]
        let listview = create_basic_listview();

        // Execute without selection.
        let result = listview.execute().unwrap();
        assert_eq!(result.len(), 4);

        // Verify the result is a ListViewVector.
        let listview_vector = result.as_list();

        // Verify offsets.
        let offsets = listview_vector.offsets().clone().into_u32();
        assert_eq!(offsets.elements().as_slice(), &[0, 3, 5, 7]);

        // Verify sizes.
        let sizes = listview_vector.sizes().clone().into_u32();
        assert_eq!(sizes.elements().as_slice(), &[3, 2, 2, 3]);

        // Verify elements are intact.
        let elements = listview_vector.elements().as_primitive().clone().into_i32();
        assert_eq!(
            elements.elements().as_slice(),
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        );

        // Verify validity is all valid.
        assert!(matches!(listview_vector.validity(), Mask::AllTrue(_)));
    }
}
