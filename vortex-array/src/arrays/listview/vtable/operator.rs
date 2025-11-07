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

    use crate::IntoArray;
    use crate::arrays::listview::tests::common::{
        create_basic_listview, create_nullable_listview, create_overlapping_listview,
    };
    use crate::arrays::{ListViewArray, PrimitiveArray};
    use crate::validity::Validity;

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

    #[test]
    fn test_listview_operator_with_selection() {
        // Create a ListView with 6 lists: [[10,11], [20,21], [30,31], [40,41], [50,51], [60,61]]
        let elements =
            PrimitiveArray::from_iter([10i32, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61])
                .into_array();
        let offsets = PrimitiveArray::from_iter([0u32, 2, 4, 6, 8, 10]).into_array();
        let sizes = PrimitiveArray::from_iter([2u32, 2, 2, 2, 2, 2]).into_array();
        let listview = ListViewArray::new(elements, offsets, sizes, Validity::AllValid);

        // Create selection mask: [true, false, true, false, true, false].
        let selection = Mask::from_iter([true, false, true, false, true, false]);

        // Execute with selection.
        let result = listview.execute_with_selection(&selection).unwrap();

        // Verify filtered length (3 lists selected).
        assert_eq!(result.len(), 3);

        let listview_vector = result.as_list();

        // Verify offsets are filtered to indices 0, 2, 4.
        let offsets = listview_vector.offsets().clone().into_u32();
        assert_eq!(offsets.elements().as_slice(), &[0, 4, 8]);

        // Verify sizes are filtered to indices 0, 2, 4.
        let sizes = listview_vector.sizes().clone().into_u32();
        assert_eq!(sizes.elements().as_slice(), &[2, 2, 2]);

        // Verify elements remain complete (not filtered).
        let elements = listview_vector.elements().as_primitive().clone().into_i32();
        assert_eq!(
            elements.elements().as_slice(),
            &[10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61]
        );
    }

    #[test]
    fn test_listview_operator_with_nulls_and_selection() {
        // Use the nullable listview: [[10,20], null, [50]]
        let listview = create_nullable_listview();

        // Create selection mask: [true, true, false].
        let selection = Mask::from_iter([true, true, false]);

        // Execute with selection.
        let result = listview.execute_with_selection(&selection).unwrap();

        // Verify filtered length (2 lists selected, including the null).
        assert_eq!(result.len(), 2);

        let listview_vector = result.as_list();

        // Verify offsets are filtered to indices 0 and 1.
        let offsets = listview_vector.offsets().clone().into_u32();
        assert_eq!(offsets.elements().as_slice(), &[0, 2]);

        // Verify sizes are filtered to indices 0 and 1.
        let sizes = listview_vector.sizes().clone().into_u32();
        assert_eq!(sizes.elements().as_slice(), &[2, 2]);

        // Verify validity mask correctly shows first list valid, second list null.
        assert!(listview_vector.validity().value(0)); // First list is valid.
        assert!(!listview_vector.validity().value(1)); // Second list is null.

        // Verify elements remain complete.
        let elements = listview_vector.elements().as_primitive().clone().into_i32();
        assert_eq!(elements.elements().as_slice(), &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_listview_operator_overlapping_with_selection() {
        // Use the overlapping listview: [[5,6,7], [2,3], [8,9], [0,1], [1,2,3,4]]
        let listview = create_overlapping_listview();

        // Create selection mask: [true, false, true, true, false].
        let selection = Mask::from_iter([true, false, true, true, false]);

        // Execute with selection.
        let result = listview.execute_with_selection(&selection).unwrap();

        // Verify filtered length (3 lists selected).
        assert_eq!(result.len(), 3);

        let listview_vector = result.as_list();

        // Verify offsets are filtered to indices 0, 2, 3 (out-of-order preserved).
        let offsets = listview_vector.offsets().clone().into_u32();
        assert_eq!(offsets.elements().as_slice(), &[5, 8, 0]);

        // Verify sizes are filtered.
        let sizes = listview_vector.sizes().clone().into_u32();
        assert_eq!(sizes.elements().as_slice(), &[3, 2, 2]);

        // Verify elements remain complete (all 10 elements).
        let elements = listview_vector.elements().as_primitive().clone().into_i32();
        assert_eq!(
            elements.elements().as_slice(),
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
    }
}
