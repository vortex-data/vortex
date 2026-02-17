// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::arrays::BoolArray;
use vortex_array::compute::ListContainsKernel;
use vortex_array::compute::ListContainsKernelAdapter;
use vortex_array::register_kernel;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::array::SequenceVTable;
use crate::compute::compare::find_intersection_scalar;

impl ListContainsKernel for SequenceVTable {
    fn list_contains(
        &self,
        list: &dyn Array,
        element: &Self::Array,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(list_scalar) = list.as_constant() else {
            return Ok(None);
        };

        let list_elements = list_scalar
            .as_list()
            .elements()
            .vortex_expect("non-null element (checked in entry)");

        let mut set_indices: Vec<usize> = Vec::new();
        for intercept in list_elements.iter() {
            let Some(intercept) = intercept.as_primitive().pvalue() else {
                continue;
            };
            if let Ok(intersection) = find_intersection_scalar(
                element.base(),
                element.multiplier(),
                element.len(),
                intercept,
            ) {
                set_indices.push(intersection)
            }
        }

        let nullability = list.dtype().nullability() | element.dtype().nullability();

        Ok(Some(
            BoolArray::from_indices(element.len(), set_indices, nullability.into()).to_array(),
        ))
    }
}

register_kernel!(ListContainsKernelAdapter(SequenceVTable).lift());

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::list_contains;
    use vortex_array::scalar::Scalar;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType::I32;

    use crate::SequenceArray;

    #[test]
    fn test_list_contains_seq() {
        let elements = ConstantArray::new(
            Scalar::list(
                Arc::new(I32.into()),
                vec![1.into(), 3.into()],
                Nullability::Nullable,
            ),
            3,
        );

        {
            // [1, 3] in  1
            //            2
            //            3
            let array = SequenceArray::typed_new(1, 1, Nullability::NonNullable, 3).unwrap();

            let result = list_contains(elements.as_ref(), array.as_ref()).unwrap();
            let expected = BoolArray::from_iter([Some(true), Some(false), Some(true)]);
            assert_arrays_eq!(result, expected);
        }

        {
            // [1, 3] in  1
            //            3
            //            5
            let array = SequenceArray::typed_new(1, 2, Nullability::NonNullable, 3).unwrap();

            let result = list_contains(elements.as_ref(), array.as_ref()).unwrap();
            let expected = BoolArray::from_iter([Some(true), Some(true), Some(false)]);
            assert_arrays_eq!(result, expected);
        }
    }
}
