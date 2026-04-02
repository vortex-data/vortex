// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementReduce;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::array::Sequence;
use crate::compute::compare::find_intersection_scalar;

impl ListContainsElementReduce for Sequence {
    fn list_contains(list: &ArrayRef, element: &Self::Array) -> VortexResult<Option<ArrayRef>> {
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
            BoolArray::from_indices(element.len(), set_indices, nullability.into()).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType::I32;
    use vortex_array::expr::list_contains;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;

    use crate::SequenceArray;

    #[test]
    fn test_list_contains_seq() {
        let list_scalar = Scalar::list(
            Arc::new(I32.into()),
            vec![1.into(), 3.into()],
            Nullability::Nullable,
        );

        {
            // [1, 3] in  1
            //            2
            //            3
            let array = SequenceArray::try_new_typed(1, 1, Nullability::NonNullable, 3).unwrap();

            let expr = list_contains(lit(list_scalar.clone()), root());
            let result = array.into_array().apply(&expr).unwrap();
            let expected = BoolArray::from_iter([Some(true), Some(false), Some(true)]);
            assert_arrays_eq!(result, expected);
        }

        {
            // [1, 3] in  1
            //            3
            //            5
            let array = SequenceArray::try_new_typed(1, 2, Nullability::NonNullable, 3).unwrap();

            let expr = list_contains(lit(list_scalar), root());
            let result = array.into_array().apply(&expr).unwrap();
            let expected = BoolArray::from_iter([Some(true), Some(true), Some(false)]);
            assert_arrays_eq!(result, expected);
        }
    }
}
