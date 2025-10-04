// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_scalar::ListScalar;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::register_kernel;
use crate::vtable::OperationsVTable;

/// MinMax implementation for [`FixedSizeListArray`].
impl MinMaxKernel for FixedSizeListVTable {
    fn min_max(&self, array: &FixedSizeListArray) -> VortexResult<Option<MinMaxResult>> {
        // Find the lexicographically minimum and maximum lists.
        let scalars: Vec<_> = (0..array.len())
            .filter_map(|i| {
                if array.is_valid(i) {
                    Some(FixedSizeListVTable::scalar_at(array, i))
                } else {
                    None
                }
            })
            .collect();

        if scalars.is_empty() {
            return Ok(None);
        }

        let minmax = scalars.iter().minmax_by(|a, b| {
            let a_list = ListScalar::try_from(*a).ok();
            let b_list = ListScalar::try_from(*b).ok();
            a_list
                .partial_cmp(&b_list)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(match minmax {
            itertools::MinMaxResult::NoElements => None,
            itertools::MinMaxResult::OneElement(scalar) => Some(MinMaxResult {
                min: (*scalar).clone(),
                max: (*scalar).clone(),
            }),
            itertools::MinMaxResult::MinMax(min, max) => Some(MinMaxResult {
                min: (*min).clone(),
                max: (*max).clone(),
            }),
        })
    }
}

register_kernel!(MinMaxKernelAdapter(FixedSizeListVTable).lift());

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::FixedSizeListArray;
    use crate::compute::min_max;
    use crate::validity::Validity;

    #[test]
    fn test_min_max() {
        // [[1, 2], [3, 4], [5, 6]]
        let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 3);

        let result = min_max(array.as_ref()).unwrap().unwrap();

        let expected_min = Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![1i32.into(), 2i32.into()],
            Nullability::NonNullable,
        );
        let expected_max = Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![5i32.into(), 6i32.into()],
            Nullability::NonNullable,
        );

        assert_eq!(result.min, expected_min);
        assert_eq!(result.max, expected_max);
    }

    #[test]
    fn test_min_max_with_duplicates() {
        // [[1, 2], [1, 2], [3, 4]]
        let elements = buffer![1i32, 2, 1, 2, 3, 4].into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 3);

        let result = min_max(array.as_ref()).unwrap().unwrap();

        let expected_min = Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![1i32.into(), 2i32.into()],
            Nullability::NonNullable,
        );
        let expected_max = Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![3i32.into(), 4i32.into()],
            Nullability::NonNullable,
        );

        assert_eq!(result.min, expected_min);
        assert_eq!(result.max, expected_max);
    }

    #[test]
    fn test_min_max_single_element() {
        // [[1, 2]]
        let elements = buffer![1i32, 2].into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 1);

        let result = min_max(array.as_ref()).unwrap().unwrap();

        let expected = Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![1i32.into(), 2i32.into()],
            Nullability::NonNullable,
        );

        assert_eq!(result.min, expected);
        assert_eq!(result.max, expected);
    }

    #[test]
    fn test_min_max_with_nulls() {
        // [null, [3, 4], [1, 2]]
        let elements = buffer![0i32, 0, 3, 4, 1, 2].into_array();
        let validity = Validity::from_iter([false, true, true]);
        let array = FixedSizeListArray::new(elements, 2, validity, 3);

        let result = min_max(array.as_ref()).unwrap().unwrap();

        // Min and max should ignore nulls
        let expected_min = Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![1i32.into(), 2i32.into()],
            Nullability::NonNullable,
        );
        let expected_max = Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![3i32.into(), 4i32.into()],
            Nullability::NonNullable,
        );

        assert_eq!(result.min, expected_min);
        assert_eq!(result.max, expected_max);
    }

    #[test]
    fn test_min_max_lexicographic() {
        // [[1, 5], [2, 3]] - lexicographically min is [1, 5], max is [2, 3]
        let elements = buffer![1i32, 5, 2, 3].into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 2);

        let result = min_max(array.as_ref()).unwrap().unwrap();

        let expected_min = Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![1i32.into(), 5i32.into()],
            Nullability::NonNullable,
        );
        let expected_max = Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![2i32.into(), 3i32.into()],
            Nullability::NonNullable,
        );

        assert_eq!(result.min, expected_min);
        assert_eq!(result.max, expected_max);
    }
}
