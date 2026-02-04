// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use vortex_error::VortexResult;
use vortex_scalar::ListScalar;

use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::compute::IsSortedKernel;
use crate::compute::IsSortedKernelAdapter;
use crate::register_kernel;

/// Implementation of IsSortedKernel for ListArray.
///
/// This implementation uses lexicographic comparison of list elements.
/// Lists are compared element-wise; if one list is a prefix of another,
/// the shorter list is considered less than the longer one.
/// Null lists are considered the smallest values.
/// Non-comparable lists (which shouldn't occur for lists with the same element type)
/// are treated as making the array not sorted.
impl IsSortedKernel for ListVTable {
    fn is_sorted(&self, array: &ListArray) -> VortexResult<Option<bool>> {
        if array.len() <= 1 {
            return Ok(Some(true));
        }
        for i in 0..array.len() - 1 {
            let scalar_a = array.scalar_at(i)?;
            let scalar_b = array.scalar_at(i + 1)?;
            let a = ListScalar::try_from(&scalar_a)?;
            let b = ListScalar::try_from(&scalar_b)?;
            // For is_sorted, we allow Less and Equal, but not Greater or incomparable (None)
            match a.partial_cmp(&b) {
                Some(Ordering::Greater) | None => return Ok(Some(false)),
                _ => {}
            }
        }
        Ok(Some(true))
    }

    fn is_strict_sorted(&self, array: &ListArray) -> VortexResult<Option<bool>> {
        if array.len() <= 1 {
            return Ok(Some(true));
        }
        for i in 0..array.len() - 1 {
            let scalar_a = array.scalar_at(i)?;
            let scalar_b = array.scalar_at(i + 1)?;
            let a = ListScalar::try_from(&scalar_a)?;
            let b = ListScalar::try_from(&scalar_b)?;
            // For is_strict_sorted, we only allow Less, not Equal, Greater, or incomparable (None)
            match a.partial_cmp(&b) {
                Some(Ordering::Greater | Ordering::Equal) | None => return Ok(Some(false)),
                _ => {}
            }
        }
        Ok(Some(true))
    }
}

register_kernel!(IsSortedKernelAdapter(ListVTable).lift());

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::ArrayRef;
    use crate::builders::{ArrayBuilder as _, ListBuilder};
    use crate::compute::is_sorted;
    use crate::compute::is_strict_sorted;

    fn create_list_array(values: Vec<Option<Vec<i32>>>, validity: Vec<bool>) -> ArrayRef {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let nullability = Nullability::Nullable;
        let mut builder = ListBuilder::<u32>::new(element_dtype, nullability);
        for i in 0..validity.len() {
            if validity[i] {
                let value = values[i].as_ref().unwrap();
                let list_scalar = value.into_iter().map(|x| Scalar::primitive(*x, Nullability::NonNullable)).collect::<Vec<_>>();
                let list = Scalar::list(Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)), list_scalar, Nullability::NonNullable);
                builder.append_value(list.as_list()).unwrap();
            } else {
                builder.append_null();
            }
        }
        builder.finish()
    }

    #[test]
    fn test_list_is_sorted_i32_sorted() {
        // Sorted: [1,2], [2,3], [3,4]
        let list = create_list_array(vec![Some(vec![1,2]), Some(vec![2,3]), Some(vec![3,4])], vec![true, true, true]);
        assert_eq!(is_sorted(list.as_ref()).unwrap(), Some(true));
    }

    #[test]
    fn test_list_is_sorted_i32_unsorted() {
        // Unsorted: [1,2], [3,4], [2,3]
        let list = create_list_array(vec![Some(vec![1,2]), Some(vec![3,4]), Some(vec![2,3])], vec![true, true, true]);
        assert_eq!(is_sorted(list.as_ref()).unwrap(), Some(false));
    }

    #[test]
    fn test_list_is_sorted_i32_duplicates() {
        // With duplicates: [1,2], [1,2], [3,4]
        let list = create_list_array(vec![Some(vec![1,2]), Some(vec![1,2]), Some(vec![3,4])], vec![true, true, true]);
        assert_eq!(is_sorted(list.as_ref()).unwrap(), Some(true));
    }

    #[test]
    fn test_list_is_strict_sorted_i32_strict() {
        // Strict sorted: [1,2], [3,4], [5,6]
        let list = create_list_array(vec![Some(vec![1,2]), Some(vec![3,4]), Some(vec![5,6])], vec![true, true, true]);
        assert_eq!(is_strict_sorted(list.as_ref()).unwrap(), Some(true));
    }

    #[test]
    fn test_list_is_strict_sorted_i32_not_strict() {
        // Not strict: [1,2], [1,2], [3,4]
        let list = create_list_array(vec![Some(vec![1,2]), Some(vec![1,2]), Some(vec![3,4])], vec![true, true, true]);
        assert_eq!(is_strict_sorted(list.as_ref()).unwrap(), Some(false));
    }

    #[test]
    fn test_list_is_sorted_with_null_beginning() {
        // Null at beginning: null, [1,2], [2,3]
        let list = create_list_array(vec![None, Some(vec![1,2]), Some(vec![2,3])], vec![false, true, true]);
        assert_eq!(is_sorted(list.as_ref()).unwrap(), Some(true));
    }

    #[test]
    fn test_list_is_sorted_with_null_middle() {
        // Null in middle: [1,2], null, [3,4]
        let list = create_list_array(vec![Some(vec![1,2]), None, Some(vec![3,4])], vec![true, false, true]);
        assert_eq!(is_sorted(list.as_ref()).unwrap(), Some(false));
    }

    #[test]
    fn test_list_is_sorted_with_null_end() {
        // Null at end: [1,2], [2,3], null
        let list = create_list_array(vec![Some(vec![1,2]), Some(vec![2,3]), None], vec![true, true, false]);
        assert_eq!(is_sorted(list.as_ref()).unwrap(), Some(false));
    }

    #[test]
    fn test_list_is_sorted_empty() {
        // Empty list
        let list = create_list_array(vec![], vec![]);
        assert_eq!(is_sorted(list.as_ref()).unwrap(), Some(true));
    }

    #[test]
    fn test_list_is_sorted_single() {
        // Single element
        let list = create_list_array(vec![Some(vec![1,2])], vec![true]);
        assert_eq!(is_sorted(list.as_ref()).unwrap(), Some(true));
    }

    #[test]
    fn test_list_is_sorted_different_lengths() {
        // [1] < [1,2], so sorted
        let list = create_list_array(vec![Some(vec![1]), Some(vec![1,2])], vec![true, true]);
        assert_eq!(is_sorted(list.as_ref()).unwrap(), Some(true));

        // [1,2] > [1], so not sorted
        let list = create_list_array(vec![Some(vec![1,2]), Some(vec![1])], vec![true, true]);
        assert_eq!(is_sorted(list.as_ref()).unwrap(), Some(false));
    }

    #[test]
    fn test_list_is_strict_sorted_different_lengths() {
        // [1] < [1,2], so strict sorted
        let list = create_list_array(vec![Some(vec![1]), Some(vec![1,2])], vec![true, true]);
        assert_eq!(is_strict_sorted(list.as_ref()).unwrap(), Some(true));

        // [1,2] > [1], so not strict sorted
        let list = create_list_array(vec![Some(vec![1,2]), Some(vec![1])], vec![true, true]);
        assert_eq!(is_strict_sorted(list.as_ref()).unwrap(), Some(false));
    }
}
