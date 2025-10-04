// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::ListScalar;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{IsSortedIteratorExt, IsSortedKernel, IsSortedKernelAdapter};
use crate::register_kernel;
use crate::vtable::OperationsVTable;

/// IsSorted implementation for [`FixedSizeListArray`].
impl IsSortedKernel for FixedSizeListVTable {
    fn is_sorted(&self, array: &FixedSizeListArray) -> VortexResult<Option<bool>> {
        // Compare lists lexicographically using the PartialOrd implementation on ListScalar.
        let scalars: Vec<Option<_>> = (0..array.len())
            .map(|i| {
                if array.is_valid(i) {
                    Some(FixedSizeListVTable::scalar_at(array, i))
                } else {
                    None
                }
            })
            .collect();

        let iter = scalars
            .iter()
            .map(|opt| opt.as_ref().and_then(|s| ListScalar::try_from(s).ok()));

        Ok(Some(iter.is_sorted()))
    }

    fn is_strict_sorted(&self, array: &FixedSizeListArray) -> VortexResult<Option<bool>> {
        // Compare lists lexicographically without duplicates.
        let scalars: Vec<Option<_>> = (0..array.len())
            .map(|i| {
                if array.is_valid(i) {
                    Some(FixedSizeListVTable::scalar_at(array, i))
                } else {
                    None
                }
            })
            .collect();

        let iter = scalars
            .iter()
            .map(|opt| opt.as_ref().and_then(|s| ListScalar::try_from(s).ok()));

        Ok(Some(iter.is_strict_sorted()))
    }
}

register_kernel!(IsSortedKernelAdapter(FixedSizeListVTable).lift());

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use super::*;
    use crate::IntoArray;
    use crate::arrays::FixedSizeListArray;
    use crate::compute::{is_sorted, is_strict_sorted};
    use crate::validity::Validity;

    #[test]
    fn test_is_sorted() {
        // [[1, 2], [3, 4], [5, 6]] - sorted
        let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 3);
        assert_eq!(is_sorted(array.as_ref()).unwrap(), Some(true));
    }

    #[test]
    fn test_is_sorted_with_duplicates() {
        // [[1, 2], [1, 2], [3, 4]] - sorted (duplicates allowed)
        let elements = buffer![1i32, 2, 1, 2, 3, 4].into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 3);
        assert_eq!(is_sorted(array.as_ref()).unwrap(), Some(true));
    }

    #[test]
    fn test_not_sorted() {
        // [[5, 6], [3, 4], [1, 2]] - not sorted
        let elements = buffer![5i32, 6, 3, 4, 1, 2].into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 3);
        assert_eq!(is_sorted(array.as_ref()).unwrap(), Some(false));
    }

    #[test]
    fn test_is_strict_sorted() {
        // [[1, 2], [3, 4], [5, 6]] - strictly sorted
        let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 3);
        assert_eq!(is_strict_sorted(array.as_ref()).unwrap(), Some(true));
    }

    #[test]
    fn test_not_strict_sorted_with_duplicates() {
        // [[1, 2], [1, 2], [3, 4]] - not strictly sorted (has duplicates)
        let elements = buffer![1i32, 2, 1, 2, 3, 4].into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 3);
        assert_eq!(is_strict_sorted(array.as_ref()).unwrap(), Some(false));
    }

    #[test]
    fn test_is_sorted_with_nulls() {
        // [null, [1, 2], [3, 4]] - sorted (nulls come first)
        let elements = buffer![0i32, 0, 1, 2, 3, 4].into_array();
        let validity = Validity::from_iter([false, true, true]);
        let array = FixedSizeListArray::new(elements, 2, validity, 3);
        assert_eq!(is_sorted(array.as_ref()).unwrap(), Some(true));
    }

    #[test]
    fn test_lexicographic_ordering() {
        // [[1, 5], [2, 3]] - sorted lexicographically (first element takes priority)
        let elements = buffer![1i32, 5, 2, 3].into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 2);
        assert_eq!(is_sorted(array.as_ref()).unwrap(), Some(true));

        // [[2, 3], [1, 5]] - not sorted lexicographically
        let elements = buffer![2i32, 3, 1, 5].into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 2);
        assert_eq!(is_sorted(array.as_ref()).unwrap(), Some(false));
    }
}
