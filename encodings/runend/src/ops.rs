// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::arrays::ConstantArray;
use vortex_array::search_sorted::{SearchResult, SearchSorted, SearchSortedSide};
use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_scalar::{PValue, Scalar};

use crate::{RunEndArray, RunEndVTable};

impl OperationsVTable<RunEndVTable> for RunEndVTable {
    fn slice(array: &RunEndArray, range: Range<usize>) -> ArrayRef {
        let new_length = range.len();

        let slice_begin = array.find_physical_index(range.start);
        let slice_end = find_slice_end_index(array.ends(), range.end + array.offset());

        // If the sliced range contains only a single run, opt to return a ConstantArray.
        if slice_begin + 1 == slice_end {
            let value = array.values().scalar_at(slice_begin);
            return ConstantArray::new(value, new_length).into_array();
        }

        // SAFETY: we maintain the ends invariant in our slice implementation
        unsafe {
            RunEndArray::new_unchecked(
                array.ends().slice(slice_begin..slice_end),
                array.values().slice(slice_begin..slice_end),
                range.start + array.offset(),
                new_length,
            )
            .into_array()
        }
    }

    fn scalar_at(array: &RunEndArray, index: usize) -> Scalar {
        array.values().scalar_at(array.find_physical_index(index))
    }
}

/// Find the physical offset for and index that would be an end of the slice i.e., one past the last element.
///
/// If the index exists in the array we want to take that position (as we are searching from the right)
/// otherwise we want to take the next one
pub(crate) fn find_slice_end_index(array: &dyn Array, index: usize) -> usize {
    let result = array
        .as_primitive_typed()
        .search_sorted(&PValue::from(index), SearchSortedSide::Right);
    match result {
        SearchResult::Found(i) => i,
        SearchResult::NotFound(i) => {
            if i == array.len() {
                i
            } else {
                i + 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::RunEndArray;

    #[test]
    fn slice_array() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap()
        .slice(3..8);
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 5);

        assert_eq!(arr.to_primitive().as_slice::<i32>(), vec![2, 2, 3, 3, 3]);
    }

    #[test]
    fn double_slice() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap()
        .slice(3..8);
        assert_eq!(arr.len(), 5);

        let doubly_sliced = arr.slice(0..3);

        assert_eq!(
            doubly_sliced.to_primitive().as_slice::<i32>(),
            vec![2, 2, 3]
        );
    }

    #[test]
    fn slice_end_inclusive() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap()
        .slice(4..10);
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 6);

        assert_eq!(arr.to_primitive().as_slice::<i32>(), vec![2, 3, 3, 3, 3, 3]);
    }

    #[test]
    fn slice_at_end() {
        let re_array = RunEndArray::try_new(
            buffer![7_u64, 10].into_array(),
            buffer![2_u64, 3].into_array(),
        )
        .unwrap();

        assert_eq!(re_array.len(), 10);

        let sliced_array = re_array.slice(re_array.len()..re_array.len());
        assert!(sliced_array.is_empty());
    }

    #[test]
    fn slice_single_end() {
        let re_array = RunEndArray::try_new(
            buffer![7_u64, 10].into_array(),
            buffer![2_u64, 3].into_array(),
        )
        .unwrap();

        assert_eq!(re_array.len(), 10);

        let sliced_array = re_array.slice(2..5);

        assert!(sliced_array.is_constant())
    }

    #[test]
    fn ree_scalar_at_end() {
        let scalar = RunEndArray::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
        )
        .unwrap()
        .scalar_at(11);
        assert_eq!(scalar, 5.into());
    }

    #[test]
    fn slice_along_run_boundaries() {
        // Create a runend array with runs: [1, 1, 1] [4, 4, 4] [2, 2] [5, 5, 5, 5]
        // Run ends at indices: 3, 6, 8, 12
        let arr = RunEndArray::try_new(
            buffer![3u32, 6, 8, 12].into_array(),
            buffer![1i32, 4, 2, 5].into_array(),
        )
        .unwrap();

        // Slice from start of first run to end of first run (indices 0..3)
        let slice1 = arr.slice(0..3);
        assert_eq!(slice1.len(), 3);
        assert_eq!(slice1.to_primitive().as_slice::<i32>(), vec![1, 1, 1]);

        // Slice from start of second run to end of second run (indices 3..6)
        let slice2 = arr.slice(3..6);
        assert_eq!(slice2.len(), 3);
        assert_eq!(slice2.to_primitive().as_slice::<i32>(), vec![4, 4, 4]);

        // Slice from start of third run to end of third run (indices 6..8)
        let slice3 = arr.slice(6..8);
        assert_eq!(slice3.len(), 2);
        assert_eq!(slice3.to_primitive().as_slice::<i32>(), vec![2, 2]);

        // Slice from start of last run to end of last run (indices 8..12)
        let slice4 = arr.slice(8..12);
        assert_eq!(slice4.len(), 4);
        assert_eq!(slice4.to_primitive().as_slice::<i32>(), vec![5, 5, 5, 5]);

        // Slice spanning exactly two runs (indices 3..8)
        let slice5 = arr.slice(3..8);
        assert_eq!(slice5.len(), 5);
        assert_eq!(slice5.to_primitive().as_slice::<i32>(), vec![4, 4, 4, 2, 2]);

        // Slice from middle of first run to end of second run (indices 1..6)
        let slice6 = arr.slice(1..6);
        assert_eq!(slice6.len(), 5);
        assert_eq!(slice6.to_primitive().as_slice::<i32>(), vec![1, 1, 4, 4, 4]);

        // Slice from start of second run to middle of third run (indices 3..7)
        let slice7 = arr.slice(3..7);
        assert_eq!(slice7.len(), 4);
        assert_eq!(slice7.to_primitive().as_slice::<i32>(), vec![4, 4, 4, 2]);
    }
}
