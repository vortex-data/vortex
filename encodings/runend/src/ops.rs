// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::search_sorted::SearchResult;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::RunEndArray;

impl OperationsVTable<RunEnd> for RunEnd {
    fn scalar_at(
        array: &RunEndArray,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array.values().scalar_at(array.find_physical_index(index)?)
    }
}

/// Find the physical offset for and index that would be an end of the slice i.e., one past the last element.
///
/// If the index exists in the array we want to take that position (as we are searching from the right)
/// otherwise we want to take the next one
pub(crate) fn find_slice_end_index(array: &ArrayRef, index: usize) -> VortexResult<usize> {
    let result = array
        .as_primitive_typed()
        .search_sorted(&PValue::from(index), SearchSortedSide::Right)?;
    Ok(match result {
        SearchResult::Found(i) => i,
        SearchResult::NotFound(i) => {
            if i == array.len() {
                i
            } else {
                i + 1
            }
        }
    })
}

#[cfg(test)]
mod tests {

    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::is_constant::is_constant;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::buffer;

    use crate::RunEndArray;

    #[test]
    fn slice_array() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap()
        .slice(3..8)
        .unwrap();
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 5);

        let expected = PrimitiveArray::from_iter(vec![2i32, 2, 3, 3, 3]).into_array();
        assert_arrays_eq!(arr, expected);
    }

    #[test]
    fn double_slice() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap()
        .slice(3..8)
        .unwrap();
        assert_eq!(arr.len(), 5);

        let doubly_sliced = arr.slice(0..3).unwrap();

        let expected = PrimitiveArray::from_iter(vec![2i32, 2, 3]).into_array();
        assert_arrays_eq!(doubly_sliced, expected);
    }

    #[test]
    fn slice_end_inclusive() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap()
        .slice(4..10)
        .unwrap();
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 6);

        let expected = PrimitiveArray::from_iter(vec![2i32, 3, 3, 3, 3, 3]).into_array();
        assert_arrays_eq!(arr, expected);
    }

    #[test]
    fn slice_at_end() {
        let re_array = RunEndArray::try_new(
            buffer![7_u64, 10].into_array(),
            buffer![2_u64, 3].into_array(),
        )
        .unwrap();

        assert_eq!(re_array.len(), 10);

        let sliced_array = re_array.slice(re_array.len()..re_array.len()).unwrap();
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

        let sliced_array = re_array.slice(2..5).unwrap();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(is_constant(&sliced_array, &mut ctx).unwrap())
    }

    #[test]
    fn ree_scalar_at_end() {
        let scalar = RunEndArray::encode(buffer![1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5].into_array())
            .unwrap()
            .scalar_at(11)
            .unwrap();
        assert_eq!(scalar, 5.into());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn slice_along_run_boundaries() {
        // Create a runend array with runs: [1, 1, 1] [4, 4, 4] [2, 2] [5, 5, 5, 5]
        // Run ends at indices: 3, 6, 8, 12
        let arr = RunEndArray::try_new(
            buffer![3u32, 6, 8, 12].into_array(),
            buffer![1i32, 4, 2, 5].into_array(),
        )
        .unwrap();

        // Slice from start of first run to end of first run (indices 0..3)
        let slice1 = arr.slice(0..3).unwrap();
        assert_eq!(slice1.len(), 3);
        let expected = PrimitiveArray::from_iter(vec![1i32, 1, 1]).into_array();
        assert_arrays_eq!(slice1, expected);

        // Slice from start of second run to end of second run (indices 3..6)
        let slice2 = arr.slice(3..6).unwrap();
        assert_eq!(slice2.len(), 3);
        let expected = PrimitiveArray::from_iter(vec![4i32, 4, 4]).into_array();
        assert_arrays_eq!(slice2, expected);

        // Slice from start of third run to end of third run (indices 6..8)
        let slice3 = arr.slice(6..8).unwrap();
        assert_eq!(slice3.len(), 2);
        let expected = PrimitiveArray::from_iter(vec![2i32, 2]).into_array();
        assert_arrays_eq!(slice3, expected);

        // Slice from start of last run to end of last run (indices 8..12)
        let slice4 = arr.slice(8..12).unwrap();
        assert_eq!(slice4.len(), 4);
        let expected = PrimitiveArray::from_iter(vec![5i32, 5, 5, 5]).into_array();
        assert_arrays_eq!(slice4, expected);

        // Slice spanning exactly two runs (indices 3..8)
        let slice5 = arr.slice(3..8).unwrap();
        assert_eq!(slice5.len(), 5);
        let expected = PrimitiveArray::from_iter(vec![4i32, 4, 4, 2, 2]).into_array();
        assert_arrays_eq!(slice5, expected);

        // Slice from middle of first run to end of second run (indices 1..6)
        let slice6 = arr.slice(1..6).unwrap();
        assert_eq!(slice6.len(), 5);
        let expected = PrimitiveArray::from_iter(vec![1i32, 1, 4, 4, 4]).into_array();
        assert_arrays_eq!(slice6, expected);

        // Slice from start of second run to middle of third run (indices 3..7)
        let slice7 = arr.slice(3..7).unwrap();
        assert_eq!(slice7.len(), 4);
        let expected = PrimitiveArray::from_iter(vec![4i32, 4, 4, 2]).into_array();
        assert_arrays_eq!(slice7, expected);
    }
}
