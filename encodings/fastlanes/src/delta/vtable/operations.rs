// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ToCanonical;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use super::DeltaVTable;
use crate::DeltaArray;

impl OperationsVTable<DeltaVTable> for DeltaVTable {
    fn scalar_at(array: &DeltaArray, index: usize) -> VortexResult<Scalar> {
        let decompressed = array.slice(index..index + 1)?.to_primitive();
        decompressed.scalar_at(0)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;

    use crate::DeltaArray;

    #[test]
    fn test_slice_non_jagged_array_first_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let actual = delta.slice(10..250).unwrap();
        let expected = PrimitiveArray::from_iter(10u32..250).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_second_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let actual = delta.slice(1000..1048).unwrap();
        let expected = PrimitiveArray::from_iter(1000u32..1048).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_four() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        let actual = delta.slice(2040..2050).unwrap();
        let expected = PrimitiveArray::from_iter(2040u32..2050).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_whole() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        let actual = delta.slice(0..4096).unwrap();
        let expected = PrimitiveArray::from_iter(0u32..4096).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_empty() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        let actual = delta.slice(0..0).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);

        let actual = delta.slice(4096..4096).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);

        let actual = delta.slice(1024..1024).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_jagged_array_second_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_jagged_array_empty() {
        let delta = DeltaArray::try_from_vec((0u32..4000).collect()).unwrap();

        let actual = delta.slice(0..0).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);

        let actual = delta.slice(4000..4000).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);

        let actual = delta.slice(1024..1024).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_of_slice_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_scalar_at_non_jagged_array() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect())
            .unwrap()
            .into_array();

        let expected = PrimitiveArray::from_iter(0u32..2048).into_array();
        assert_arrays_eq!(delta, expected);
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_non_jagged_array_oob() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect())
            .unwrap()
            .into_array();
        delta.scalar_at(2048).unwrap();
    }
    #[test]
    fn test_scalar_at_jagged_array() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect())
            .unwrap()
            .into_array();

        let expected = PrimitiveArray::from_iter(0u32..2000).into_array();
        assert_arrays_eq!(delta, expected);
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_jagged_array_oob() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect())
            .unwrap()
            .into_array();
        delta.scalar_at(2000).unwrap();
    }

    #[rstest]
    // Basic delta arrays
    #[case::delta_u32(DeltaArray::try_from_vec((0u32..100).collect()).unwrap())]
    #[case::delta_u64(DeltaArray::try_from_vec((0..100).map(|i| i as u64 * 10).collect()).unwrap())]
    // Large arrays (multiple chunks)
    #[case::delta_large_u32(DeltaArray::try_from_vec((0u32..2048).collect()).unwrap())]
    #[case::delta_large_u64(DeltaArray::try_from_vec((0u64..2048).collect()).unwrap())]
    // Single element
    #[case::delta_single(DeltaArray::try_from_vec(vec![42u32]).unwrap())]
    fn test_delta_consistency(#[case] array: DeltaArray) {
        test_array_consistency(array.as_ref());
    }

    #[rstest]
    #[case::delta_u8_basic(DeltaArray::try_from_vec(vec![1u8, 1, 1, 1, 1]).unwrap())]
    #[case::delta_u16_basic(DeltaArray::try_from_vec(vec![1u16, 1, 1, 1, 1]).unwrap())]
    #[case::delta_u32_basic(DeltaArray::try_from_vec(vec![1u32, 1, 1, 1, 1]).unwrap())]
    #[case::delta_u64_basic(DeltaArray::try_from_vec(vec![1u64, 1, 1, 1, 1]).unwrap())]
    #[case::delta_u32_large(DeltaArray::try_from_vec(vec![1u32; 100]).unwrap())]
    fn test_delta_binary_numeric(#[case] array: DeltaArray) {
        test_binary_numeric_array(array.into_array());
    }
}
