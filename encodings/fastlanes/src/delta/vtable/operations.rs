// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ToCanonical;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use super::Delta;
use crate::DeltaArray;

impl OperationsVTable<Delta> for Delta {
    fn scalar_at(array: &DeltaArray, index: usize) -> VortexResult<Scalar> {
        let decompressed = array.slice(index..index + 1)?.to_primitive();
        decompressed.scalar_at(0)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::DeltaArray;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_slice_non_jagged_array_first_chunk_of_two() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2048).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let actual = delta.slice(10..250).unwrap();
        let expected = PrimitiveArray::from_iter(10u32..250).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_second_chunk_of_two() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2048).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_two() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2048).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let actual = delta.slice(1000..1048).unwrap();
        let expected = PrimitiveArray::from_iter(1000u32..1048).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_four() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..4096).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let actual = delta.slice(2040..2050).unwrap();
        let expected = PrimitiveArray::from_iter(2040u32..2050).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_whole() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..4096).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let actual = delta.slice(0..4096).unwrap();
        let expected = PrimitiveArray::from_iter(0u32..4096).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_empty() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..4096).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

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
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2000).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_jagged_array_empty() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..4000).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

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
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2048).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_of_jagged() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2000).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_non_jagged() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2048).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_jagged() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2000).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_non_jagged() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2048).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_jagged() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2000).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_scalar_at_non_jagged_array() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2048).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .into_array();

        let expected = PrimitiveArray::from_iter(0u32..2048).into_array();
        assert_arrays_eq!(delta, expected);
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_non_jagged_array_oob() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2048).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .into_array();
        delta.scalar_at(2048).unwrap();
    }
    #[test]
    fn test_scalar_at_jagged_array() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2000).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .into_array();

        let expected = PrimitiveArray::from_iter(0u32..2000).into_array();
        assert_arrays_eq!(delta, expected);
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_jagged_array_oob() {
        let delta = DeltaArray::try_from_primitive_array(
            &(0u32..2000).collect(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .into_array();
        delta.scalar_at(2000).unwrap();
    }

    #[rstest]
    // Basic delta arrays
    #[case::delta_u32((0u32..100).collect())]
    #[case::delta_u64((0..100).map(|i| i as u64 * 10).collect())]
    // Large arrays (multiple chunks)
    #[case::delta_large_u32((0u32..2048).collect())]
    #[case::delta_large_u64((0u64..2048).collect())]
    // Single element
    #[case::delta_single(PrimitiveArray::new(buffer![42u32], Validity::NonNullable))]
    fn test_delta_consistency(#[case] array: PrimitiveArray) {
        test_array_consistency(
            &DeltaArray::try_from_primitive_array(&array, &mut SESSION.create_execution_ctx())
                .unwrap()
                .into_array(),
        );
    }

    #[rstest]
    #[case::delta_u8_basic(PrimitiveArray::new(buffer![1u8, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u16_basic(PrimitiveArray::new(buffer![1u16, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u32_basic(PrimitiveArray::new(buffer![1u32, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u64_basic(PrimitiveArray::new(buffer![1u64, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u32_large(PrimitiveArray::new(buffer![1u32; 100], Validity::NonNullable))]
    fn test_delta_binary_numeric(#[case] array: PrimitiveArray) {
        test_binary_numeric_array(
            DeltaArray::try_from_primitive_array(&array, &mut SESSION.create_execution_ctx())
                .unwrap()
                .into_array(),
        );
    }
}
