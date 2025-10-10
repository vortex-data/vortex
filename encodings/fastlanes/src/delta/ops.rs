// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;
use std::ops::Range;

use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_scalar::Scalar;

use crate::{DeltaArray, DeltaVTable};

impl OperationsVTable<DeltaVTable> for DeltaVTable {
    fn slice(array: &DeltaArray, range: Range<usize>) -> ArrayRef {
        let physical_start = range.start + array.offset();
        let physical_stop = range.end + array.offset();

        let start_chunk = physical_start / 1024;
        let stop_chunk = physical_stop.div_ceil(1024);

        let bases = array.bases();
        let deltas = array.deltas();
        let lanes = array.lanes();

        let new_bases = bases.slice(
            min(start_chunk * lanes, array.bases_len())..min(stop_chunk * lanes, array.bases_len()),
        );

        let new_deltas = deltas.slice(
            min(start_chunk * 1024, array.deltas_len())..min(stop_chunk * 1024, array.deltas_len()),
        );

        // SAFETY: slicing valid bases/deltas preserves correctness
        unsafe {
            DeltaArray::new_unchecked(new_bases, new_deltas, physical_start % 1024, range.len())
                .into_array()
        }
    }

    fn scalar_at(array: &DeltaArray, index: usize) -> Scalar {
        let decompressed = array.slice(index..index + 1).to_primitive();
        decompressed.scalar_at(0)
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::{IntoArray, ToCanonical};

    use super::*;

    #[test]
    fn test_slice_non_jagged_array_first_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        assert_eq!(
            delta.slice(10..250).to_primitive().as_slice::<u32>(),
            (10u32..250).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_second_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        assert_eq!(
            delta
                .slice(1024 + 10..1024 + 250)
                .to_primitive()
                .as_slice::<u32>(),
            ((1024 + 10u32)..(1024 + 250)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        assert_eq!(
            delta.slice(1000..1048).to_primitive().as_slice::<u32>(),
            (1000u32..1048).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_four() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        assert_eq!(
            delta.slice(2040..2050).to_primitive().as_slice::<u32>(),
            (2040u32..2050).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_whole() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        assert_eq!(
            delta.slice(0..4096).to_primitive().as_slice::<u32>(),
            (0u32..4096).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_empty() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        assert_eq!(
            delta.slice(0..0).to_primitive().as_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            delta.slice(4096..4096).to_primitive().as_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            delta.slice(1024..1024).to_primitive().as_slice::<u32>(),
            Vec::<u32>::new(),
        );
    }

    #[test]
    fn test_slice_jagged_array_second_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        assert_eq!(
            delta
                .slice(1024 + 10..1024 + 250)
                .to_primitive()
                .as_slice::<u32>(),
            ((1024 + 10u32)..(1024 + 250)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_jagged_array_empty() {
        let delta = DeltaArray::try_from_vec((0u32..4000).collect()).unwrap();

        assert_eq!(
            delta.slice(0..0).to_primitive().as_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            delta.slice(4000..4000).to_primitive().as_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            delta.slice(1024..1024).to_primitive().as_slice::<u32>(),
            Vec::<u32>::new(),
        );
    }

    #[test]
    fn test_slice_of_slice_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = delta.slice(10..1013);
        let sliced_again = sliced.slice(0..2);

        assert_eq!(sliced_again.to_primitive().as_slice::<u32>(), vec![10, 11]);
    }

    #[test]
    fn test_slice_of_slice_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = delta.slice(10..1013);
        let sliced_again = sliced.slice(0..2);

        assert_eq!(sliced_again.to_primitive().as_slice::<u32>(), vec![10, 11]);
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = delta.slice(1034..1050);
        let sliced_again = sliced.slice(0..2);

        assert_eq!(
            sliced_again.to_primitive().as_slice::<u32>(),
            vec![1034, 1035]
        );
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = delta.slice(1034..1050);
        let sliced_again = sliced.slice(0..2);

        assert_eq!(
            sliced_again.to_primitive().as_slice::<u32>(),
            vec![1034, 1035]
        );
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = delta.slice(1010..1050);
        let sliced_again = sliced.slice(5..20);

        assert_eq!(
            sliced_again.to_primitive().as_slice::<u32>(),
            (1015..1030).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = delta.slice(1010..1050);
        let sliced_again = sliced.slice(5..20);

        assert_eq!(
            sliced_again.to_primitive().as_slice::<u32>(),
            (1015..1030).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_scalar_at_non_jagged_array() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect())
            .unwrap()
            .into_array();

        assert_eq!(delta.scalar_at(0), 0_u32.into());
        assert_eq!(delta.scalar_at(1), 1_u32.into());
        assert_eq!(delta.scalar_at(10), 10_u32.into());
        assert_eq!(delta.scalar_at(1023), 1023_u32.into());
        assert_eq!(delta.scalar_at(1024), 1024_u32.into());
        assert_eq!(delta.scalar_at(1025), 1025_u32.into());
        assert_eq!(delta.scalar_at(2047), 2047_u32.into());
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_non_jagged_array_oob() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect())
            .unwrap()
            .into_array();
        delta.scalar_at(2048);
    }
    #[test]
    fn test_scalar_at_jagged_array() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect())
            .unwrap()
            .into_array();

        assert_eq!(delta.scalar_at(0), 0_u32.into());
        assert_eq!(delta.scalar_at(1), 1_u32.into());
        assert_eq!(delta.scalar_at(10), 10_u32.into());
        assert_eq!(delta.scalar_at(1023), 1023_u32.into());
        assert_eq!(delta.scalar_at(1024), 1024_u32.into());
        assert_eq!(delta.scalar_at(1025), 1025_u32.into());
        assert_eq!(delta.scalar_at(1999), 1999_u32.into());
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_jagged_array_oob() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect())
            .unwrap()
            .into_array();
        delta.scalar_at(2000);
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
