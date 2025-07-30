// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;

use vortex_array::vtable::{OperationsVTable, ValidityHelper};
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{DeltaArray, DeltaVTable};

impl OperationsVTable<DeltaVTable> for DeltaVTable {
    fn slice(array: &DeltaArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let physical_start = start + array.offset();
        let physical_stop = stop + array.offset();

        let start_chunk = physical_start / 1024;
        let stop_chunk = physical_stop.div_ceil(1024);

        let bases = array.bases();
        let deltas = array.deltas();
        let validity = array.validity();
        let lanes = array.lanes();

        let new_bases = bases.slice(
            min(start_chunk * lanes, array.bases_len()),
            min(stop_chunk * lanes, array.bases_len()),
        )?;

        let new_deltas = deltas.slice(
            min(start_chunk * 1024, array.deltas_len()),
            min(stop_chunk * 1024, array.deltas_len()),
        )?;

        let new_validity = validity.slice(start, stop)?;

        let logical_len = stop - start;

        let arr = DeltaArray::try_new(
            new_bases,
            new_deltas,
            new_validity,
            physical_start % 1024,
            logical_len,
        )?;

        Ok(arr.into_array())
    }

    fn scalar_at(array: &DeltaArray, index: usize) -> VortexResult<Scalar> {
        let decompressed = array.slice(index, index + 1)?.to_primitive()?;
        decompressed.scalar_at(0)
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ToCanonical;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_error::VortexError;

    use super::*;

    #[test]
    fn test_slice_non_jagged_array_first_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        assert_eq!(
            delta
                .slice(10, 250)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            (10u32..250).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_second_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        assert_eq!(
            delta
                .slice(1024 + 10, 1024 + 250)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            ((1024 + 10u32)..(1024 + 250)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        assert_eq!(
            delta
                .slice(1000, 1048)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            (1000u32..1048).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_four() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        assert_eq!(
            delta
                .slice(2040, 2050)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            (2040u32..2050).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_whole() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        assert_eq!(
            delta
                .slice(0, 4096)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            (0u32..4096).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_empty() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        assert_eq!(
            delta
                .slice(0, 0)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            delta
                .slice(4096, 4096)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            delta
                .slice(1024, 1024)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            Vec::<u32>::new(),
        );
    }

    #[test]
    fn test_slice_jagged_array_second_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        assert_eq!(
            delta
                .slice(1024 + 10, 1024 + 250)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            ((1024 + 10u32)..(1024 + 250)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_jagged_array_empty() {
        let delta = DeltaArray::try_from_vec((0u32..4000).collect()).unwrap();

        assert_eq!(
            delta
                .slice(0, 0)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            delta
                .slice(4000, 4000)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            delta
                .slice(1024, 1024)
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<u32>(),
            Vec::<u32>::new(),
        );
    }

    #[test]
    fn test_slice_of_slice_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = delta.slice(10, 1013).unwrap();
        let sliced_again = sliced.slice(0, 2).unwrap();

        assert_eq!(
            sliced_again.to_primitive().unwrap().as_slice::<u32>(),
            vec![10, 11]
        );
    }

    #[test]
    fn test_slice_of_slice_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = delta.slice(10, 1013).unwrap();
        let sliced_again = sliced.slice(0, 2).unwrap();

        assert_eq!(
            sliced_again.to_primitive().unwrap().as_slice::<u32>(),
            vec![10, 11]
        );
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = delta.slice(1034, 1050).unwrap();
        let sliced_again = sliced.slice(0, 2).unwrap();

        assert_eq!(
            sliced_again.to_primitive().unwrap().as_slice::<u32>(),
            vec![1034, 1035]
        );
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = delta.slice(1034, 1050).unwrap();
        let sliced_again = sliced.slice(0, 2).unwrap();

        assert_eq!(
            sliced_again.to_primitive().unwrap().as_slice::<u32>(),
            vec![1034, 1035]
        );
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = delta.slice(1010, 1050).unwrap();
        let sliced_again = sliced.slice(5, 20).unwrap();

        assert_eq!(
            sliced_again.to_primitive().unwrap().as_slice::<u32>(),
            (1015..1030).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = delta.slice(1010, 1050).unwrap();
        let sliced_again = sliced.slice(5, 20).unwrap();

        assert_eq!(
            sliced_again.to_primitive().unwrap().as_slice::<u32>(),
            (1015..1030).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_scalar_at_non_jagged_array() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect())
            .unwrap()
            .into_array();

        assert_eq!(delta.scalar_at(0).unwrap(), 0_u32.into());
        assert_eq!(delta.scalar_at(1).unwrap(), 1_u32.into());
        assert_eq!(delta.scalar_at(10).unwrap(), 10_u32.into());
        assert_eq!(delta.scalar_at(1023).unwrap(), 1023_u32.into());
        assert_eq!(delta.scalar_at(1024).unwrap(), 1024_u32.into());
        assert_eq!(delta.scalar_at(1025).unwrap(), 1025_u32.into());
        assert_eq!(delta.scalar_at(2047).unwrap(), 2047_u32.into());

        assert!(matches!(
            delta.scalar_at(2048),
            Err(VortexError::OutOfBounds(2048, 0, 2048, _))
        ));

        assert!(matches!(
            delta.scalar_at(2049),
            Err(VortexError::OutOfBounds(2049, 0, 2048, _))
        ));
    }

    #[test]
    fn test_scalar_at_jagged_array() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect())
            .unwrap()
            .into_array();

        assert_eq!(delta.scalar_at(0).unwrap(), 0_u32.into());
        assert_eq!(delta.scalar_at(1).unwrap(), 1_u32.into());
        assert_eq!(delta.scalar_at(10).unwrap(), 10_u32.into());
        assert_eq!(delta.scalar_at(1023).unwrap(), 1023_u32.into());
        assert_eq!(delta.scalar_at(1024).unwrap(), 1024_u32.into());
        assert_eq!(delta.scalar_at(1025).unwrap(), 1025_u32.into());
        assert_eq!(delta.scalar_at(1999).unwrap(), 1999_u32.into());

        assert!(matches!(
            delta.scalar_at(2000),
            Err(VortexError::OutOfBounds(2000, 0, 2000, _))
        ));

        assert!(matches!(
            delta.scalar_at(2001),
            Err(VortexError::OutOfBounds(2001, 0, 2000, _))
        ));
    }

    #[rstest]
    // Basic delta arrays
    #[case::delta_u32(DeltaArray::try_from_vec((0..100).map(|i| i as u32).collect()).unwrap())]
    #[case::delta_i32(DeltaArray::try_from_vec((-50..50).map(|i| i as i32).collect()).unwrap())]
    #[case::delta_u64(DeltaArray::try_from_vec((0..100).map(|i| i as u64 * 10).collect()).unwrap())]
    #[case::delta_i64(DeltaArray::try_from_vec((-100..100).map(|i| i as i64 * 5).collect()).unwrap())]
    // Large arrays (multiple chunks)
    #[case::delta_large_u32(DeltaArray::try_from_vec((0..2048).map(|i| i as u32).collect()).unwrap())]
    #[case::delta_large_i32(DeltaArray::try_from_vec((-1024..1024).map(|i| i as i32).collect()).unwrap())]
    // Single element
    #[case::delta_single(DeltaArray::try_from_vec(vec![42i32]).unwrap())]
    fn test_delta_consistency(#[case] array: DeltaArray) {
        test_array_consistency(array.as_ref());
    }

    #[rstest]
    #[case::delta_u32_basic(DeltaArray::try_from_vec((10..20).map(|i| i as u32).collect()).unwrap())]
    #[case::delta_i32_basic(DeltaArray::try_from_vec((100..110).map(|i| i as i32).collect()).unwrap())]
    #[case::delta_u64_basic(DeltaArray::try_from_vec((1000..1010).map(|i| i as u64).collect()).unwrap())]
    #[case::delta_i64_basic(DeltaArray::try_from_vec((5000..5010).map(|i| i as i64).collect()).unwrap())]
    #[case::delta_u32_large(DeltaArray::try_from_vec((0..100).map(|i| i as u32 * 2).collect()).unwrap())]
    fn test_delta_binary_numeric(#[case] array: DeltaArray) {
        test_binary_numeric_array(array.into_array());
    }
}
