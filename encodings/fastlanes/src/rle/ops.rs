// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::vtable::OperationsVTable;
use vortex_array::{ArrayRef, IntoArray};
use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::{FL_CHUNK_SIZE, RLEArray, RLEVTable};

impl OperationsVTable<RLEVTable> for RLEVTable {
    fn slice(array: &RLEArray, range: Range<usize>) -> ArrayRef {
        let offset_in_chunk = array.offset();
        let chunk_start_idx = (offset_in_chunk + range.start) / FL_CHUNK_SIZE;
        let chunk_end_idx = (offset_in_chunk + range.end).div_ceil(FL_CHUNK_SIZE);

        let values_start_idx = array.values_idx_offset(chunk_start_idx);
        let values_end_idx = if chunk_end_idx < array.values_idx_offsets().len() {
            array.values_idx_offset(chunk_end_idx)
        } else {
            array.values().len()
        };

        let sliced_values = array.values().slice(values_start_idx..values_end_idx);

        let sliced_values_idx_offsets = array
            .values_idx_offsets()
            .slice(chunk_start_idx..chunk_end_idx);

        let sliced_indices = array
            .indices
            .slice(chunk_start_idx * FL_CHUNK_SIZE..chunk_end_idx * FL_CHUNK_SIZE);

        // SAFETY: Slicing preserves all invariants.
        unsafe {
            RLEArray::new_unchecked(
                sliced_values,
                sliced_indices,
                sliced_values_idx_offsets,
                array.dtype.clone(),
                // Keep the offset relative to the first chunk.
                (array.offset + range.start) % FL_CHUNK_SIZE,
                range.len(),
            )
            .into_array()
        }
    }

    fn scalar_at(array: &RLEArray, index: usize) -> Scalar {
        let offset_in_chunk = array.offset();
        let chunk_relative_idx = array.indices().scalar_at(offset_in_chunk + index);

        let chunk_relative_idx = chunk_relative_idx
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("Index must not be null");

        let chunk_id = (offset_in_chunk + index) / FL_CHUNK_SIZE;
        let value_idx_offset = array.values_idx_offset(chunk_id);

        let scalar = array
            .values()
            .scalar_at(value_idx_offset + chunk_relative_idx);

        Scalar::new(array.dtype().clone(), scalar.into_value())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::Buffer;

    use super::*;

    mod fixture {
        use super::*;

        pub(super) fn rle_array() -> RLEArray {
            let values = PrimitiveArray::from_iter([10u32, 20u32, 30u32]).into_array();
            let indices = PrimitiveArray::from_iter(
                [0u16, 0u16, 1u16, 1u16, 1u16, 2u16, 0u16]
                    .iter()
                    .cycle()
                    .take(1024)
                    .copied(),
            )
            .into_array();
            let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

            RLEArray::try_new(values, indices.clone(), values_idx_offsets, indices.len()).unwrap()
        }

        pub(super) fn rle_array_with_nulls() -> RLEArray {
            let values = PrimitiveArray::from_iter([10u32, 20u32, 30u32]).into_array();
            let pattern = [0u16, 0u16, 1u16, 1u16, 1u16, 2u16, 0u16];
            let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

            // Repeat the validity pattern to match indices length
            let validity = Validity::from_iter(
                [true, false, true, true, false, true, true]
                    .iter()
                    .cycle()
                    .take(1024)
                    .copied(),
            );

            let indices = PrimitiveArray::new(
                pattern
                    .iter()
                    .cycle()
                    .take(1024)
                    .copied()
                    .collect::<Buffer<u16>>(),
                validity,
            )
            .into_array();

            RLEArray::try_new(values, indices.clone(), values_idx_offsets, indices.len()).unwrap()
        }
    }

    #[test]
    fn test_scalar_at() {
        let array = fixture::rle_array();

        assert_eq!(array.scalar_at(0), 10u32.into());
        assert_eq!(array.scalar_at(1), 10u32.into());
        assert_eq!(array.scalar_at(2), 20u32.into());
        assert_eq!(array.scalar_at(3), 20u32.into());
        assert_eq!(array.scalar_at(4), 20u32.into());
        assert_eq!(array.scalar_at(5), 30u32.into());
        assert_eq!(array.scalar_at(6), 10u32.into());

        assert!(!array.scalar_at(1).is_null());
        assert!(!array.scalar_at(4).is_null());
    }

    #[test]
    fn test_scalar_at_with_nulls() {
        let array = fixture::rle_array_with_nulls();

        assert_eq!(array.scalar_at(0), 10u32.into());
        assert_eq!(array.scalar_at(2), 20u32.into());
        assert_eq!(array.scalar_at(3), 20u32.into());
        assert_eq!(array.scalar_at(5), 30u32.into());
        assert_eq!(array.scalar_at(6), 10u32.into());

        assert!(array.scalar_at(1).is_null());
        assert!(array.scalar_at(4).is_null());
    }

    #[test]
    fn test_scalar_at_slice() {
        let array = fixture::rle_array();
        let sliced = array.slice(2..6); // [20, 20, 20, 30]

        assert_eq!(sliced.len(), 4);
        assert_eq!(sliced.scalar_at(0), 20u32.into());
        assert_eq!(sliced.scalar_at(1), 20u32.into());
        assert_eq!(sliced.scalar_at(2), 20u32.into());
        assert_eq!(sliced.scalar_at(3), 30u32.into());

        assert!(!sliced.scalar_at(0).is_null());
        assert!(!sliced.scalar_at(3).is_null());
    }

    #[test]
    fn test_scalar_at_slice_with_nulls() {
        let array = fixture::rle_array_with_nulls();
        let sliced = array.slice(2..6); // [20, 20, 20, 30]

        assert_eq!(sliced.len(), 4);
        assert_eq!(sliced.scalar_at(0), 20u32.into());
        assert_eq!(sliced.scalar_at(1), 20u32.into());
        assert_eq!(sliced.scalar_at(2), Scalar::null_typed::<u32>());
        assert_eq!(sliced.scalar_at(3), 30u32.into());

        assert!(!sliced.scalar_at(0).is_null());
        assert!(!sliced.scalar_at(1).is_null());
        assert!(sliced.scalar_at(2).is_null());
        assert!(!sliced.scalar_at(3).is_null());
    }

    #[test]
    fn test_scalar_at_multiple_chunks() {
        // Test accessing elements around chunk boundaries
        let values: Buffer<u16> = (0..3000).map(|i| (i / 50) as u16).collect();
        let expected: Vec<u16> = (0..3000).map(|i| (i / 50) as u16).collect();
        let array = values.into_array();

        let encoded = RLEArray::encode(&array.to_primitive()).unwrap();

        // Access scalars from multiple chunks.
        for &idx in &[1023, 1024, 1025, 2047, 2048, 2049] {
            if idx < encoded.len() {
                let original_value = expected[idx];
                let encoded_value = encoded.scalar_at(idx).as_primitive().as_::<u16>().unwrap();
                assert_eq!(original_value, encoded_value, "Mismatch at index {}", idx);
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_out_of_bounds() {
        let array = fixture::rle_array();
        array.scalar_at(1025);
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_slice_out_of_bounds() {
        let array = fixture::rle_array().slice(0..1);
        array.scalar_at(1);
    }

    #[test]
    fn test_slice_full_range() {
        let array = fixture::rle_array_with_nulls();
        let sliced = array.slice(0..7);

        assert_eq!(sliced.len(), 7);
        assert_eq!(sliced.scalar_at(0), 10u32.into());
        assert_eq!(sliced.scalar_at(5), 30u32.into());
    }

    #[test]
    fn test_slice_partial_range() {
        let array = fixture::rle_array();
        let sliced = array.slice(4..6); // [20, 30]

        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.scalar_at(0), 20u32.into());
        assert_eq!(sliced.scalar_at(1), 30u32.into());
    }

    #[test]
    fn test_slice_single_element() {
        let array = fixture::rle_array();
        let sliced = array.slice(5..6); // [30]

        assert_eq!(sliced.len(), 1);
        assert_eq!(sliced.scalar_at(0), 30u32.into());
    }

    #[test]
    fn test_slice_empty_range() {
        let array = fixture::rle_array();
        let sliced = array.slice(3..3);

        assert_eq!(sliced.len(), 0);
    }

    #[test]
    fn test_slice_with_nulls() {
        let array = fixture::rle_array_with_nulls();
        let sliced = array.slice(1..4); // [null, 20, 20]

        assert_eq!(sliced.len(), 3);
        assert!(sliced.scalar_at(0).is_null());
        assert_eq!(sliced.scalar_at(1), 20u32.into());
        assert_eq!(sliced.scalar_at(2), 20u32.into());
    }

    #[test]
    fn test_slice_decode_with_nulls() {
        let array = fixture::rle_array_with_nulls();
        let sliced = array.slice(1..4).to_array().to_primitive(); // [null, 20, 20]

        assert_eq!(sliced.len(), 3);
        assert!(sliced.scalar_at(0).is_null());
        assert_eq!(sliced.scalar_at(1), 20u32.into());
        assert_eq!(sliced.scalar_at(2), 20u32.into());
    }

    #[test]
    fn test_slice_preserves_dtype() {
        let array = fixture::rle_array();
        let sliced = array.slice(1..4);

        assert_eq!(array.dtype(), sliced.dtype());
    }

    #[test]
    fn test_slice_across_chunk_boundaries() {
        let values: Buffer<u32> = (0..2100).map(|i| (i / 100) as u32).collect();
        let expected: Vec<u32> = (0..2100).map(|i| (i / 100) as u32).collect();
        let array = values.into_array();

        let encoded = RLEArray::encode(&array.to_primitive()).unwrap();

        // Slice across first and second chunk.
        let slice = encoded.slice(500..1500);
        let decoded_slice = slice.to_primitive();
        assert_eq!(decoded_slice.as_slice::<u32>(), &expected[500..1500]);

        // Slice across second and third chunk.
        let slice = encoded.slice(1000..2000);
        let decoded_slice = slice.to_primitive();
        assert_eq!(decoded_slice.as_slice::<u32>(), &expected[1000..2000]);
    }
}
