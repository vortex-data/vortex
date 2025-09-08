// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::vtable::OperationsVTable;
use vortex_array::{ArrayRef, IntoArray};
use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::{RLEArray, RLEVTable};

impl OperationsVTable<RLEVTable> for RLEVTable {
    fn slice(array: &RLEArray, range: Range<usize>) -> ArrayRef {
        let start = range.start;
        let length = range.end - range.start;

        // SAFETY: Slicing preserves all RLE invariants as we're creating a
        // view into the same underlying data with adjusted offset and length.
        unsafe {
            RLEArray::new_unchecked(
                array.values.clone(),
                array.indices.clone(),
                array.value_chunk_offsets.clone(),
                array.validity.slice(range),
                array.dtype.clone(),
                array.offset + start,
                length,
            )
            .into_array()
        }
    }

    fn scalar_at(array: &RLEArray, index: usize) -> Scalar {
        // Slice local index as validity is sliced on `slice`.
        if !array.validity.is_valid(index) {
            return Scalar::null(array.dtype.clone());
        }

        let abs_position = array.offset() + index;
        let chunk_local_index = array.indices().scalar_at(abs_position);

        let chunk_local_idx = chunk_local_index
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("Index must not be null");

        let chunk_id = array.chunk_idx(abs_position);
        let value_chunk_offset = array.value_chunk_offset(chunk_id);

        let value_scalar = array
            .values()
            .scalar_at(value_chunk_offset + chunk_local_idx);

        Scalar::new(array.dtype().clone(), value_scalar.into_value())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray};

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
            let value_chunk_offsets = PrimitiveArray::from_iter([0u64]).into_array();

            RLEArray::try_new(
                values,
                indices.clone(),
                value_chunk_offsets,
                Validity::NonNullable,
                indices.len(),
            )
            .unwrap()
        }

        pub(super) fn rle_array_with_nulls() -> RLEArray {
            let values = PrimitiveArray::from_iter([10u32, 20u32, 30u32]).into_array();
            let pattern = [0u16, 0u16, 1u16, 1u16, 1u16, 2u16, 0u16];
            let repeated: Vec<u16> = pattern.iter().cycle().take(1024).copied().collect();
            let indices = PrimitiveArray::from_iter(repeated).into_array();
            let value_chunk_offsets = PrimitiveArray::from_iter([0u64]).into_array();

            // Repeat the validity pattern to match indices length
            let validity = Validity::from_iter(
                [true, false, true, true, false, true, true]
                    .iter()
                    .cycle()
                    .take(1024)
                    .copied(),
            );

            RLEArray::try_new(
                values,
                indices.clone(),
                value_chunk_offsets,
                validity,
                indices.len(),
            )
            .unwrap()
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
    #[should_panic]
    fn test_scalar_at_out_of_bounds() {
        let array = fixture::rle_array();
        array.scalar_at(1025);
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
    fn test_slice_preserves_dtype() {
        let array = fixture::rle_array();
        let sliced = array.slice(1..4);

        assert_eq!(array.dtype(), sliced.dtype());
    }

    #[test]
    fn test_slice_across_chunk_boundaries() {
        use vortex_array::{IntoArray, ToCanonical};
        use vortex_buffer::Buffer;

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
