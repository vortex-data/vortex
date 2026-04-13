// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::RLE;
use crate::FL_CHUNK_SIZE;
use crate::rle::RLEArrayExt;

impl OperationsVTable<RLE> for RLE {
    fn scalar_at(
        array: ArrayView<'_, RLE>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let offset_in_chunk = array.offset();
        let chunk_relative_idx = array.indices().scalar_at(offset_in_chunk + index)?;

        let chunk_relative_idx = chunk_relative_idx
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("Index must not be null");

        let chunk_id = (offset_in_chunk + index) / FL_CHUNK_SIZE;
        let value_idx_offset = array.values_idx_offset(chunk_id);

        let scalar = array
            .values()
            .scalar_at(value_idx_offset + chunk_relative_idx)?;

        Scalar::try_new(array.dtype().clone(), scalar.into_value())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use super::*;
    use crate::RLE;
    use crate::RLEArray;
    use crate::RLEData;

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

            RLE::try_new(
                values,
                indices.clone(),
                values_idx_offsets,
                0,
                indices.len(),
            )
            .vortex_expect("RLEData is always valid")
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

            RLE::try_new(
                values,
                indices.clone(),
                values_idx_offsets,
                0,
                indices.len(),
            )
            .vortex_expect("RLEData is always valid")
        }
    }

    #[test]
    fn test_scalar_at() {
        use vortex_array::assert_arrays_eq;

        let array = fixture::rle_array();
        let expected = PrimitiveArray::from_iter([10u32, 10, 20, 20, 20, 30, 10]);
        assert_arrays_eq!(array.slice(0..7).unwrap(), expected);
    }

    #[test]
    fn test_scalar_at_with_nulls() {
        use vortex_array::assert_arrays_eq;

        let array = fixture::rle_array_with_nulls();
        let expected = PrimitiveArray::from_option_iter([
            Some(10u32),
            None,
            Some(20),
            Some(20),
            None,
            Some(30),
            Some(10),
        ]);
        assert_arrays_eq!(array.slice(0..7).unwrap(), expected);
    }

    #[test]
    fn test_scalar_at_slice() {
        use vortex_array::assert_arrays_eq;

        let array = fixture::rle_array();
        let sliced = array.slice(2..6).unwrap(); // [20, 20, 20, 30]

        assert_eq!(sliced.len(), 4);
        let expected = PrimitiveArray::from_iter([20u32, 20, 20, 30]);
        assert_arrays_eq!(sliced, expected);
    }

    #[test]
    fn test_scalar_at_slice_with_nulls() {
        use vortex_array::assert_arrays_eq;

        let array = fixture::rle_array_with_nulls();
        let sliced = array.slice(2..6).unwrap(); // [20, 20, null, 30]

        assert_eq!(sliced.len(), 4);
        let expected = PrimitiveArray::from_option_iter([Some(20u32), Some(20), None, Some(30)]);
        assert_arrays_eq!(sliced, expected);
    }

    #[test]
    fn test_scalar_at_multiple_chunks() {
        // Test accessing elements around chunk boundaries
        let values: Buffer<u16> = (0..3000).map(|i| (i / 50) as u16).collect();
        let expected: Vec<u16> = (0..3000).map(|i| (i / 50) as u16).collect();
        let array = values.into_array();

        let encoded = RLEData::encode(array.to_primitive().as_view()).unwrap();

        // Access scalars from multiple chunks.
        for &idx in &[1023, 1024, 1025, 2047, 2048, 2049] {
            if idx < encoded.len() {
                let original_value = expected[idx];
                let encoded_value = encoded
                    .scalar_at(idx)
                    .unwrap()
                    .as_primitive()
                    .as_::<u16>()
                    .unwrap();
                assert_eq!(original_value, encoded_value, "Mismatch at index {}", idx);
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_out_of_bounds() {
        let array = fixture::rle_array();
        array.scalar_at(1025).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_slice_out_of_bounds() {
        let array = fixture::rle_array().slice(0..1).unwrap();
        array.scalar_at(1).unwrap();
    }

    #[test]
    fn test_slice_full_range() {
        let array = fixture::rle_array_with_nulls();
        let sliced = array.slice(0..7).unwrap();

        let expected = PrimitiveArray::from_option_iter([
            Some(10u32),
            None,
            Some(20),
            Some(20),
            None,
            Some(30),
            Some(10),
        ]);
        assert_arrays_eq!(sliced.into_array(), expected.into_array());
    }

    #[test]
    fn test_slice_partial_range() {
        let array = fixture::rle_array();
        let sliced = array.slice(4..6).unwrap(); // [20, 30]

        let expected = buffer![20u32, 30].into_array();
        assert_arrays_eq!(sliced, expected);
    }

    #[test]
    fn test_slice_single_element() {
        let array = fixture::rle_array();
        let sliced = array.slice(5..6).unwrap(); // [30]

        let expected = buffer![30u32].into_array();
        assert_arrays_eq!(sliced, expected);
    }

    #[test]
    fn test_slice_empty_range() {
        let array = fixture::rle_array();
        let sliced = array.slice(3..3).unwrap();

        assert_eq!(sliced.len(), 0);
    }

    #[test]
    fn test_slice_with_nulls() {
        let array = fixture::rle_array_with_nulls();
        let sliced = array.slice(1..4).unwrap(); // [null, 20, 20]

        let expected = PrimitiveArray::from_option_iter([Option::<u32>::None, Some(20), Some(20)]);
        assert_arrays_eq!(sliced.into_array(), expected.into_array());
    }

    #[test]
    fn test_slice_decode_with_nulls() {
        let array = fixture::rle_array_with_nulls();
        let sliced = array.slice(1..4).unwrap().to_primitive(); // [null, 20, 20]

        let expected = PrimitiveArray::from_option_iter([Option::<u32>::None, Some(20), Some(20)]);
        assert_arrays_eq!(sliced.into_array(), expected.into_array());
    }

    #[test]
    fn test_slice_preserves_dtype() {
        let array = fixture::rle_array();
        let sliced = array.slice(1..4).unwrap();

        assert_eq!(array.dtype(), sliced.dtype());
    }

    #[test]
    fn test_slice_across_chunk_boundaries() {
        let values: Buffer<u32> = (0..2100).map(|i| (i / 100) as u32).collect();
        let expected: Vec<u32> = (0..2100).map(|i| (i / 100) as u32).collect();
        let array = values.into_array();

        let encoded = RLEData::encode(array.to_primitive().as_view()).unwrap();

        // Slice across first and second chunk.
        let slice = encoded.slice(500..1500).unwrap();
        assert_arrays_eq!(
            slice,
            PrimitiveArray::from_iter(expected[500..1500].iter().copied())
        );

        // Slice across second and third chunk.
        let slice = encoded.slice(1000..2000).unwrap();
        assert_arrays_eq!(
            slice,
            PrimitiveArray::from_iter(expected[1000..2000].iter().copied())
        );
    }
}
