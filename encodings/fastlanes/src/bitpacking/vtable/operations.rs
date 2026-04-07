// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::bitpack_decompress;
use crate::bitpacking::array::BitPackedArrayExt;
impl OperationsVTable<BitPacked> for BitPacked {
    fn scalar_at(
        array: ArrayView<'_, BitPacked>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(
            if let Some(patches) = array.patches()
                && let Some(patch) = patches.get_patched(index)?
            {
                patch
            } else {
                bitpack_decompress::unpack_single(array, index)
            },
        )
    }
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::patches::Patches;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Alignment;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;

    use crate::BitPacked;
    use crate::BitPackedArray;
    use crate::BitPackedData;
    use crate::bitpacking::array::BitPackedArrayExt;

    fn bp(array: &ArrayRef, bit_width: u8) -> BitPackedArray {
        BitPackedData::encode(array, bit_width).unwrap()
    }

    fn slice_via_reduce(array: &BitPackedArray, range: Range<usize>) -> BitPackedArray {
        let array_ref = array.clone().into_array();
        let slice_array = SliceArray::new(array_ref.clone(), range);
        let sliced = array_ref
            .reduce_parent(&slice_array.into_array(), 0)
            .expect("execute_parent failed")
            .expect("expected slice kernel to execute");
        sliced.as_::<BitPacked>().into_owned()
    }

    #[test]
    pub fn slice_block() {
        let arr = bp(
            &PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)).into_array(),
            6,
        );
        let sliced = slice_via_reduce(&arr, 1024..2048);
        assert_nth_scalar!(sliced, 0, 1024u32 % 64);
        assert_nth_scalar!(sliced, 1023, 2047u32 % 64);
        assert_eq!(sliced.offset(), 0);
        assert_eq!(sliced.len(), 1024);
    }

    #[test]
    pub fn slice_within_block() {
        let arr = bp(
            &PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)).into_array(),
            6,
        );
        let sliced = slice_via_reduce(&arr, 512..1434);
        assert_nth_scalar!(sliced, 0, 512u32 % 64);
        assert_nth_scalar!(sliced, 921, 1433u32 % 64);
        assert_eq!(sliced.offset(), 512);
        assert_eq!(sliced.len(), 922);
    }

    #[test]
    fn slice_within_block_u8s() {
        let packed = bp(
            &PrimitiveArray::from_iter((0..10_000).map(|i| (i % 63) as u8)).into_array(),
            7,
        );

        let compressed = packed.slice(768..9999).unwrap();
        assert_nth_scalar!(compressed, 0, (768 % 63) as u8);
        assert_nth_scalar!(compressed, compressed.len() - 1, (9998 % 63) as u8);
    }

    #[test]
    fn slice_block_boundary_u8s() {
        let packed = bp(
            &PrimitiveArray::from_iter((0..10_000).map(|i| (i % 63) as u8)).into_array(),
            7,
        );

        let compressed = packed.slice(7168..9216).unwrap();
        assert_nth_scalar!(compressed, 0, (7168 % 63) as u8);
        assert_nth_scalar!(compressed, compressed.len() - 1, (9215 % 63) as u8);
    }

    #[test]
    fn double_slice_within_block() {
        let arr = bp(
            &PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)).into_array(),
            6,
        );
        let sliced = slice_via_reduce(&arr, 512..1434);
        assert_nth_scalar!(sliced, 0, 512u32 % 64);
        assert_nth_scalar!(sliced, 921, 1433u32 % 64);
        assert_eq!(sliced.offset(), 512);
        assert_eq!(sliced.len(), 922);
        let doubly_sliced = slice_via_reduce(&sliced, 127..911);
        assert_nth_scalar!(doubly_sliced, 0, (512u32 + 127) % 64);
        assert_nth_scalar!(doubly_sliced, 783, (512u32 + 910) % 64);
        assert_eq!(doubly_sliced.offset(), 639);
        assert_eq!(doubly_sliced.len(), 784);
    }

    #[test]
    fn slice_empty_patches() {
        // We create an array that has 1 element that does not fit in the 6-bit range.
        let array = BitPackedData::encode(&buffer![0u32..=64].into_array(), 6).unwrap();

        assert!(array.patches().is_some());

        let patch_indices = array.patches().unwrap().indices().clone();
        assert_eq!(patch_indices.len(), 1);

        // Slicing drops the empty patches array.
        let sliced_bp = slice_via_reduce(&array, 0..64);
        assert!(sliced_bp.patches().is_none());
    }

    #[test]
    fn take_after_slice() {
        // Check that our take implementation respects the offsets applied after slicing.

        let array = bp(
            &PrimitiveArray::from_iter((63u32..).take(3072)).into_array(),
            6,
        );

        // Slice the array.
        // The resulting array will still have 3 1024-element chunks.
        let sliced = array.slice(922..2061).unwrap();

        // Take one element from each chunk.
        // Chunk 1: physical indices  922-1023, logical indices    0-101
        // Chunk 2: physical indices 1024-2047, logical indices  102-1125
        // Chunk 3: physical indices 2048-2060, logical indices 1126-1138

        let taken = sliced
            .take(buffer![101i64, 1125, 1138].into_array())
            .unwrap();
        assert_eq!(taken.len(), 3);
    }

    #[test]
    fn scalar_at_invalid_patches() {
        let packed_array = BitPacked::try_new(
            BufferHandle::new_host(ByteBuffer::copy_from_aligned(
                [0u8; 128],
                Alignment::of::<u32>(),
            )),
            PType::U32,
            Validity::AllInvalid,
            Some(
                Patches::new(
                    8,
                    0,
                    buffer![1u32].into_array(),
                    PrimitiveArray::new(buffer![999u32], Validity::AllValid).into_array(),
                    None,
                )
                .unwrap(),
            ),
            1,
            8,
            0,
        )
        .unwrap()
        .into_array();
        assert_eq!(
            packed_array.scalar_at(1).unwrap(),
            Scalar::null(DType::Primitive(PType::U32, Nullability::Nullable))
        );
    }

    #[test]
    fn scalar_at() {
        let values = (0u32..257).collect::<Buffer<_>>();
        let uncompressed = values.clone().into_array();
        let packed = BitPackedData::encode(&uncompressed, 8).unwrap();
        assert!(packed.patches().is_some());

        let patches = packed.patches().unwrap().indices().clone();
        assert_eq!(
            usize::try_from(&patches.scalar_at(0).unwrap()).unwrap(),
            256
        );

        let expected = PrimitiveArray::from_iter(values.iter().copied());
        assert_arrays_eq!(packed, expected);
    }
}
