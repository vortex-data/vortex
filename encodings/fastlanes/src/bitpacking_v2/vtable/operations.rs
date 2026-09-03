// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use crate::BitPackedV2;
use crate::bitpacking_v2::array::BitPackedV2ArrayExt;
use crate::bitpacking_v2::bitpack_decompress;
impl OperationsVTable<BitPackedV2> for BitPackedV2 {
    fn scalar_at(
        array: ArrayView<'_, BitPackedV2>,
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
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
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

    use crate::BitPackedV2;
    use crate::BitPackedV2Array;
    use crate::BitPackedV2Data;
    use crate::ChunkWidths;
    use crate::bitpacking_v2::array::BitPackedV2ArrayExt;
    use crate::test::SESSION;

    fn bp(array: &ArrayRef, bit_width: u8) -> BitPackedV2Array {
        BitPackedV2Data::encode(array, bit_width, &mut SESSION.create_execution_ctx()).unwrap()
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
        let packed_array = BitPackedV2::try_new(
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
            ChunkWidths::uniform(1, 1),
            8,
            0,
        )
        .unwrap()
        .into_array();
        assert_eq!(
            packed_array
                .execute_scalar(1, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::null(DType::Primitive(PType::U32, Nullability::Nullable))
        );
    }

    #[test]
    fn scalar_at() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0u32..257).collect::<Buffer<_>>();
        let uncompressed = values.clone().into_array();
        let packed = BitPackedV2Data::encode(&uncompressed, 8, &mut ctx).unwrap();
        assert!(packed.patches().is_some());

        let patches = packed.patches().unwrap().indices().clone();
        assert_eq!(
            usize::try_from(
                &patches
                    .execute_scalar(0, &mut SESSION.create_execution_ctx())
                    .unwrap()
            )
            .unwrap(),
            256
        );

        let expected = PrimitiveArray::from_iter(values.iter().copied());
        assert_arrays_eq!(packed, expected, &mut ctx);
    }
}
