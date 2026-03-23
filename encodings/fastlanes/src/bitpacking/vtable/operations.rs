// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::bitpack_decompress;

impl OperationsVTable<BitPacked> for BitPacked {
    fn scalar_at(array: &BitPackedArray, index: usize) -> VortexResult<Scalar> {
        // NOTE(aduffy): this is the only code path in `BitPackedArray` that handles interior
        //  patches. All other compute goes through the execute/optimize pipeline which will
        //  convert the interior Patches into a wrapping `PatchedArray` instead.
        Ok(
            if let Some(patches) = array.patches.as_ref()
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
    use std::sync::LazyLock;

    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::assert_nth_scalar;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_array::vtable::VTable;
    use vortex_buffer::buffer;

    use crate::BitPacked;
    use crate::BitPackedArray;
    use crate::bitpack_compress::BitPackEncoder;

    static SESSION: LazyLock<vortex_session::VortexSession> =
        LazyLock::new(|| vortex_session::VortexSession::empty().with::<ArraySession>());

    fn slice_via_kernel(array: &BitPackedArray, range: Range<usize>) -> BitPackedArray {
        let slice_array = SliceArray::new(array.clone().into_array(), range);
        let mut ctx = SESSION.create_execution_ctx();
        let sliced =
            <BitPacked as VTable>::execute_parent(array, &slice_array.into_array(), 0, &mut ctx)
                .expect("execute_parent failed")
                .expect("expected slice kernel to execute");
        sliced.as_::<BitPacked>().clone()
    }

    #[test]
    pub fn slice_block() {
        let arr = BitPackEncoder::new(&PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)))
            .with_bit_width(6)
            .pack()
            .unwrap()
            .into_packed();
        let sliced = slice_via_kernel(&arr, 1024..2048);
        assert_nth_scalar!(sliced, 0, 1024u32 % 64);
        assert_nth_scalar!(sliced, 1023, 2047u32 % 64);
        assert_eq!(sliced.offset(), 0);
        assert_eq!(sliced.len(), 1024);
    }

    #[test]
    pub fn slice_within_block() {
        let arr = BitPackEncoder::new(&PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)))
            .with_bit_width(6)
            .pack()
            .unwrap()
            .into_packed();

        let sliced = slice_via_kernel(&arr, 512..1434);
        assert_nth_scalar!(sliced, 0, 512u32 % 64);
        assert_nth_scalar!(sliced, 921, 1433u32 % 64);
        assert_eq!(sliced.offset(), 512);
        assert_eq!(sliced.len(), 922);
    }

    #[test]
    fn slice_within_block_u8s() {
        let packed = BitPackEncoder::new(&PrimitiveArray::from_iter(
            (0..10_000).map(|i| (i % 63) as u8),
        ))
        .with_bit_width(7)
        .pack()
        .unwrap()
        .into_packed();

        let compressed = packed.slice(768..9999).unwrap();
        assert_nth_scalar!(compressed, 0, (768 % 63) as u8);
        assert_nth_scalar!(compressed, compressed.len() - 1, (9998 % 63) as u8);
    }

    #[test]
    fn slice_block_boundary_u8s() {
        let packed = BitPackEncoder::new(&PrimitiveArray::from_iter(
            (0..10_000).map(|i| (i % 63) as u8),
        ))
        .with_bit_width(7)
        .pack()
        .unwrap()
        .into_packed();

        let compressed = packed.slice(7168..9216).unwrap();
        assert_nth_scalar!(compressed, 0, (7168 % 63) as u8);
        assert_nth_scalar!(compressed, compressed.len() - 1, (9215 % 63) as u8);
    }

    #[test]
    fn double_slice_within_block() {
        let arr = BitPackEncoder::new(&PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)))
            .with_bit_width(6)
            .pack()
            .unwrap()
            .into_packed();
        let sliced = slice_via_kernel(&arr, 512..1434);
        assert_nth_scalar!(sliced, 0, 512u32 % 64);
        assert_nth_scalar!(sliced, 921, 1433u32 % 64);
        assert_eq!(sliced.offset(), 512);
        assert_eq!(sliced.len(), 922);
        let doubly_sliced = slice_via_kernel(&sliced, 127..911);
        assert_nth_scalar!(doubly_sliced, 0, (512u32 + 127) % 64);
        assert_nth_scalar!(doubly_sliced, 783, (512u32 + 910) % 64);
        assert_eq!(doubly_sliced.offset(), 639);
        assert_eq!(doubly_sliced.len(), 784);
    }

    #[test]
    fn take_after_slice() {
        // Check that our take implementation respects the offsets applied after slicing.

        let array = BitPackEncoder::new(&PrimitiveArray::from_iter((63u32..).take(3072)))
            .with_bit_width(6)
            .pack()
            .unwrap()
            .into_array()
            .unwrap()
            .as_::<BitPacked>()
            .clone();

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
}
