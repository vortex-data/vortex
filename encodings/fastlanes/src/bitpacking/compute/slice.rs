use std::cmp::max;

use vortex_array::array::SparseArray;
use vortex_array::compute::{slice, SliceFn};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::{VortexExpect, VortexResult};

use crate::{BitPackedArray, BitPackedEncoding};

impl SliceFn<BitPackedArray> for BitPackedEncoding {
    fn slice(&self, array: &BitPackedArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let offset_start = start + array.offset() as usize;
        let offset_stop = stop + array.offset() as usize;
        let offset = offset_start % 1024;
        let block_start = max(0, offset_start - offset);
        let block_stop = ((offset_stop + 1023) / 1024) * 1024;

        let encoded_start = (block_start / 8) * array.bit_width() as usize;
        let encoded_stop = (block_stop / 8) * array.bit_width() as usize;
        // slice the buffer using the encoded start/stop values
        BitPackedArray::try_new_from_offset(
            array.packed().slice(encoded_start..encoded_stop),
            array.ptype(),
            array.validity().slice(start, stop)?,
            array
                .patches()
                .map(|p| slice(&p, start, stop))
                .transpose()?
                .filter(|a| {
                    // If the sliced patch_indices is empty, we should not propagate the patches.
                    // There may be other logic that depends on Some(patches) indicating non-empty.
                    !SparseArray::try_from(a.clone())
                        .vortex_expect("BitPackedArray must encode patches as SparseArray")
                        .indices()
                        .is_empty()
                }),
            array.bit_width(),
            stop - start,
            offset as u16,
        )
        .map(|a| a.into_array())
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::array::{PrimitiveArray, SparseArray};
    use vortex_array::compute::unary::scalar_at;
    use vortex_array::compute::{slice, take, TakeOptions};
    use vortex_array::{ArrayLen, IntoArrayData};

    use crate::BitPackedArray;

    #[test]
    pub fn slice_block() {
        let arr = BitPackedArray::encode(
            PrimitiveArray::from((0u32..2048).map(|v| v % 64).collect::<Vec<_>>()).as_ref(),
            6,
        )
        .unwrap()
        .into_array();
        let sliced = BitPackedArray::try_from(slice(&arr, 1024, 2048).unwrap()).unwrap();
        assert_eq!(
            scalar_at(sliced.as_ref(), 0).unwrap(),
            (1024u32 % 64).into()
        );
        assert_eq!(
            scalar_at(sliced.as_ref(), 1023).unwrap(),
            (2047u32 % 64).into()
        );
        assert_eq!(sliced.offset(), 0);
        assert_eq!(sliced.len(), 1024);
    }

    #[test]
    pub fn slice_within_block() {
        let arr = BitPackedArray::encode(
            PrimitiveArray::from((0u32..2048).map(|v| v % 64).collect::<Vec<_>>()).as_ref(),
            6,
        )
        .unwrap()
        .into_array();
        let sliced = BitPackedArray::try_from(slice(&arr, 512, 1434).unwrap()).unwrap();
        assert_eq!(scalar_at(sliced.as_ref(), 0).unwrap(), (512u32 % 64).into());
        assert_eq!(
            scalar_at(sliced.as_ref(), 921).unwrap(),
            (1433u32 % 64).into()
        );
        assert_eq!(sliced.offset(), 512);
        assert_eq!(sliced.len(), 922);
    }

    #[test]
    fn slice_within_block_u8s() {
        let packed = BitPackedArray::encode(
            PrimitiveArray::from((0..10_000).map(|i| (i % 63) as u8).collect::<Vec<_>>()).as_ref(),
            7,
        )
        .unwrap();

        let compressed = slice(packed.as_ref(), 768, 9999).unwrap();
        assert_eq!(
            scalar_at(&compressed, 0).unwrap(),
            ((768 % 63) as u8).into()
        );
        assert_eq!(
            scalar_at(&compressed, compressed.len() - 1).unwrap(),
            ((9998 % 63) as u8).into()
        );
    }

    #[test]
    fn slice_block_boundary_u8s() {
        let packed = BitPackedArray::encode(
            PrimitiveArray::from((0..10_000).map(|i| (i % 63) as u8).collect::<Vec<_>>()).as_ref(),
            7,
        )
        .unwrap();

        let compressed = slice(packed.as_ref(), 7168, 9216).unwrap();
        assert_eq!(
            scalar_at(&compressed, 0).unwrap(),
            ((7168 % 63) as u8).into()
        );
        assert_eq!(
            scalar_at(&compressed, compressed.len() - 1).unwrap(),
            ((9215 % 63) as u8).into()
        );
    }

    #[test]
    fn double_slice_within_block() {
        let arr = BitPackedArray::encode(
            PrimitiveArray::from((0u32..2048).map(|v| v % 64).collect::<Vec<_>>()).as_ref(),
            6,
        )
        .unwrap()
        .into_array();
        let sliced = BitPackedArray::try_from(slice(&arr, 512, 1434).unwrap()).unwrap();
        assert_eq!(scalar_at(sliced.as_ref(), 0).unwrap(), (512u32 % 64).into());
        assert_eq!(
            scalar_at(sliced.as_ref(), 921).unwrap(),
            (1433u32 % 64).into()
        );
        assert_eq!(sliced.offset(), 512);
        assert_eq!(sliced.len(), 922);
        let doubly_sliced =
            BitPackedArray::try_from(slice(sliced.as_ref(), 127, 911).unwrap()).unwrap();
        assert_eq!(
            scalar_at(doubly_sliced.as_ref(), 0).unwrap(),
            ((512u32 + 127) % 64).into()
        );
        assert_eq!(
            scalar_at(doubly_sliced.as_ref(), 783).unwrap(),
            ((512u32 + 910) % 64).into()
        );
        assert_eq!(doubly_sliced.offset(), 639);
        assert_eq!(doubly_sliced.len(), 784);
    }

    #[test]
    fn slice_empty_patches() {
        // We create an array that has 1 element that does not fit in the 6-bit range.
        let array =
            BitPackedArray::encode(PrimitiveArray::from((0u32..=64).collect_vec()).as_ref(), 6)
                .unwrap();

        assert!(array.patches().is_some());

        let patch_indices = SparseArray::try_from(array.patches().unwrap())
            .unwrap()
            .indices();
        assert_eq!(patch_indices.len(), 1);

        // Slicing drops the empty patches array.
        let sliced = slice(array, 0, 64).unwrap();
        let sliced_bp = BitPackedArray::try_from(sliced).unwrap();
        assert!(sliced_bp.patches().is_none());
    }

    #[test]
    fn take_after_slice() {
        // Check that our take implementation respects the offsets applied after slicing.

        let array = BitPackedArray::encode(
            PrimitiveArray::from((63u32..).take(3072).collect_vec()).as_ref(),
            6,
        )
        .unwrap();

        // Slice the array.
        // The resulting array will still have 3 1024-element chunks.
        let sliced = slice(array.as_ref(), 922, 2061).unwrap();

        // Take one element from each chunk.
        // Chunk 1: physical indices  922-1023, logical indices    0-101
        // Chunk 2: physical indices 1024-2047, logical indices  102-1125
        // Chunk 3: physical indices 2048-2060, logical indices 1126-1138

        let taken = take(
            &sliced,
            PrimitiveArray::from(vec![101i64, 1125i64, 1138i64]).as_ref(),
            TakeOptions::default(),
        )
        .unwrap();
        assert_eq!(taken.len(), 3);
    }
}
