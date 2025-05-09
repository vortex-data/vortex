use std::cmp::max;

use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{BitPackedArray, unpack_single};

impl ArrayOperationsImpl for BitPackedArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let offset_start = start + self.offset() as usize;
        let offset_stop = stop + self.offset() as usize;
        let offset = offset_start % 1024;
        let block_start = max(0, offset_start - offset);
        let block_stop = offset_stop.div_ceil(1024) * 1024;

        let encoded_start = (block_start / 8) * self.bit_width() as usize;
        let encoded_stop = (block_stop / 8) * self.bit_width() as usize;

        // slice the buffer using the encoded start/stop values
        // SAFETY: the invariants of the original BitPackedArray are preserved when slicing.
        unsafe {
            BitPackedArray::new_unchecked_with_offset(
                self.packed().slice(encoded_start..encoded_stop),
                self.ptype(),
                self.validity().slice(start, stop)?,
                self.patches()
                    .map(|p| p.slice(start, stop))
                    .transpose()?
                    .flatten(),
                self.bit_width(),
                stop - start,
                offset as u16,
            )
        }
        .map(|a| a.into_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        if let Some(patches) = self.patches() {
            if let Some(patch) = patches.get_patched(index)? {
                return Ok(patch);
            }
        }
        unpack_single(self, index)?.cast(self.dtype())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::take;
    use vortex_array::patches::Patches;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray};
    use vortex_buffer::{Alignment, Buffer, ByteBuffer, buffer};
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::BitPackedArray;

    #[test]
    pub fn slice_block() {
        let arr =
            BitPackedArray::encode(&PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)), 6)
                .unwrap();
        let sliced = BitPackedArray::try_from(arr.slice(1024, 2048).unwrap()).unwrap();
        assert_eq!(sliced.scalar_at(0).unwrap(), (1024u32 % 64).into());
        assert_eq!(sliced.scalar_at(1023).unwrap(), (2047u32 % 64).into());
        assert_eq!(sliced.offset(), 0);
        assert_eq!(sliced.len(), 1024);
    }

    #[test]
    pub fn slice_within_block() {
        let arr =
            BitPackedArray::encode(&PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)), 6)
                .unwrap()
                .into_array();
        let sliced = BitPackedArray::try_from(arr.slice(512, 1434).unwrap()).unwrap();
        assert_eq!(sliced.scalar_at(0).unwrap(), (512u32 % 64).into());
        assert_eq!(sliced.scalar_at(921).unwrap(), (1433u32 % 64).into());
        assert_eq!(sliced.offset(), 512);
        assert_eq!(sliced.len(), 922);
    }

    #[test]
    fn slice_within_block_u8s() {
        let packed = BitPackedArray::encode(
            &PrimitiveArray::from_iter((0..10_000).map(|i| (i % 63) as u8)),
            7,
        )
        .unwrap();

        let compressed = packed.slice(768, 9999).unwrap();
        assert_eq!(compressed.scalar_at(0).unwrap(), ((768 % 63) as u8).into());
        assert_eq!(
            compressed.scalar_at(compressed.len() - 1).unwrap(),
            ((9998 % 63) as u8).into()
        );
    }

    #[test]
    fn slice_block_boundary_u8s() {
        let packed = BitPackedArray::encode(
            &PrimitiveArray::from_iter((0..10_000).map(|i| (i % 63) as u8)),
            7,
        )
        .unwrap();

        let compressed = packed.slice(7168, 9216).unwrap();
        assert_eq!(compressed.scalar_at(0).unwrap(), ((7168 % 63) as u8).into());
        assert_eq!(
            compressed.scalar_at(compressed.len() - 1).unwrap(),
            ((9215 % 63) as u8).into()
        );
    }

    #[test]
    fn double_slice_within_block() {
        let arr =
            BitPackedArray::encode(&PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)), 6)
                .unwrap()
                .into_array();
        let sliced = BitPackedArray::try_from(arr.slice(512, 1434).unwrap()).unwrap();
        assert_eq!(sliced.scalar_at(0).unwrap(), (512u32 % 64).into());
        assert_eq!(sliced.scalar_at(921).unwrap(), (1433u32 % 64).into());
        assert_eq!(sliced.offset(), 512);
        assert_eq!(sliced.len(), 922);
        let doubly_sliced = BitPackedArray::try_from(sliced.slice(127, 911).unwrap()).unwrap();
        assert_eq!(
            doubly_sliced.scalar_at(0).unwrap(),
            ((512u32 + 127) % 64).into()
        );
        assert_eq!(
            doubly_sliced.scalar_at(783).unwrap(),
            ((512u32 + 910) % 64).into()
        );
        assert_eq!(doubly_sliced.offset(), 639);
        assert_eq!(doubly_sliced.len(), 784);
    }

    #[test]
    fn slice_empty_patches() {
        // We create an array that has 1 element that does not fit in the 6-bit range.
        let array = BitPackedArray::encode(&PrimitiveArray::from_iter(0u32..=64), 6).unwrap();

        assert!(array.patches().is_some());

        let patch_indices = array.patches().unwrap().indices().clone();
        assert_eq!(patch_indices.len(), 1);

        // Slicing drops the empty patches array.
        let sliced = array.slice(0, 64).unwrap();
        let sliced_bp = BitPackedArray::try_from(sliced).unwrap();
        assert!(sliced_bp.patches().is_none());
    }

    #[test]
    fn take_after_slice() {
        // Check that our take implementation respects the offsets applied after slicing.

        let array =
            BitPackedArray::encode(&PrimitiveArray::from_iter((63u32..).take(3072)), 6).unwrap();

        // Slice the array.
        // The resulting array will still have 3 1024-element chunks.
        let sliced = array.slice(922, 2061).unwrap();

        // Take one element from each chunk.
        // Chunk 1: physical indices  922-1023, logical indices    0-101
        // Chunk 2: physical indices 1024-2047, logical indices  102-1125
        // Chunk 3: physical indices 2048-2060, logical indices 1126-1138

        let taken = take(&sliced, &PrimitiveArray::from_iter([101i64, 1125, 1138])).unwrap();
        assert_eq!(taken.len(), 3);
    }

    #[test]
    fn scalar_at_invalid_patches() {
        // SAFETY: using unsigned PType
        let packed_array = unsafe {
            BitPackedArray::new_unchecked(
                ByteBuffer::copy_from_aligned([0u8; 128], Alignment::of::<u32>()),
                PType::U32,
                Validity::AllInvalid,
                Some(Patches::new(
                    8,
                    0,
                    buffer![1u32].into_array(),
                    PrimitiveArray::new(buffer![999u32], Validity::AllValid).to_array(),
                )),
                1,
                8,
            )
        }
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
        let packed = BitPackedArray::encode(&uncompressed, 8).unwrap();
        assert!(packed.patches().is_some());

        let patches = packed.patches().unwrap().indices().clone();
        assert_eq!(
            usize::try_from(&patches.scalar_at(0).unwrap()).unwrap(),
            256
        );

        values.iter().enumerate().for_each(|(i, v)| {
            assert_eq!(
                u32::try_from(packed.scalar_at(i).unwrap().as_ref()).unwrap(),
                *v
            );
        });
    }
}
