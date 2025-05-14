use std::cmp::max;
use std::ops::Range;

use itertools::Itertools;
use lending_iterator::LendingIterator;
use num_traits::AsPrimitive;
use vortex_array::arrays::{IS_CONST_LANE_WIDTH, PrimitiveArray, compute_is_constant};
use vortex_array::vtable::{OperationsVTable, ValidityHelper};
use vortex_array::{ArrayRef, Cost, IntoArray, ToCanonical};
use vortex_dtype::{NativePType, match_each_integer_ptype, match_each_unsigned_integer_ptype};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::unpack_iter::BitPacked;
use crate::{BitPackedArray, BitPackedVTable, unpack_single};

impl OperationsVTable<BitPackedVTable> for BitPackedVTable {
    fn slice(array: &BitPackedArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let offset_start = start + array.offset() as usize;
        let offset_stop = stop + array.offset() as usize;
        let offset = offset_start % 1024;
        let block_start = max(0, offset_start - offset);
        let block_stop = offset_stop.div_ceil(1024) * 1024;

        let encoded_start = (block_start / 8) * array.bit_width() as usize;
        let encoded_stop = (block_stop / 8) * array.bit_width() as usize;

        // slice the buffer using the encoded start/stop values
        // SAFETY: the invariants of the original BitPackedArray are preserved when slicing.
        unsafe {
            BitPackedArray::new_unchecked_with_offset(
                array.packed().slice(encoded_start..encoded_stop),
                array.ptype(),
                array.validity().slice(start, stop)?,
                array
                    .patches()
                    .map(|p| p.slice(start, stop))
                    .transpose()?
                    .flatten(),
                array.bit_width(),
                stop - start,
                offset as u16,
            )
        }
        .map(|a| a.into_array())
    }

    fn scalar_at(array: &BitPackedArray, index: usize) -> VortexResult<Scalar> {
        if let Some(patches) = array.patches() {
            if let Some(patch) = patches.get_patched(index)? {
                return Ok(patch);
            }
        }
        unpack_single(array, index)?.cast(array.dtype())
    }

    fn is_constant(array: &BitPackedArray, cost: Cost) -> VortexResult<Option<bool>> {
        if cost.is_negligible() {
            return Ok(None);
        }
        match_each_integer_ptype!(array.ptype(), |$P| {
            bitpacked_is_constant::<$P, {IS_CONST_LANE_WIDTH / size_of::<$P>()}>(array)
        })
        .map(Some)
    }
}

fn bitpacked_is_constant<T: BitPacked, const WIDTH: usize>(
    array: &BitPackedArray,
) -> VortexResult<bool> {
    let mut bit_unpack_iterator = array.unpacked_chunks::<T>();
    let patches = array
        .patches()
        .map(|p| {
            let values = p.values().to_primitive()?;
            let indices = p.indices().to_primitive()?;
            let offset = p.offset();
            VortexResult::Ok((indices, values, offset))
        })
        .transpose()?;

    let mut header_constant_value = None;
    let mut current_idx = 0;
    if let Some(header) = bit_unpack_iterator.initial() {
        if let Some((indices, patches, offset)) = &patches {
            apply_patches(
                header,
                current_idx..header.len(),
                indices,
                patches.as_slice::<T>(),
                *offset,
            )
        }

        if !compute_is_constant::<_, WIDTH>(header) {
            return Ok(false);
        }
        header_constant_value = Some(header[0]);
        current_idx = header.len();
    }

    let mut first_chunk_value = None;
    let mut chunks_iter = bit_unpack_iterator.full_chunks();
    while let Some(chunk) = chunks_iter.next() {
        if let Some((indices, patches, offset)) = &patches {
            let chunk_len = chunk.len();
            apply_patches(
                chunk,
                current_idx..current_idx + chunk_len,
                indices,
                patches.as_slice::<T>(),
                *offset,
            )
        }

        if !compute_is_constant::<_, WIDTH>(chunk) {
            return Ok(false);
        }

        if let Some(chunk_value) = first_chunk_value {
            if chunk_value != chunk[0] {
                return Ok(false);
            }
        } else {
            if let Some(header_value) = header_constant_value {
                if header_value != chunk[0] {
                    return Ok(false);
                }
            }
            first_chunk_value = Some(chunk[0]);
        }

        current_idx += chunk.len();
    }

    if let Some(trailer) = bit_unpack_iterator.trailer() {
        if let Some((indices, patches, offset)) = &patches {
            let chunk_len = trailer.len();
            apply_patches(
                trailer,
                current_idx..current_idx + chunk_len,
                indices,
                patches.as_slice::<T>(),
                *offset,
            )
        }

        if !compute_is_constant::<_, WIDTH>(trailer) {
            return Ok(false);
        }

        if let Some(previous_const_value) = header_constant_value.or(first_chunk_value) {
            if previous_const_value != trailer[0] {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

fn apply_patches<T: BitPacked>(
    values: &mut [T],
    values_range: Range<usize>,
    patch_indices: &PrimitiveArray,
    patch_values: &[T],
    indices_offset: usize,
) {
    match_each_unsigned_integer_ptype!(patch_indices.ptype(), |$I| {
        apply_patches_idx_typed(values, values_range, patch_indices.as_slice::<$I>(), patch_values, indices_offset)
    });
}

fn apply_patches_idx_typed<T: BitPacked, I: NativePType + AsPrimitive<usize>>(
    values: &mut [T],
    values_range: Range<usize>,
    patch_indices: &[I],
    patch_values: &[T],
    indices_offset: usize,
) {
    for (i, &v) in patch_indices
        .iter()
        .map(|i| i.as_() - indices_offset)
        .zip_eq(patch_values)
        .skip_while(|(i, _)| i < &values_range.start)
        .take_while(|(i, _)| i < &values_range.end)
    {
        values[i - values_range.start] = v
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

    use crate::{BitPackedArray, BitPackedVTable};

    #[test]
    pub fn slice_block() {
        let arr = BitPackedArray::encode(
            PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)).as_ref(),
            6,
        )
        .unwrap();
        let sliced = arr
            .slice(1024, 2048)
            .unwrap()
            .as_::<BitPackedVTable>()
            .clone();
        assert_eq!(sliced.scalar_at(0).unwrap(), (1024u32 % 64).into());
        assert_eq!(sliced.scalar_at(1023).unwrap(), (2047u32 % 64).into());
        assert_eq!(sliced.offset(), 0);
        assert_eq!(sliced.len(), 1024);
    }

    #[test]
    pub fn slice_within_block() {
        let arr = BitPackedArray::encode(
            PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)).as_ref(),
            6,
        )
        .unwrap()
        .into_array();
        let sliced = arr
            .slice(512, 1434)
            .unwrap()
            .as_::<BitPackedVTable>()
            .clone();
        assert_eq!(sliced.scalar_at(0).unwrap(), (512u32 % 64).into());
        assert_eq!(sliced.scalar_at(921).unwrap(), (1433u32 % 64).into());
        assert_eq!(sliced.offset(), 512);
        assert_eq!(sliced.len(), 922);
    }

    #[test]
    fn slice_within_block_u8s() {
        let packed = BitPackedArray::encode(
            PrimitiveArray::from_iter((0..10_000).map(|i| (i % 63) as u8)).as_ref(),
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
            PrimitiveArray::from_iter((0..10_000).map(|i| (i % 63) as u8)).as_ref(),
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
        let arr = BitPackedArray::encode(
            PrimitiveArray::from_iter((0u32..2048).map(|v| v % 64)).as_ref(),
            6,
        )
        .unwrap()
        .into_array();
        let sliced = arr
            .slice(512, 1434)
            .unwrap()
            .as_::<BitPackedVTable>()
            .clone();
        assert_eq!(sliced.scalar_at(0).unwrap(), (512u32 % 64).into());
        assert_eq!(sliced.scalar_at(921).unwrap(), (1433u32 % 64).into());
        assert_eq!(sliced.offset(), 512);
        assert_eq!(sliced.len(), 922);
        let doubly_sliced = sliced
            .slice(127, 911)
            .unwrap()
            .as_::<BitPackedVTable>()
            .clone();
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
        let array =
            BitPackedArray::encode(PrimitiveArray::from_iter(0u32..=64).as_ref(), 6).unwrap();

        assert!(array.patches().is_some());

        let patch_indices = array.patches().unwrap().indices().clone();
        assert_eq!(patch_indices.len(), 1);

        // Slicing drops the empty patches array.
        let sliced = array.slice(0, 64).unwrap();
        let sliced_bp = sliced.as_::<BitPackedVTable>();
        assert!(sliced_bp.patches().is_none());
    }

    #[test]
    fn take_after_slice() {
        // Check that our take implementation respects the offsets applied after slicing.

        let array =
            BitPackedArray::encode(PrimitiveArray::from_iter((63u32..).take(3072)).as_ref(), 6)
                .unwrap();

        // Slice the array.
        // The resulting array will still have 3 1024-element chunks.
        let sliced = array.slice(922, 2061).unwrap();

        // Take one element from each chunk.
        // Chunk 1: physical indices  922-1023, logical indices    0-101
        // Chunk 2: physical indices 1024-2047, logical indices  102-1125
        // Chunk 3: physical indices 2048-2060, logical indices 1126-1138

        let taken = take(
            &sliced,
            PrimitiveArray::from_iter([101i64, 1125, 1138]).as_ref(),
        )
        .unwrap();
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

    #[test]
    fn is_constant_with_patches() {
        let array = BitPackedArray::encode(&buffer![4; 1025].into_array(), 2).unwrap();
        assert!(array.is_constant());
    }
}
