// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;

use fastlanes::FoR;
use num_traits::{PrimInt, WrappingAdd, WrappingSub};
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::stats::Stat;
use vortex_array::vtable::ValidityHelper;
use vortex_array::{IntoArray, ToCanonical};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{
    NativePType, PhysicalPType, UnsignedPType, match_each_integer_ptype,
    match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_scalar::FromPrimitiveOrF16;

use crate::{BitPackedArray, BitPackedVTable, FoRArray};

impl FoRArray {
    pub fn encode(array: PrimitiveArray) -> VortexResult<FoRArray> {
        let min = array
            .statistics()
            .compute_stat(Stat::Min)?
            .ok_or_else(|| vortex_err!("Min stat not found"))?;

        let encoded = match_each_integer_ptype!(array.ptype(), |T| {
            compress_primitive::<T>(array, T::try_from(&min)?)?.into_array()
        });
        FoRArray::try_new(encoded, min)
    }
}

#[allow(clippy::cast_possible_truncation)]
fn compress_primitive<T: NativePType + WrappingSub + PrimInt>(
    parray: PrimitiveArray,
    min: T,
) -> VortexResult<PrimitiveArray> {
    // Set null values to the min value, ensuring that decompress into a value in the primitive
    // range (and stop them wrapping around)
    parray.map_each_with_validity::<T, _, _>(|(v, bool)| {
        if bool {
            v.wrapping_sub(&min)
        } else {
            T::zero()
        }
    })
}

pub fn decompress(array: &FoRArray) -> PrimitiveArray {
    let ptype = array.ptype();

    // try to do fused unpack
    if array.dtype().is_unsigned_int()
        && let Some(bp) = array.encoded().as_opt::<BitPackedVTable>()
        && bp.patches().is_none()
        && bp.all_valid()
    {
        return match_each_unsigned_integer_ptype!(array.ptype(), |T| {
            fused_decompress::<T>(array, bp)
        });
    }

    // TODO(ngates): do we need this to be into_encoded() somehow?
    let encoded = array.encoded().to_primitive();
    let validity = encoded.validity().clone();

    match_each_integer_ptype!(ptype, |T| {
        let min = array
            .reference_scalar()
            .as_primitive()
            .typed_value::<T>()
            .vortex_expect("reference must be non-null");
        if min == 0 {
            encoded
        } else {
            PrimitiveArray::new(
                decompress_primitive(encoded.into_buffer_mut::<T>(), min),
                validity,
            )
        }
    })
}

fn fused_decompress<T: PhysicalPType + UnsignedPType + FoR + FromPrimitiveOrF16>(
    for_: &FoRArray,
    bp: &BitPackedArray,
) -> PrimitiveArray {
    const CHUNK_SIZE: usize = 1024;

    let offset = bp.offset() as usize;
    let len = bp.len();
    let bit_width = bp.bit_width() as usize;
    let elems_per_chunk = 128 * bit_width / size_of::<T>();
    let num_chunks = (offset + len).div_ceil(CHUNK_SIZE);
    let last_chunk_length = (offset + len) % CHUNK_SIZE;

    let mut builder = PrimitiveBuilder::<T>::with_capacity(for_.dtype().nullability(), len);
    let mut uninit_range = builder.uninit_range(len);

    let packed_buffer = Buffer::<T>::from_byte_buffer(bp.packed().clone());
    let packed_slice = packed_buffer.as_slice();
    let ref_ = for_
        .reference
        .as_primitive()
        .as_::<T>()
        .vortex_expect("cannot be null");

    let first_chunk_is_sliced = offset != 0;
    let last_chunk_is_sliced = last_chunk_length != 0;

    // Shared temp buffer for partial chunks
    let mut temp_buffer = [mem::MaybeUninit::<T>::uninit(); CHUNK_SIZE];

    // Track position in output relative to the start of the UninitRange.
    let mut local_idx = 0;

    // Handle initial partial chunk if offset != 0 or if there's only one chunk
    // # Safety
    //
    // See `unpack_iter.rs`.
    if first_chunk_is_sliced || num_chunks == 1 {
        let chunk = &packed_slice[..elems_per_chunk];

        unsafe {
            let dst: &mut [T] = mem::transmute(&mut temp_buffer[..]);
            FoR::unchecked_unfor_pack(bit_width, chunk, ref_, dst);

            let header_end_slice = if num_chunks == 1 {
                len
            } else {
                CHUNK_SIZE - offset
            };

            // Copy the relevant portion to output
            let src = mem::transmute::<&[mem::MaybeUninit<T>], &[T]>(
                &temp_buffer[offset..][..header_end_slice],
            );
            let dst = uninit_range.slice_uninit_mut(local_idx, header_end_slice);
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.as_mut_ptr() as *mut T,
                header_end_slice,
            );

            local_idx += header_end_slice;
        }
    }

    // Handle full middle chunks
    if num_chunks > 1 {
        let full_chunks_start = if first_chunk_is_sliced { 1 } else { 0 };
        let full_chunks_end = num_chunks - (last_chunk_is_sliced as usize);

        for i in full_chunks_start..full_chunks_end {
            let chunk = &packed_slice[i * elems_per_chunk..][..elems_per_chunk];

            unsafe {
                let uninit_dst = uninit_range.slice_uninit_mut(local_idx, CHUNK_SIZE);
                let dst: &mut [T] = mem::transmute(uninit_dst);
                FoR::unchecked_unfor_pack(bit_width, chunk, ref_, dst);
            }
            local_idx += CHUNK_SIZE;
        }
    }

    // Handle trailing partial chunk if len % 1024 != 0
    if last_chunk_is_sliced && num_chunks > 1 {
        let chunk = &packed_slice[(num_chunks - 1) * elems_per_chunk..][..elems_per_chunk];

        unsafe {
            let dst: &mut [T] = mem::transmute(&mut temp_buffer[..]);
            FoR::unchecked_unfor_pack(bit_width, chunk, ref_, dst);

            // Copy only the valid portion to output
            let src =
                mem::transmute::<&[mem::MaybeUninit<T>], &[T]>(&temp_buffer[..last_chunk_length]);
            let dst = uninit_range.slice_uninit_mut(local_idx, last_chunk_length);
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.as_mut_ptr() as *mut T,
                last_chunk_length,
            );
        }
    }

    unsafe {
        uninit_range.finish();
    }

    builder.finish_into_primitive()
}

fn decompress_primitive<T: NativePType + WrappingAdd + PrimInt>(
    values: BufferMut<T>,
    min: T,
) -> Buffer<T> {
    values.map_each(move |v| v.wrapping_add(&min)).freeze()
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::ToCanonical;
    use vortex_array::stats::StatsProvider;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_dtype::PType;
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    fn test_compress_round_trip_small() {
        let array = PrimitiveArray::new((1i32..10).collect::<Buffer<_>>(), Validity::NonNullable);
        let compressed = FoRArray::encode(array.clone()).unwrap();
        assert_eq!(i32::try_from(compressed.reference_scalar()).unwrap(), 1);

        let decompressed = compressed.to_primitive();
        assert_eq!(decompressed.as_slice::<i32>(), array.as_slice::<i32>());
    }

    #[test]
    fn test_compress() {
        // Create a range offset by a million
        let array = PrimitiveArray::new(
            (0u32..10_000).map(|v| v + 1_000_000).collect::<Buffer<_>>(),
            Validity::NonNullable,
        );
        let compressed = FoRArray::encode(array).unwrap();
        assert_eq!(
            u32::try_from(compressed.reference_scalar()).unwrap(),
            1_000_000u32
        );
    }

    #[test]
    fn test_zeros() {
        let array = PrimitiveArray::new(buffer![0i32; 100], Validity::NonNullable);
        assert_eq!(array.statistics().len(), 0);

        let dtype = array.dtype().clone();
        let compressed = FoRArray::encode(array).unwrap();
        assert_eq!(compressed.dtype(), &dtype);
        assert!(compressed.dtype().is_signed_int());
        assert!(compressed.encoded().dtype().is_signed_int());

        let constant = compressed.encoded().as_constant().unwrap();
        assert_eq!(constant, Scalar::from(0i32));
    }

    #[test]
    fn test_decompress() {
        // Create a range offset by a million
        let array = PrimitiveArray::from_iter((0u32..100_000).step_by(1024).map(|v| v + 1_000_000));
        let compressed = FoRArray::encode(array.clone()).unwrap();
        let decompressed = compressed.to_primitive();
        assert_eq!(decompressed.as_slice::<u32>(), array.as_slice::<u32>());
    }

    #[test]
    fn test_decompress_fused() {
        // Create a range offset by a million
        let expect = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7 + 10));
        let array = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7));
        let bp = BitPackedArray::encode(array.as_ref(), 3).unwrap();
        let compressed = FoRArray::try_new(bp.into_array(), 10u32.into()).unwrap();
        let decompressed = compressed.to_primitive();
        assert_eq!(decompressed.as_slice::<u32>(), expect.as_slice::<u32>());
    }

    #[test]
    fn test_overflow() {
        let array = PrimitiveArray::from_iter(i8::MIN..=i8::MAX);
        let compressed = FoRArray::encode(array.clone()).unwrap();
        assert_eq!(
            i8::MIN,
            compressed
                .reference_scalar()
                .as_primitive()
                .typed_value::<i8>()
                .unwrap()
        );

        let encoded = compressed
            .encoded()
            .to_primitive()
            .reinterpret_cast(PType::U8);
        let encoded_bytes: &[u8] = encoded.as_slice::<u8>();
        let unsigned: Vec<u8> = (0..=u8::MAX).collect_vec();
        assert_eq!(encoded_bytes, unsigned.as_slice());

        let decompressed = compressed.to_primitive();
        assert_eq!(decompressed.as_slice::<i8>(), array.as_slice::<i8>());
        array
            .as_slice::<i8>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
                assert_eq!(*v, i8::try_from(compressed.scalar_at(i).as_ref()).unwrap());
            });
    }
}
