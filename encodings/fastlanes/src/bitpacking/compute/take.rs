use fastlanes::BitPacking;
use vortex_array::array::PrimitiveArray;
use vortex_array::compute::{take, TakeFn};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{
    ArrayDType, ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant, IntoCanonical, ToArrayData,
};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{
    match_each_integer_ptype, match_each_unsigned_integer_ptype, NativePType, PType,
};
use vortex_error::{VortexExpect as _, VortexResult};

use super::chunked_indices;
use crate::{unpack_single_primitive, BitPackedArray, BitPackedEncoding};

// assuming the buffer is already allocated (which will happen at most once) then unpacking
// all 1024 elements takes ~8.8x as long as unpacking a single element on an M2 Macbook Air.
// see https://github.com/spiraldb/vortex/pull/190#issue-2223752833
pub(super) const UNPACK_CHUNK_THRESHOLD: usize = 8;

impl TakeFn<BitPackedArray> for BitPackedEncoding {
    fn take(&self, array: &BitPackedArray, indices: &ArrayData) -> VortexResult<ArrayData> {
        // If the indices are large enough, it's faster to flatten and take the primitive array.
        if indices.len() * UNPACK_CHUNK_THRESHOLD > array.len() {
            return take(array.clone().into_canonical()?.into_primitive()?, indices);
        }

        // NOTE: we use the unsigned PType because all values in the BitPackedArray must
        //  be non-negative (pre-condition of creating the BitPackedArray).
        let ptype: PType = PType::try_from(array.dtype())?;
        let validity = array.validity();
        let taken_validity = validity.take(indices)?;

        let indices = indices.clone().into_primitive()?;
        let taken = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                take_primitive::<$T, $I>(array, &indices, taken_validity)?
            })
        });
        Ok(taken.reinterpret_cast(ptype).into_array())
    }
}

fn take_primitive<T: NativePType + BitPacking, I: NativePType>(
    array: &BitPackedArray,
    indices: &PrimitiveArray,
    taken_validity: Validity,
) -> VortexResult<PrimitiveArray> {
    if indices.is_empty() {
        return Ok(PrimitiveArray::new(Buffer::<T>::empty(), taken_validity));
    }

    let offset = array.offset() as usize;
    let bit_width = array.bit_width() as usize;

    let packed = array.packed_slice::<T>();

    // Group indices by 1024-element chunk, *without* allocating on the heap
    let indices_iter = indices.as_slice::<I>().iter().map(|i| {
        i.to_usize()
            .vortex_expect("index must be expressible as usize")
    });

    let mut output = BufferMut::with_capacity(indices.len());
    let mut unpacked = [T::zero(); 1024];
    let chunk_len = 128 * bit_width / size_of::<T>();

    chunked_indices(indices_iter, offset, |chunk_idx, indices_within_chunk| {
        let packed = &packed[chunk_idx * chunk_len..][..chunk_len];

        // array_chunks produced a fixed size array, doesn't heap allocate
        let mut have_unpacked = false;
        let mut offset_chunk_iter = indices_within_chunk
            .iter()
            .copied()
            .array_chunks::<UNPACK_CHUNK_THRESHOLD>();

        // this loop only runs if we have at least UNPACK_CHUNK_THRESHOLD offsets
        for offset_chunk in &mut offset_chunk_iter {
            if !have_unpacked {
                unsafe {
                    BitPacking::unchecked_unpack(bit_width, packed, &mut unpacked);
                }
                have_unpacked = true;
            }

            for index in offset_chunk {
                output.push(unpacked[index]);
            }
        }

        // if we have a remainder (i.e., < UNPACK_CHUNK_THRESHOLD leftover offsets), we need to handle it
        if let Some(remainder) = offset_chunk_iter.into_remainder() {
            if have_unpacked {
                // we already bulk unpacked this chunk, so we can just push the remaining elements
                for index in remainder {
                    output.push(unpacked[index]);
                }
            } else {
                // we had fewer than UNPACK_CHUNK_THRESHOLD offsets in the first place,
                // so we need to unpack each one individually
                for index in remainder {
                    output.push(unsafe { unpack_single_primitive::<T>(packed, bit_width, index) });
                }
            }
        }
    });

    let mut unpatched_taken = PrimitiveArray::new(output, taken_validity);
    // Flip back to signed type before patching.
    if array.ptype().is_signed_int() {
        unpatched_taken = unpatched_taken.reinterpret_cast(array.ptype());
    }
    if let Some(patches) = array.patches() {
        if let Some(patches) = patches.take(&indices.to_array())? {
            return unpatched_taken.patch(patches);
        }
    }

    Ok(unpatched_taken)
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod test {
    use rand::distributions::Uniform;
    use rand::{thread_rng, Rng};
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::{scalar_at, slice, take};
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArrayData, IntoArrayVariant};
    use vortex_buffer::{buffer, Buffer};

    use crate::BitPackedArray;

    #[test]
    fn take_indices() {
        let indices = buffer![0, 125, 2047, 2049, 2151, 2790].into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();

        let primitive_result = take(bitpacked.as_ref(), &indices)
            .unwrap()
            .into_primitive()
            .unwrap();
        let res_bytes = primitive_result.as_slice::<u8>();
        assert_eq!(res_bytes, &[0, 62, 31, 33, 9, 18]);
    }

    #[test]
    fn take_with_patches() {
        let unpacked = Buffer::from_iter(0u32..1024).into_array();
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 2).unwrap();

        let indices = PrimitiveArray::from_iter([0, 2, 4, 6]);

        let primitive_result = take(bitpacked.as_ref(), &indices)
            .unwrap()
            .into_primitive()
            .unwrap();
        let res_bytes = primitive_result.as_slice::<u32>();
        assert_eq!(res_bytes, &[0, 2, 4, 6]);
    }

    #[test]
    fn take_sliced_indices() {
        let indices = buffer![1919, 1921].into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();
        let sliced = slice(bitpacked.as_ref(), 128, 2050).unwrap();

        let primitive_result = take(&sliced, &indices).unwrap().into_primitive().unwrap();
        let res_bytes = primitive_result.as_slice::<u8>();
        assert_eq!(res_bytes, &[31, 33]);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // This test is too slow on miri
    fn take_random_indices() {
        let num_patches: usize = 128;
        let values = (0..u16::MAX as u32 + num_patches as u32).collect::<Buffer<_>>();
        let uncompressed = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let packed = BitPackedArray::encode(uncompressed.as_ref(), 16).unwrap();
        assert!(packed.patches().is_some());

        let rng = thread_rng();
        let range = Uniform::new(0, values.len());
        let random_indices =
            PrimitiveArray::from_iter(rng.sample_iter(range).take(10_000).map(|i| i as u32));
        let taken = take(packed.as_ref(), random_indices.as_ref()).unwrap();

        // sanity check
        random_indices
            .as_slice::<u32>()
            .iter()
            .enumerate()
            .for_each(|(ti, i)| {
                assert_eq!(
                    u32::try_from(scalar_at(packed.as_ref(), *i as usize).unwrap().as_ref())
                        .unwrap(),
                    values[*i as usize]
                );
                assert_eq!(
                    u32::try_from(scalar_at(&taken, ti).unwrap().as_ref()).unwrap(),
                    values[*i as usize]
                );
            });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn take_signed_with_patches() {
        let start =
            BitPackedArray::encode(&buffer![1i32, 2i32, 3i32, 4i32].into_array(), 1).unwrap();

        let taken_primitive = super::take_primitive::<u32, u64>(
            &start,
            &PrimitiveArray::from_iter([0u64, 1, 2, 3]),
            Validity::NonNullable,
        )
        .unwrap();
        assert_eq!(taken_primitive.as_slice::<i32>(), &[1i32, 2, 3, 4]);
    }
}
