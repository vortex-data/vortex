use fastlanes::BitPacking;
use itertools::Itertools;
use vortex_array::array::PrimitiveArray;
use vortex_array::compute::{take, try_cast, TakeFn};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{
    ArrayDType, ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant, IntoCanonical, ToArrayData,
};
use vortex_dtype::{
    match_each_integer_ptype, match_each_unsigned_integer_ptype, DType, NativePType, Nullability,
    PType,
};
use vortex_error::{VortexExpect as _, VortexResult};

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

        let ptype: PType = array.dtype().try_into()?;
        let validity = array.validity();
        let taken_validity = validity.take(indices)?;

        let indices = indices.clone().into_primitive()?;
        let taken = match_each_unsigned_integer_ptype!(ptype, |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                PrimitiveArray::from_vec(take_primitive::<$T, $I>(array, &indices)?, taken_validity)
            })
        });
        Ok(taken.reinterpret_cast(ptype).into_array())
    }
}

fn take_primitive<T: NativePType + BitPacking, I: NativePType>(
    array: &BitPackedArray,
    indices: &PrimitiveArray,
) -> VortexResult<Vec<T>> {
    if indices.is_empty() {
        return Ok(vec![]);
    }

    let offset = array.offset() as usize;
    let bit_width = array.bit_width() as usize;

    let packed = array.packed_slice::<T>();

    // Group indices by 1024-element chunk, *without* allocating on the heap
    let chunked_indices = &indices
        .maybe_null_slice::<I>()
        .iter()
        .map(|i| {
            i.to_usize()
                .vortex_expect("index must be expressible as usize")
                + offset
        })
        .chunk_by(|idx| idx / 1024);

    let mut output = Vec::with_capacity(indices.len());
    let mut unpacked = [T::zero(); 1024];

    for (chunk, offsets) in chunked_indices {
        let chunk_size = 128 * bit_width / size_of::<T>();
        let packed_chunk = &packed[chunk * chunk_size..][..chunk_size];

        // array_chunks produced a fixed size array, doesn't heap allocate
        let mut have_unpacked = false;
        let mut offset_chunk_iter = offsets
            // relativize indices to the start of the chunk
            .map(|i| i % 1024)
            .array_chunks::<UNPACK_CHUNK_THRESHOLD>();

        // this loop only runs if we have at least UNPACK_CHUNK_THRESHOLD offsets
        for offset_chunk in &mut offset_chunk_iter {
            if !have_unpacked {
                unsafe {
                    BitPacking::unchecked_unpack(bit_width, packed_chunk, &mut unpacked);
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
                    output.push(unsafe {
                        unpack_single_primitive::<T>(packed_chunk, bit_width, index)
                    });
                }
            }
        }
    }

    if let Some(patches) = array
        .patches()
        .map(|p| p.take(&indices.to_array()))
        .transpose()?
        .flatten()
    {
        let indices = try_cast(
            patches.indices(),
            &DType::Primitive(PType::U64, Nullability::NonNullable),
        )?
        .into_primitive()?;

        // TODO(ngates): can patch values themselves have nulls, or do we ensure they're in our
        //  validity bitmap?
        let values = patches.values().clone().into_primitive()?;
        let values_slice = values.maybe_null_slice::<T>();

        for (idx, v) in indices.maybe_null_slice::<u64>().iter().zip(values_slice) {
            output[*idx as usize] = *v;
        }
    }

    Ok(output)
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod test {
    use itertools::Itertools;
    use rand::distributions::Uniform;
    use rand::{thread_rng, Rng};
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::{scalar_at, slice, take};
    use vortex_array::{IntoArrayData, IntoArrayVariant};

    use crate::BitPackedArray;

    #[test]
    fn take_indices() {
        let indices = PrimitiveArray::from(vec![0, 125, 2047, 2049, 2151, 2790]).into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from((0..4096).map(|i| (i % 63) as u8).collect::<Vec<_>>());
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();

        let primitive_result = take(bitpacked.as_ref(), &indices)
            .unwrap()
            .into_primitive()
            .unwrap();
        let res_bytes = primitive_result.maybe_null_slice::<u8>();
        assert_eq!(res_bytes, &[0, 62, 31, 33, 9, 18]);
    }

    #[test]
    fn take_with_patches() {
        let unpacked = PrimitiveArray::from((0u32..100_000).collect_vec()).into_array();
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 2).unwrap();

        let indices = PrimitiveArray::from(vec![0, 2, 4, 6]);

        let primitive_result = take(bitpacked.as_ref(), &indices)
            .unwrap()
            .into_primitive()
            .unwrap();
        let res_bytes = primitive_result.maybe_null_slice::<u32>();
        assert_eq!(res_bytes, &[0, 2, 4, 6]);
    }

    #[test]
    fn take_sliced_indices() {
        let indices = PrimitiveArray::from(vec![1919, 1921]).into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from((0..4096).map(|i| (i % 63) as u8).collect::<Vec<_>>());
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();
        let sliced = slice(bitpacked.as_ref(), 128, 2050).unwrap();

        let primitive_result = take(&sliced, &indices).unwrap().into_primitive().unwrap();
        let res_bytes = primitive_result.maybe_null_slice::<u8>();
        assert_eq!(res_bytes, &[31, 33]);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // This test is too slow on miri
    fn take_random_indices() {
        let num_patches: usize = 128;
        let values = (0..u16::MAX as u32 + num_patches as u32).collect::<Vec<_>>();
        let uncompressed = PrimitiveArray::from(values.clone());
        let packed = BitPackedArray::encode(uncompressed.as_ref(), 16).unwrap();
        assert!(packed.patches().is_some());

        let rng = thread_rng();
        let range = Uniform::new(0, values.len());
        let random_indices: PrimitiveArray = rng
            .sample_iter(range)
            .take(10_000)
            .map(|i| i as u32)
            .collect_vec()
            .into();
        let taken = take(packed.as_ref(), random_indices.as_ref()).unwrap();

        // sanity check
        random_indices
            .maybe_null_slice::<u32>()
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
}
