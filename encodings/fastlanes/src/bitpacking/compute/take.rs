use std::cmp::min;

use fastlanes::BitPacking;
use itertools::Itertools;
use vortex_array::array::{PrimitiveArray, SparseArray};
use vortex_array::compute::{slice, take, TakeFn, TakeOptions};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{
    ArrayDType, ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant, IntoCanonical,
};
use vortex_dtype::{
    match_each_integer_ptype, match_each_unsigned_integer_ptype, NativePType, PType,
};
use vortex_error::{VortexExpect as _, VortexResult};

use crate::{unpack_single_primitive, BitPackedArray, BitPackedEncoding};

// assuming the buffer is already allocated (which will happen at most once) then unpacking
// all 1024 elements takes ~8.8x as long as unpacking a single element on an M2 Macbook Air.
// see https://github.com/spiraldb/vortex/pull/190#issue-2223752833
pub(super) const UNPACK_CHUNK_THRESHOLD: usize = 8;
const BULK_PATCH_THRESHOLD: usize = 64;

impl TakeFn<BitPackedArray> for BitPackedEncoding {
    fn take(
        &self,
        array: &BitPackedArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        // If the indices are large enough, it's faster to flatten and take the primitive array.
        if indices.len() * UNPACK_CHUNK_THRESHOLD > array.len() {
            return take(
                array.clone().into_canonical()?.into_primitive()?,
                indices,
                options,
            );
        }

        let ptype: PType = array.dtype().try_into()?;
        let validity = array.validity();
        let taken_validity = validity.take(indices, options)?;

        let indices = indices.clone().into_primitive()?;
        let taken = match_each_unsigned_integer_ptype!(ptype, |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                PrimitiveArray::from_vec(take_primitive::<$T, $I>(array, &indices, options)?, taken_validity)
            })
        });
        Ok(taken.reinterpret_cast(ptype).into_array())
    }
}

fn take_primitive<T: NativePType + BitPacking, I: NativePType>(
    array: &BitPackedArray,
    indices: &PrimitiveArray,
    options: TakeOptions,
) -> VortexResult<Vec<T>> {
    if indices.is_empty() {
        return Ok(vec![]);
    }

    let offset = array.offset() as usize;
    let bit_width = array.bit_width() as usize;

    let packed = array.packed_slice::<T>();
    let patches = array.patches().map(SparseArray::try_from).transpose()?;

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

    let mut batch_count = 0_usize;
    for (chunk, offsets) in chunked_indices {
        batch_count += 1;
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

    if let Some(ref patches) = patches {
        patch_for_take_primitive::<T, I>(
            patches,
            indices,
            offset,
            batch_count,
            &mut output,
            options,
        )?;
    }

    Ok(output)
}

fn patch_for_take_primitive<T: NativePType, I: NativePType>(
    patches: &SparseArray,
    indices: &PrimitiveArray,
    offset: usize,
    batch_count: usize,
    output: &mut [T],
    options: TakeOptions,
) -> VortexResult<()> {
    #[inline]
    fn inner_patch<T: NativePType>(
        patches: &SparseArray,
        indices: &PrimitiveArray,
        output: &mut [T],
        options: TakeOptions,
    ) -> VortexResult<()> {
        let taken_patches = take(patches.as_ref(), indices.as_ref(), options)?;
        let taken_patches = SparseArray::try_from(taken_patches)?;

        let base_index = output.len() - indices.len();
        let output_patches = taken_patches
            .values()
            .into_primitive()?
            .reinterpret_cast(T::PTYPE);
        taken_patches
            .resolved_indices()
            .iter()
            .map(|idx| base_index + *idx)
            .zip_eq(output_patches.maybe_null_slice::<T>())
            .for_each(|(idx, val)| {
                output[idx] = *val;
            });

        Ok(())
    }

    // if we have a small number of relatively large batches, we gain by slicing and then patching inside the loop
    // if we have a large number of relatively small batches, the overhead isn't worth it, and we're better off with a bulk patch
    // roughly, if we have an average of less than 64 elements per batch, we prefer bulk patching
    let prefer_bulk_patch = batch_count * BULK_PATCH_THRESHOLD > indices.len();
    if prefer_bulk_patch {
        return inner_patch(patches, indices, output, options);
    }

    let min_index = patches.min_index().unwrap_or_default();
    let max_index = patches.max_index().unwrap_or_default();

    // Group indices into 1024-element chunks and relativise them to the beginning of each chunk
    let chunked_indices = &indices
        .maybe_null_slice::<I>()
        .iter()
        .map(|i| {
            i.to_usize()
                .vortex_expect("index must be expressible as usize")
                + offset
        })
        .filter(|i| *i >= min_index && *i <= max_index) // short-circuit
        .chunk_by(|idx| idx / 1024);

    for (chunk, offsets) in chunked_indices {
        // NOTE: we need to subtract the array offset before slicing into the patches.
        // This is because BitPackedArray is rounded to block boundaries, but patches
        // is sliced exactly.
        let patches_start = if chunk == 0 {
            0
        } else {
            (chunk * 1024) - offset
        };
        let patches_end = min((chunk + 1) * 1024 - offset, patches.len());
        let patches_slice = slice(patches.as_ref(), patches_start, patches_end)?;
        let patches_slice = SparseArray::try_from(patches_slice)?;
        if patches_slice.is_empty() {
            continue;
        }

        let min_index = patches_slice.min_index().unwrap_or_default();
        let max_index = patches_slice.max_index().unwrap_or_default();
        let offsets = offsets
            .map(|i| (i % 1024) as u16)
            .filter(|i| *i as usize >= min_index && *i as usize <= max_index)
            .collect_vec();
        if offsets.is_empty() {
            continue;
        }

        inner_patch(
            &patches_slice,
            &PrimitiveArray::from(offsets),
            output,
            options,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rand::distributions::Uniform;
    use rand::{thread_rng, Rng};
    use vortex_array::array::{PrimitiveArray, SparseArray};
    use vortex_array::compute::unary::scalar_at;
    use vortex_array::compute::{slice, take, TakeOptions};
    use vortex_array::{IntoArrayData, IntoArrayVariant};

    use crate::BitPackedArray;

    #[test]
    fn take_indices() {
        let indices = PrimitiveArray::from(vec![0, 125, 2047, 2049, 2151, 2790]).into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from((0..4096).map(|i| (i % 63) as u8).collect::<Vec<_>>());
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();

        let primitive_result = take(bitpacked.as_ref(), &indices, TakeOptions::default())
            .unwrap()
            .into_primitive()
            .unwrap();
        let res_bytes = primitive_result.maybe_null_slice::<u8>();
        assert_eq!(res_bytes, &[0, 62, 31, 33, 9, 18]);
    }

    #[test]
    fn take_sliced_indices() {
        let indices = PrimitiveArray::from(vec![1919, 1921]).into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from((0..4096).map(|i| (i % 63) as u8).collect::<Vec<_>>());
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();
        let sliced = slice(bitpacked.as_ref(), 128, 2050).unwrap();

        let primitive_result = take(&sliced, &indices, TakeOptions::default())
            .unwrap()
            .into_primitive()
            .unwrap();
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

        let patches = SparseArray::try_from(packed.patches().unwrap()).unwrap();
        assert_eq!(
            patches.resolved_indices(),
            ((values.len() + 1 - num_patches)..values.len()).collect_vec()
        );

        let rng = thread_rng();
        let range = Uniform::new(0, values.len());
        let random_indices: PrimitiveArray = rng
            .sample_iter(range)
            .take(10_000)
            .map(|i| i as u32)
            .collect_vec()
            .into();
        let taken = take(
            packed.as_ref(),
            random_indices.as_ref(),
            TakeOptions::default(),
        )
        .unwrap();

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

    #[test]
    fn test_scalar_at() {
        let values = (0u32..257).collect_vec();
        let uncompressed = PrimitiveArray::from(values.clone()).into_array();
        let packed = BitPackedArray::encode(&uncompressed, 8).unwrap();
        assert!(packed.patches().is_some());

        let patches = SparseArray::try_from(packed.patches().unwrap()).unwrap();
        assert_eq!(patches.resolved_indices(), vec![256]);

        values.iter().enumerate().for_each(|(i, v)| {
            assert_eq!(
                u32::try_from(scalar_at(packed.as_ref(), i).unwrap().as_ref()).unwrap(),
                *v
            );
        });
    }
}
