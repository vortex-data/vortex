use arrow_buffer::ArrowNativeType;
use fastlanes::BitPacking;
use itertools::Itertools;
use vortex_array::array::{PrimitiveArray, SparseArray};
use vortex_array::compute::{filter, FilterFn, FilterIter, FilterMask};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_dtype::{match_each_unsigned_integer_ptype, NativePType};
use vortex_error::VortexResult;

use crate::bitpacking::compute::take::UNPACK_CHUNK_THRESHOLD;
use crate::BitPackedArray;

impl FilterFn for BitPackedArray {
    fn filter(&self, mask: FilterMask) -> VortexResult<ArrayData> {
        let primitive = match_each_unsigned_integer_ptype!(self.ptype(), |$I| {
            filter_primitive::<$I>(self, mask)
        });
        Ok(primitive?.into_array())
    }
}

fn filter_primitive<T: NativePType + BitPacking + ArrowNativeType>(
    array: &BitPackedArray,
    mask: FilterMask,
) -> VortexResult<PrimitiveArray> {
    let validity = array.validity().filter(&mask)?;

    let patches = array
        .patches()
        .map(|patches| filter(&patches, mask.clone()))
        .transpose()?
        .map(SparseArray::try_from)
        .transpose()?;

    // Short-circuit if the selectivity is high enough.
    if mask.selectivity() > 0.8 {
        return filter(array.clone().into_primitive()?.as_ref(), mask)
            .and_then(|a| a.into_primitive());
    }

    let values: Vec<T> = match mask.iter()? {
        FilterIter::Indices(indices) => {
            filter_indices(array, mask.true_count(), indices.iter().copied())
        }
        FilterIter::IndicesIter(iter) => filter_indices(array, mask.true_count(), iter),
        FilterIter::Slices(slices) => {
            filter_slices(array, mask.true_count(), slices.iter().copied())
        }
        FilterIter::SlicesIter(iter) => filter_slices(array, mask.true_count(), iter),
    };

    let mut values = PrimitiveArray::from_vec(values, validity);

    if let Some(patches) = patches {
        let patch_values = patches.values().into_primitive()?;
        values = values.patch(
            &patches.resolved_indices(),
            patch_values.maybe_null_slice::<T>(),
            patch_values.validity(),
        )?;
    }

    Ok(values)
}

fn filter_indices<T: NativePType + BitPacking + ArrowNativeType>(
    array: &BitPackedArray,
    indices_len: usize,
    indices: impl Iterator<Item = usize>,
) -> Vec<T> {
    let offset = array.offset() as usize;
    let mut values = Vec::with_capacity(indices_len);

    // Some re-usable memory to store per-chunk indices.
    let mut indices_within_chunk: Vec<usize> = Vec::with_capacity(1024);
    let mut unpacked = vec![T::zero(); 1024];

    // Group the indices by the FastLanes chunk they belong to.
    let chunked = indices.chunk_by(|&idx| (idx + offset) / 1024);
    let chunk_len = 128 * array.bit_width() as usize / size_of::<T>();

    chunked.into_iter().for_each(|(chunk_idx, indices)| {
        let packed = &array.packed_slice::<T>()[chunk_idx * chunk_len..(chunk_idx + 1) * chunk_len];

        // Re-use the indices buffer to store the indices within the current chunk.
        indices_within_chunk.clear();
        indices_within_chunk.extend(indices.map(|idx| (idx + offset) % 1024));

        if indices_within_chunk.len() == 1024 {
            // Unpack the entire chunk.
            unsafe { BitPacking::unchecked_unpack(array.bit_width() as usize, packed, &mut values) }
        } else if indices_within_chunk.len() > UNPACK_CHUNK_THRESHOLD {
            // Unpack into a temporary chunk and then copy the values.
            unsafe {
                BitPacking::unchecked_unpack(array.bit_width() as usize, packed, &mut unpacked)
            }
            values.extend(
                indices_within_chunk
                    .iter()
                    .map(|&idx| unsafe { *unpacked.get_unchecked(idx) }),
            );
        } else {
            // Otherwise, unpack each element individually.
            values.extend(indices_within_chunk.iter().map(|&idx| unsafe {
                BitPacking::unchecked_unpack_single(array.bit_width() as usize, packed, idx)
            }));
        }
    });

    values
}

fn filter_slices<T: NativePType + BitPacking + ArrowNativeType>(
    array: &BitPackedArray,
    indices_len: usize,
    slices: impl Iterator<Item = (usize, usize)>,
) -> Vec<T> {
    // TODO(ngates): do this more efficiently.
    filter_indices(
        array,
        indices_len,
        slices.into_iter().flat_map(|(start, end)| start..end),
    )
}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::{filter, slice, FilterMask};
    use vortex_array::IntoArrayVariant;

    use crate::BitPackedArray;

    #[test]
    fn take_indices() {
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from((0..4096).map(|i| (i % 63) as u8).collect::<Vec<_>>());
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();

        let mask = FilterMask::from_indices(bitpacked.len(), [0, 125, 2047, 2049, 2151, 2790]);

        let primitive_result = filter(bitpacked.as_ref(), mask)
            .unwrap()
            .into_primitive()
            .unwrap();
        let res_bytes = primitive_result.maybe_null_slice::<u8>();
        assert_eq!(res_bytes, &[0, 62, 31, 33, 9, 18]);
    }

    #[test]
    fn take_sliced_indices() {
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from((0..4096).map(|i| (i % 63) as u8).collect::<Vec<_>>());
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();
        let sliced = slice(bitpacked.as_ref(), 128, 2050).unwrap();

        let mask = FilterMask::from_indices(sliced.len(), [1919, 1921]);

        let primitive_result = filter(&sliced, mask).unwrap().into_primitive().unwrap();
        let res_bytes = primitive_result.maybe_null_slice::<u8>();
        assert_eq!(res_bytes, &[31, 33]);
    }
}
