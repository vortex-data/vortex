use std::ops::Range;

use itertools::Itertools;
use lending_iterator::LendingIterator;
use num_traits::AsPrimitive;
use vortex_array::ToCanonical;
use vortex_array::arrays::{IS_CONST_LANE_WIDTH, PrimitiveArray, compute_is_constant};
use vortex_array::compute::{IsConstantFn, IsConstantOpts};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_dtype::{NativePType, match_each_integer_ptype, match_each_unsigned_integer_ptype};
use vortex_error::VortexResult;

use crate::unpack_iter::BitPacked;
use crate::{BitPackedArray, BitPackedEncoding};

impl IsConstantFn<&BitPackedArray> for BitPackedEncoding {
    fn is_constant(
        &self,
        array: &BitPackedArray,
        _opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
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
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::compute::is_constant;
    use vortex_buffer::buffer;
    use vortex_error::VortexUnwrap;

    use crate::BitPackedArray;

    #[test]
    fn is_constant_with_patches() {
        let array = BitPackedArray::encode(&buffer![4; 1025].into_array(), 2).vortex_unwrap();
        assert!(is_constant(&array).vortex_unwrap());
    }
}
