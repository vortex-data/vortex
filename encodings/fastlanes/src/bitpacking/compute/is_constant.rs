use lending_iterator::LendingIterator;
use vortex_array::arrays::{IS_CONST_LANE_WIDTH, compute_is_constant};
use vortex_array::compute::{
    IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, is_constant_opts,
};
use vortex_array::{Array, register_kernel};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::unpack_iter::BitPacked;
use crate::{BitPackedArray, BitPackedEncoding};

impl IsConstantKernel for BitPackedEncoding {
    fn is_constant(
        &self,
        array: &BitPackedArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        match_each_integer_ptype!(array.ptype(), |$P| {
            bitpacked_is_constant::<$P, {IS_CONST_LANE_WIDTH / size_of::<$P>()}>(array, opts)
        })
    }
}

register_kernel!(IsConstantKernelAdapter(BitPackedEncoding).lift());

fn bitpacked_is_constant<T: BitPacked, const WIDTH: usize>(
    array: &BitPackedArray,
    opts: &IsConstantOpts,
) -> VortexResult<Option<bool>> {
    let mut bit_unpack_iterator = array.unpacked_chunks::<T>();

    // Bitpacked arrays with patches are only constant if all values are in patches
    if let Some(patches) = array.patches() {
        return if patches.num_patches() == array.len() {
            is_constant_opts(patches.values(), opts)
        } else {
            Ok(Some(false))
        };
    }

    let mut header_constant_value = None;
    if let Some(header) = bit_unpack_iterator.initial() {
        if !compute_is_constant::<_, WIDTH>(header) {
            return Ok(Some(false));
        }
        header_constant_value = Some(header[0]);
    }

    let mut first_chunk_value = None;
    let mut chunks_iter = bit_unpack_iterator.full_chunks();
    while let Some(chunk) = chunks_iter.next() {
        if !compute_is_constant::<_, WIDTH>(chunk) {
            return Ok(Some(false));
        }

        if let Some(chunk_value) = first_chunk_value {
            if chunk_value != chunk[0] {
                return Ok(Some(false));
            }
        } else {
            if let Some(header_value) = header_constant_value {
                if header_value != chunk[0] {
                    return Ok(Some(false));
                }
            }
            first_chunk_value = Some(chunk[0]);
        }
    }

    if let Some(trailer) = bit_unpack_iterator.trailer() {
        if !compute_is_constant::<_, WIDTH>(trailer) {
            return Ok(Some(false));
        }

        if let Some(previous_const_value) = header_constant_value.or(first_chunk_value) {
            if previous_const_value != trailer[0] {
                return Ok(Some(false));
            }
        }
    }

    Ok(Some(true))
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::compute::is_constant;
    use vortex_buffer::buffer;

    use crate::BitPackedArray;

    #[test]
    fn is_constant_with_patches() {
        let array = BitPackedArray::encode(&buffer![4; 1025].into_array(), 2).unwrap();
        assert!(is_constant(&array).unwrap().unwrap());
    }
}
