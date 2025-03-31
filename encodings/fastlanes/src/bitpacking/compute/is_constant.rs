use vortex_array::ToCanonical;
use vortex_array::arrays::{IS_CONST_LANE_WIDTH, compute_is_constant};
use vortex_array::compute::{IsConstantFn, IsConstantOpts, is_constant, scalar_at};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_dtype::{match_each_integer_ptype, match_each_unsigned_integer_ptype};
use vortex_error::VortexResult;

use crate::unpack_iter::{BitPacked, BitUnpackedChunks};
use crate::{BitPackedArray, BitPackedEncoding, unpack_single};

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
    let mut bit_unpack_iterator = BitUnpackedChunks::<T>::new(array);
    if let Some(header) = bit_unpack_iterator.header() {
        if !compute_is_constant::<_, WIDTH>(header) {
            return Ok(false);
        }
    }

    for chunk in bit_unpack_iterator.full_chunks() {
        if !compute_is_constant::<_, WIDTH>(chunk) {
            return Ok(false);
        }
    }

    if let Some(trailer) = bit_unpack_iterator.trailer() {
        if !compute_is_constant::<_, WIDTH>(trailer) {
            return Ok(false);
        }
    }

    if let Some(patches) = array.patches() {
        let constant_patches = is_constant(patches.values())?;
        if !constant_patches {
            return Ok(false);
        }

        let primitive_indices = patches.indices().to_primitive()?;
        let (unpatched_idx, patched_idx) = match_each_unsigned_integer_ptype!(patches.indices_ptype(), |$I| {
            let indices = primitive_indices.as_slice::<$I>();
            let mut unpatched_idx = 0;
            let mut patch_idx = indices[0];
            for idx in &indices[1..] {
                if *idx == unpatched_idx {
                    unpatched_idx += 1;
                } else {
                    patch_idx = *idx;
                }
            }
            (unpatched_idx as usize, patch_idx as usize)
        });
        return Ok(
            scalar_at(patches.values(), patched_idx)? == unpack_single(array, unpatched_idx)?
        );
    }

    Ok(true)
}
