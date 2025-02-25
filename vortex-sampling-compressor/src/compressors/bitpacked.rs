#![allow(clippy::cast_possible_truncation)]
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayExt, Encoding, EncodingId, ToCanonical};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexResult};
use vortex_fastlanes::{
    bitpack_unchecked, count_exceptions, find_best_bit_width, find_min_patchless_bit_width,
    gather_patches, BitPackedArray, BitPackedEncoding,
};

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

pub const BITPACK_WITH_PATCHES: BitPackedCompressor = BitPackedCompressor {
    allow_patches: true,
};
pub const BITPACK_NO_PATCHES: BitPackedCompressor = BitPackedCompressor {
    allow_patches: false,
};

#[derive(Debug)]
pub struct BitPackedCompressor {
    allow_patches: bool,
}

impl BitPackedCompressor {
    fn find_bit_width(&self, array: &PrimitiveArray) -> VortexResult<u8> {
        if self.allow_patches {
            find_best_bit_width(array)
        } else {
            find_min_patchless_bit_width(array)
        }
    }
}

impl EncodingCompressor for BitPackedCompressor {
    fn id(&self) -> &str {
        if self.allow_patches {
            "fastlanes.bitpacked"
        } else {
            "fastlanes.bitpacked_no_patches"
        }
    }

    fn cost(&self) -> u8 {
        if self.allow_patches {
            constants::BITPACKED_WITH_PATCHES_COST
        } else {
            constants::BITPACKED_NO_PATCHES_COST
        }
    }

    fn can_compress(&self, array: &dyn Array) -> Option<&dyn EncodingCompressor> {
        // Only support primitive arrays
        let parray = array.as_opt::<PrimitiveArray>()?;

        // Only integer arrays can be bit-packed
        if !parray.ptype().is_int() {
            return None;
        }

        // Only arrays with non-negative values can be bit-packed
        if !parray.ptype().is_unsigned_int() {
            let has_negative_elements = match_each_integer_ptype!(parray.ptype(), |$P| {
                parray.statistics().compute_min::<Option<$P>>().unwrap_or_default().unwrap_or_default() < 0
            });

            if has_negative_elements {
                return None;
            }
        }

        let bit_width = self.find_bit_width(parray).ok()?;

        // Check that the bit width is less than the type's bit width
        if bit_width == parray.ptype().bit_width() as u8 {
            return None;
        }

        Some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &dyn Array,
        _like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let parray = array.to_primitive()?;
        // Only arrays with non-negative values can be bit-packed
        if !parray.ptype().is_unsigned_int() {
            let has_negative_elements = match_each_integer_ptype!(parray.ptype(), |$P| {
                parray.statistics().compute_min::<Option<$P>>().unwrap_or_default().unwrap_or_default() < 0
            });

            if has_negative_elements {
                vortex_bail!("Cannot BitPackCompressor::compress an array with negative values");
            }
        }

        let bit_width_freq = parray
            .statistics()
            .compute_bit_width_freq()
            .ok_or_else(|| vortex_err!(ComputeError: "missing bit width frequency"))?;

        let bit_width = self.find_bit_width(&parray)?;
        let num_exceptions = count_exceptions(bit_width, &bit_width_freq);
        if !self.allow_patches && num_exceptions > 0 {
            vortex_panic!(
                "Found {} exceptions with patchless bit width {}",
                num_exceptions,
                bit_width
            )
        }

        if bit_width == parray.ptype().bit_width() as u8 {
            // Nothing we can do
            return Ok(CompressedArray::uncompressed(array.to_array()));
        }

        let validity = ctx.compress_validity(parray.validity().clone())?;
        // SAFETY: we check that the array only contains non-negative values.
        let packed_buffer = unsafe { bitpack_unchecked(&parray, bit_width)? };
        let patches = (num_exceptions > 0)
            .then(|| {
                gather_patches(&parray, bit_width, num_exceptions).map(|p| {
                    ctx.auxiliary("patches")
                        .excluding(&BITPACK_WITH_PATCHES)
                        .including(&BITPACK_NO_PATCHES)
                        .compress_patches(p)
                })
            })
            .flatten()
            .transpose()?;

        Ok(CompressedArray::compressed(
            // SAFETY: we ensure the array contains no negative values.
            unsafe {
                BitPackedArray::new_unchecked(
                    packed_buffer,
                    parray.ptype(),
                    validity,
                    patches,
                    bit_width,
                    parray.len(),
                )?
            }
            .into_array(),
            Some(CompressionTree::new(self, vec![])),
            array,
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingId> {
        HashSet::from([BitPackedEncoding::ID])
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::ConstantArray;
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;

    use crate::compressors::bitpacked::{BITPACK_NO_PATCHES, BITPACK_WITH_PATCHES};
    use crate::compressors::EncodingCompressor;
    use crate::SamplingCompressor;

    #[test]
    fn cannot_compress() {
        // cannot compress when array contains negative values
        assert!(BITPACK_NO_PATCHES
            .can_compress(&buffer![-1i32, 0i32, 1i32].into_array())
            .is_none());

        // Non-integer primitive array.
        assert!(BITPACK_NO_PATCHES
            .can_compress(&buffer![0f32, 1f32].into_array())
            .is_none());

        // non-PrimitiveArray
        assert!(BITPACK_NO_PATCHES
            .can_compress(&ConstantArray::new(3u32, 10))
            .is_none());
    }

    #[test]
    fn can_compress() {
        // Unsigned integers
        assert!(BITPACK_NO_PATCHES
            .can_compress(&buffer![0u32, 1u32, 2u32].into_array())
            .is_some());

        // Signed non-negative integers
        assert!(BITPACK_WITH_PATCHES
            .can_compress(&buffer![0i32, 1i32, 2i32].into_array())
            .is_some());
    }

    #[test]
    fn compress_negatives_fails() {
        assert!(BITPACK_NO_PATCHES
            .compress(
                &buffer![-1i32, 0i32].into_array(),
                None,
                SamplingCompressor::default(),
            )
            .is_err());
    }
}
