use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::PrimitiveArray;
use vortex_array::encoding::EncodingRef;
use vortex_array::stats::ArrayStatistics;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};
use vortex_error::{vortex_err, vortex_panic, VortexResult};
use vortex_fastlanes::{
    bitpack, count_exceptions, find_best_bit_width, find_min_patchless_bit_width, gather_patches,
    BitPackedArray, BitPackedEncoding,
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

    fn can_compress(&self, array: &ArrayData) -> Option<&dyn EncodingCompressor> {
        // Only support primitive arrays
        let parray = PrimitiveArray::try_from(array.clone()).ok()?;

        // Only supports unsigned ints
        if !parray.ptype().is_unsigned_int() {
            return None;
        }

        let bit_width = self.find_bit_width(&parray).ok()?;

        // Check that the bit width is less than the type's bit width
        if bit_width == parray.ptype().bit_width() as u8 {
            return None;
        }

        Some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &ArrayData,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let parray = array.clone().into_primitive()?;
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
            return Ok(CompressedArray::uncompressed(array.clone()));
        }

        let validity = ctx.compress_validity(parray.validity())?;
        let packed_buffer = bitpack(&parray, bit_width)?;
        let patches = (num_exceptions > 0)
            .then(|| {
                gather_patches(&parray, bit_width, num_exceptions).map(|p| {
                    ctx.auxiliary("patches")
                        .excluding(&BITPACK_WITH_PATCHES)
                        .including(&BITPACK_NO_PATCHES)
                        .compress(&p, like.as_ref().and_then(|l| l.child(0)))
                })
            })
            .flatten()
            .transpose()?;

        Ok(CompressedArray::compressed(
            BitPackedArray::try_new(
                packed_buffer,
                parray.ptype(),
                validity,
                patches.as_ref().map(|p| p.array.clone()),
                bit_width,
                parray.len(),
            )?
            .into_array(),
            Some(CompressionTree::new(
                self,
                vec![patches.and_then(|p| p.path)],
            )),
            Some(array.statistics()),
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&BitPackedEncoding as EncodingRef])
    }
}
