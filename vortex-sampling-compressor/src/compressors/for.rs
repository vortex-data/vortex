use vortex::aliases::hash_set::HashSet;
use vortex::array::PrimitiveArray;
use vortex::encoding::EncodingRef;
use vortex::stats::{trailing_zeros, ArrayStatistics};
use vortex::validity::ArrayValidity;
use vortex::{Array, ArrayDef, IntoArray, IntoArrayVariant};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_fastlanes::{for_compress, FoR, FoRArray, FoREncoding};

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct FoRCompressor;

impl EncodingCompressor for FoRCompressor {
    fn id(&self) -> &str {
        FoR::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::FOR_COST
    }

    fn can_compress(&self, array: &Array) -> Option<&dyn EncodingCompressor> {
        // Only support primitive arrays
        let parray = PrimitiveArray::try_from(array).ok()?;

        // Only supports integers
        if !parray.ptype().is_int() {
            return None;
        }

        // For all-null, cannot encode.
        if parray.logical_validity().all_invalid() {
            return None;
        }

        // Nothing for us to do if the min is already zero and tz == 0
        let shift = trailing_zeros(array);
        match_each_integer_ptype!(parray.ptype(), |$P| {
            let min: $P = parray.statistics().compute_min()?;
            if min == 0 && shift == 0 && parray.ptype().is_unsigned_int() {
                return None;
            }
        });

        Some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &Array,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let for_compressed = for_compress(&array.clone().into_primitive()?)?;

        match FoRArray::try_from(for_compressed.clone()) {
            Ok(for_array) => {
                let compressed_child = ctx
                    .named("for")
                    .excluding(self)
                    .compress(&for_array.encoded(), like.as_ref().and_then(|l| l.child(0)))?;
                Ok(CompressedArray::new(
                    FoRArray::try_new(
                        compressed_child.array,
                        for_array.owned_reference_scalar(),
                        for_array.shift(),
                    )
                    .map(|a| a.into_array())?,
                    Some(CompressionTree::new(self, vec![compressed_child.path])),
                ))
            }
            Err(_) => {
                let compressed_child = ctx
                    .named("for")
                    .excluding(self)
                    .compress(&for_compressed, like.as_ref())?;
                Ok(CompressedArray::new(
                    compressed_child.array,
                    Some(CompressionTree::new(self, vec![compressed_child.path])),
                ))
            }
        }
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&FoREncoding as EncodingRef])
    }
}
