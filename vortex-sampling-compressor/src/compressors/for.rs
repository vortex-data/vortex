use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::PrimitiveArray;
use vortex_array::encoding::EncodingRef;
use vortex_array::stats::{trailing_zeros, ArrayStatistics};
use vortex_array::validity::ArrayValidity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, ArrayDef, IntoArrayData, IntoArrayVariant};
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

    fn can_compress(&self, array: &ArrayData) -> Option<&dyn EncodingCompressor> {
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
        array: &ArrayData,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let compressed = for_compress(&array.clone().into_primitive()?)?;

        if let Ok(for_array) = FoRArray::try_from(compressed.clone()) {
            let compressed_child = ctx
                .named("for")
                .excluding(self)
                .compress(&for_array.encoded(), like.as_ref().and_then(|l| l.child(0)))?;
            Ok(CompressedArray::compressed(
                FoRArray::try_new(
                    compressed_child.array,
                    for_array.owned_reference_scalar(),
                    for_array.shift(),
                )
                .map(|a| a.into_array())?,
                Some(CompressionTree::new(self, vec![compressed_child.path])),
                Some(array.statistics()),
            ))
        } else {
            // otherwise, we chose a different encoding (e.g., constant or sparse), try compressing that
            // (will no-op for constant, may compress indices/values for sparse)
            let compressed_child = ctx
                .named("for")
                .excluding(self)
                .compress(&compressed, like.as_ref().and_then(|l| l.child(0)))?;
            Ok(CompressedArray::compressed(
                compressed_child.array,
                Some(CompressionTree::new(self, vec![compressed_child.path])),
                Some(array.statistics()),
            ))
        }
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&FoREncoding as EncodingRef])
    }
}
