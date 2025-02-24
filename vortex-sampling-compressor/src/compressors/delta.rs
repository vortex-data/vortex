use vortex_array::aliases::hash_set::HashSet;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayExt, Encoding, EncodingId};
use vortex_error::VortexResult;
use vortex_fastlanes::{delta_compress, DeltaArray, DeltaEncoding};

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct DeltaCompressor;

impl EncodingCompressor for DeltaCompressor {
    fn id(&self) -> &str {
        DeltaEncoding::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::DELTA_COST
    }

    fn can_compress(&self, array: &dyn Array) -> Option<&dyn EncodingCompressor> {
        // Only support primitive arrays
        let parray = array.as_opt::<PrimitiveArray>()?;

        // Only supports ints
        if !parray.ptype().is_unsigned_int() {
            return None;
        }

        Some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &dyn Array,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let parray = PrimitiveArray::try_from(array.to_array())?;
        let validity = ctx.compress_validity(parray.validity().clone())?;

        // Compress the filled array
        let (bases, deltas) = delta_compress(&parray)?;

        // Recursively compress the bases and deltas
        let bases = ctx
            .named("bases")
            .compress(&bases, like.as_ref().and_then(|l| l.child(0)))?;
        let deltas = ctx
            .named("deltas")
            .compress(&deltas, like.as_ref().and_then(|l| l.child(1)))?;

        Ok(CompressedArray::compressed(
            DeltaArray::try_from_delta_compress_parts(bases.array, deltas.array, validity)?
                .into_array(),
            Some(CompressionTree::new(self, vec![bases.path, deltas.path])),
            array,
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingId> {
        HashSet::from([DeltaEncoding::ID])
    }
}
