use vortex::aliases::hash_set::HashSet;
use vortex::array::PrimitiveArray;
use vortex::encoding::EncodingRef;
use vortex::{Array, ArrayDef, IntoArray};
use vortex_error::VortexResult;
use vortex_fastlanes::{delta_compress, Delta, DeltaArray, DeltaEncoding};

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct DeltaCompressor;

impl EncodingCompressor for DeltaCompressor {
    fn id(&self) -> &str {
        Delta::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::DELTA_COST
    }

    fn can_compress(&self, array: &Array) -> Option<&dyn EncodingCompressor> {
        // Only support primitive arrays
        let parray = PrimitiveArray::try_from(array).ok()?;

        // Only supports ints
        if !parray.ptype().is_unsigned_int() {
            return None;
        }

        Some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &Array,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let parray = PrimitiveArray::try_from(array)?;
        let validity = ctx.compress_validity(parray.validity())?;

        // Compress the filled array
        let (bases, deltas) = delta_compress(&parray)?;

        // Recursively compress the bases and deltas
        let bases = ctx
            .named("bases")
            .compress(bases.as_ref(), like.as_ref().and_then(|l| l.child(0)))?;
        let deltas = ctx
            .named("deltas")
            .compress(deltas.as_ref(), like.as_ref().and_then(|l| l.child(1)))?;

        Ok(CompressedArray::new(
            DeltaArray::try_from_delta_compress_parts(bases.array, deltas.array, validity)?
                .into_array(),
            Some(CompressionTree::new(self, vec![bases.path, deltas.path])),
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&DeltaEncoding as EncodingRef])
    }
}
