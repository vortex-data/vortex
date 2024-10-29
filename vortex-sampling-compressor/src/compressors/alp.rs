use vortex::aliases::hash_set::HashSet;
use vortex::array::PrimitiveArray;
use vortex::encoding::EncodingRef;
use vortex::{Array, ArrayDef, IntoArray};
use vortex_alp::{
    alp_encode_components, match_each_alp_float_ptype, ALPArray, ALPEncoding, ALPRDEncoding, ALP,
};
use vortex_dtype::PType;
use vortex_error::VortexResult;

use super::alp_rd::ALPRDCompressor;
use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct ALPCompressor;

impl EncodingCompressor for ALPCompressor {
    fn id(&self) -> &str {
        ALP::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::ALP_COST
    }

    fn can_compress(&self, array: &Array) -> Option<&dyn EncodingCompressor> {
        // Only support primitive arrays
        let parray = PrimitiveArray::try_from(array).ok()?;

        // Only supports f32 and f64
        if !matches!(parray.ptype(), PType::F32 | PType::F64) {
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
        // TODO(robert): Fill forward nulls?
        let parray = array.as_primitive();

        let (exponents, encoded, patches) = match_each_alp_float_ptype!(
            parray.ptype(), |$T| {
            alp_encode_components::<$T>(&parray, None)
        });

        let compressed_encoded = ctx
            .named("packed")
            .excluding(self)
            .compress(&encoded, like.as_ref().and_then(|l| l.child(0)))?;

        let compressed_patches = patches
            .map(|p| {
                ctx.auxiliary("patches")
                    .excluding(self)
                    .including(&ALPRDCompressor)
                    .compress(&p, like.as_ref().and_then(|l| l.child(1)))
            })
            .transpose()?;

        Ok(CompressedArray::new(
            ALPArray::try_new(
                compressed_encoded.array,
                exponents,
                compressed_patches.as_ref().map(|p| p.array.clone()),
            )?
            .into_array(),
            Some(CompressionTree::new(
                self,
                vec![
                    compressed_encoded.path,
                    compressed_patches.and_then(|p| p.path),
                ],
            )),
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&ALPEncoding as EncodingRef, &ALPRDEncoding])
    }
}
