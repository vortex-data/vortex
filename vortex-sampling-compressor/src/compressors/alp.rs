use vortex_alp::{alp_encode_components, ALPArray, ALPEncoding, ALPRDEncoding};
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayExt, Encoding, EncodingId, ToCanonical};
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPackedEncoding;

use super::alp_rd::ALPRDCompressor;
use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct ALPCompressor;

impl EncodingCompressor for ALPCompressor {
    fn id(&self) -> &str {
        ALPEncoding::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::ALP_COST
    }

    fn can_compress(&self, array: &dyn Array) -> Option<&dyn EncodingCompressor> {
        // Only support primitive arrays
        let parray = array.as_opt::<PrimitiveArray>()?;

        // Only supports f32 and f64
        if !matches!(parray.ptype(), PType::F32 | PType::F64) {
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
        let (exponents, encoded, patches) = alp_encode_components(&array.to_primitive()?)?;

        let compressed_encoded = ctx
            .named("packed")
            .excluding(self)
            .compress(&encoded, like.as_ref().and_then(|l| l.child(0)))?;

        // Attempt to compress patches with ALP-RD encoding
        let compressed_patches = patches
            .map(|p| {
                ctx.auxiliary("patches")
                    .excluding(self)
                    .including(&ALPRDCompressor)
                    .compress_patches(p)
            })
            .transpose()?;

        Ok(CompressedArray::compressed(
            ALPArray::try_new(compressed_encoded.array, exponents, compressed_patches)?
                .into_array(),
            Some(CompressionTree::new(self, vec![compressed_encoded.path])),
            array,
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingId> {
        HashSet::from([
            ALPEncoding::ID,
            // ALP-RD + BitPacking possibly used for patches
            ALPRDEncoding::ID,
            BitPackedEncoding::ID,
        ])
    }
}
