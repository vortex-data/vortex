use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::{VarBinArray, VarBinEncoding};
use vortex_array::{Array, Encoding, EncodingId, IntoArray};
use vortex_error::VortexResult;

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::downscale::downscale_integer_array;
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct VarBinCompressor;

impl EncodingCompressor for VarBinCompressor {
    fn id(&self) -> &str {
        VarBinEncoding::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::VARBIN_COST
    }

    fn can_compress(&self, array: &Array) -> Option<&dyn EncodingCompressor> {
        array.is_encoding(VarBinEncoding::ID).then_some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &Array,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let varbin_array = VarBinArray::try_from(array.clone())?;
        let offsets = ctx.auxiliary("offsets").compress(
            &downscale_integer_array(varbin_array.offsets())?,
            like.as_ref().and_then(|l| l.child(0)),
        )?;
        Ok(CompressedArray::compressed(
            VarBinArray::try_new(
                offsets.array,
                varbin_array.bytes(), // we don't compress the raw bytes
                array.dtype().clone(),
                varbin_array.validity(),
            )?
            .into_array(),
            Some(CompressionTree::new(self, vec![offsets.path, None, None])),
            array,
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingId> {
        HashSet::from([VarBinEncoding::ID])
    }
}
