use vortex_array::aliases::hash_set::HashSet;
use vortex_array::arrays::{PrimitiveEncoding, VarBinEncoding, VarBinViewEncoding};
use vortex_array::{Array, Encoding, EncodingId};
use vortex_dict::builders::dict_encode;
use vortex_dict::{DictArray, DictEncoding};
use vortex_error::VortexResult;

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::downscale::downscale_integer_array;
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct DictCompressor;

impl EncodingCompressor for DictCompressor {
    fn id(&self) -> &str {
        DictEncoding::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::DICT_COST
    }

    fn can_compress(&self, array: &dyn Array) -> Option<&dyn EncodingCompressor> {
        if !array.is_encoding(PrimitiveEncoding::ID)
            && !array.is_encoding(VarBinEncoding::ID)
            && !array.is_encoding(VarBinViewEncoding::ID)
        {
            return None;
        };

        // No point dictionary coding if the array is unique.
        // We don't have a unique stat yet, but strict-sorted implies unique.
        if array
            .statistics()
            .compute_is_strict_sorted()
            .unwrap_or(false)
        {
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
        let dict = dict_encode(array)?;
        let codes = dict.codes();
        let values = dict.values();

        let (codes, values) = (
            ctx.auxiliary("codes").excluding(self).compress(
                &downscale_integer_array(codes)?,
                like.as_ref().and_then(|l| l.child(0)),
            )?,
            ctx.named("values")
                .excluding(self)
                .compress(values, like.as_ref().and_then(|l| l.child(1)))?,
        );

        Ok(CompressedArray::compressed(
            DictArray::try_new(codes.array, values.array)?.into_array(),
            Some(CompressionTree::new(self, vec![codes.path, values.path])),
            array,
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingId> {
        HashSet::from([DictEncoding::ID])
    }
}
