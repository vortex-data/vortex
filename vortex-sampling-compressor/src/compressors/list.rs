use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::{ListArray, ListEncoding};
use vortex_array::{Array, Encoding, EncodingId, IntoArray};
use vortex_error::VortexResult;

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::downscale::downscale_integer_array;
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct ListCompressor;

impl EncodingCompressor for ListCompressor {
    fn id(&self) -> &str {
        ListEncoding::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::LIST_COST
    }

    fn can_compress(&self, array: &Array) -> Option<&dyn EncodingCompressor> {
        array.is_encoding(ListEncoding::ID).then_some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &Array,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let list_array = ListArray::try_from(array.clone())?;
        let compressed_elements = ctx.named("elements").compress(
            &list_array.elements(),
            like.as_ref().and_then(|l| l.child(0)),
        )?;
        let compressed_offsets = ctx.auxiliary("offsets").compress(
            &downscale_integer_array(list_array.offsets())?,
            like.as_ref().and_then(|l| l.child(1)),
        )?;
        Ok(CompressedArray::compressed(
            ListArray::try_new(
                compressed_elements.array,
                compressed_offsets.array,
                list_array.validity(),
            )?
            .into_array(),
            Some(CompressionTree::new(
                self,
                vec![compressed_elements.path, compressed_offsets.path, None],
            )),
            array,
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingId> {
        HashSet::from([ListEncoding::ID])
    }
}
