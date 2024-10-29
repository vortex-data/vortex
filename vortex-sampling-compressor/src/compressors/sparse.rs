use vortex::aliases::hash_set::HashSet;
use vortex::array::{Sparse, SparseArray, SparseEncoding};
use vortex::encoding::EncodingRef;
use vortex::{Array, ArrayDef, IntoArray};
use vortex_error::VortexResult;

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct SparseCompressor;

impl EncodingCompressor for SparseCompressor {
    fn id(&self) -> &str {
        Sparse::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::SPARSE_COST
    }

    fn can_compress(&self, array: &Array) -> Option<&dyn EncodingCompressor> {
        array.is_encoding(Sparse::ID).then_some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &Array,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let sparse_array = SparseArray::try_from(array)?;
        let indices = ctx.auxiliary("indices").compress(
            &sparse_array.indices(),
            like.as_ref().and_then(|l| l.child(0)),
        )?;
        let values = ctx.named("values").compress(
            &sparse_array.values(),
            like.as_ref().and_then(|l| l.child(0)),
        )?;
        Ok(CompressedArray::new(
            SparseArray::try_new(
                indices.array,
                values.array,
                sparse_array.len(),
                sparse_array.fill_value().clone(),
            )?
            .into_array(),
            Some(CompressionTree::new(self, vec![indices.path, values.path])),
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&SparseEncoding as EncodingRef])
    }
}
