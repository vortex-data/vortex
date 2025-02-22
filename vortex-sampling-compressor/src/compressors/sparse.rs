use vortex_array::aliases::hash_set::HashSet;
use vortex_array::{Array, Encoding, EncodingId};
use vortex_error::VortexResult;
use vortex_sparse::{SparseArray, SparseEncoding};

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct SparseCompressor;

impl EncodingCompressor for SparseCompressor {
    fn id(&self) -> &str {
        SparseEncoding::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::SPARSE_COST
    }

    fn can_compress(&self, array: &dyn Array) -> Option<&dyn EncodingCompressor> {
        array.is_encoding(SparseEncoding::ID).then_some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &dyn Array,
        _: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let sparse_array = SparseArray::try_from(array.to_array())?;
        let compressed_patches = ctx
            .auxiliary("patches")
            .compress_patches(sparse_array.patches().clone())?;
        Ok(CompressedArray::compressed(
            SparseArray::try_new_from_patches(
                compressed_patches,
                sparse_array.fill_scalar().clone(),
            )?
            .into_array(),
            Some(CompressionTree::new(self, vec![])),
            array,
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingId> {
        HashSet::from([SparseEncoding::ID])
    }
}
