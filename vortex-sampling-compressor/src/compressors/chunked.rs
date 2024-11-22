use std::any::Any;
use std::sync::Arc;

use log::info;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::{Chunked, ChunkedArray};
use vortex_array::compress::compute_precompression_stats;
use vortex_array::encoding::EncodingRef;
use vortex_array::stats::ArrayStatistics;
use vortex_array::{ArrayDType, ArrayData, ArrayDef, IntoArrayData};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use super::EncoderMetadata;
use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct ChunkedCompressor {
    relatively_good_ratio: f32,
}

pub const DEFAULT_CHUNKED_COMPRESSOR: ChunkedCompressor = ChunkedCompressor {
    relatively_good_ratio: 1.2,
};

pub struct ChunkedCompressorMetadata(Option<f32>);

impl EncoderMetadata for ChunkedCompressorMetadata {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl EncodingCompressor for ChunkedCompressor {
    fn id(&self) -> &str {
        Chunked::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::CHUNKED_COST
    }

    fn can_compress(&self, array: &ArrayData) -> Option<&dyn EncodingCompressor> {
        array.is_encoding(Chunked::ID).then_some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &ArrayData,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let chunked_array = ChunkedArray::try_from(array.clone())?;
        self.compress_chunked(&chunked_array, like, ctx)
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([])
    }
}

impl ChunkedCompressor {
    /// How far the compression ratio is allowed to grow from one chunk to another chunk.
    ///
    /// As long as a compressor compresses subsequent chunks "reasonably well" we should continue to
    /// use it, which saves us the cost of searching for a good compressor. This constant quantifies
    /// "reasonably well" as
    ///
    /// ```text
    /// new_ratio <= old_ratio * self.relatively_good_ratio
    /// ```
    fn relatively_good_ratio(&self) -> f32 {
        self.relatively_good_ratio
    }

    fn compress_chunked<'a>(
        &'a self,
        array: &ChunkedArray,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let less_chunked = array.rechunk(
            ctx.options().target_block_bytesize,
            ctx.options().target_block_size,
        )?;

        let mut previous = like_into_parts(like)?;
        let mut compressed_chunks = Vec::with_capacity(less_chunked.nchunks());
        let mut compressed_trees = Vec::with_capacity(less_chunked.nchunks() + 1);
        compressed_trees.push(None); // for the chunk offsets

        for (index, chunk) in less_chunked.chunks().enumerate() {
            // these are extremely valuable when reading/writing, but are potentially much more expensive
            // to compute post-compression. That's because not all encodings implement stats, so we would
            // potentially have to canonicalize during writes just to get stats, which would be silly.
            // Also, we only really require them for column chunks, not for every array.
            compute_precompression_stats(&chunk)?;

            let like = previous.as_ref().map(|(like, _)| like);
            let (compressed_chunk, tree) = ctx
                .named(&format!("chunk-{}", index))
                .compress(&chunk, like)?
                .into_parts();

            let ratio = (compressed_chunk.nbytes() as f32) / (chunk.nbytes() as f32);
            let exceeded_target_ratio = previous
                .as_ref()
                .map(|(_, target_ratio)| ratio > target_ratio * self.relatively_good_ratio())
                .unwrap_or(false);

            if ratio > 1.0 || exceeded_target_ratio {
                info!("unsatisfactory ratio {}, previous: {:?}", ratio, previous);
                let (compressed_chunk, tree) = ctx.compress_array(&chunk)?.into_parts();
                let new_ratio = (compressed_chunk.nbytes() as f32) / (chunk.nbytes() as f32);

                compressed_chunks.push(compressed_chunk);
                compressed_trees.push(tree.clone());
                previous = tree.map(|tree| (tree, new_ratio));
            } else {
                compressed_chunks.push(compressed_chunk);
                compressed_trees.push(tree.clone());
                previous = previous.or_else(|| tree.map(|tree| (tree, ratio)));
            }
        }

        let ratio = previous.map(|(_, ratio)| ratio);
        Ok(CompressedArray::compressed(
            ChunkedArray::try_new(compressed_chunks, array.dtype().clone())?.into_array(),
            Some(CompressionTree::new_with_metadata(
                self,
                compressed_trees,
                Arc::new(ChunkedCompressorMetadata(ratio)),
            )),
            Some(array.statistics()),
        ))
    }
}

fn like_into_parts(
    tree: Option<CompressionTree<'_>>,
) -> VortexResult<Option<(CompressionTree<'_>, f32)>> {
    let (_, mut children, metadata) = match tree {
        None => return Ok(None),
        Some(tree) => tree.into_parts(),
    };

    // must have one for the chunk offsets and one per chunk (and at least one chunk!)
    if children.len() < 2 {
        vortex_bail!("Chunked array compression tree must have at least two children")
    }

    // since we compress sequentially, we take the last child as the previous (and thus presumably most-similar) chunk
    let latest_child = children
        .pop()
        .vortex_expect("Unreachable: tree must have at least two children");

    let Some(target_ratio) = metadata else {
        vortex_bail!("Chunked array compression tree must have metadata")
    };
    let Some(ChunkedCompressorMetadata(target_ratio)) =
        target_ratio.as_ref().as_any().downcast_ref()
    else {
        vortex_bail!("Chunked array compression tree must be ChunkedCompressorMetadata")
    };

    match (latest_child, target_ratio) {
        (None, None) => Ok(None),
        (Some(child), Some(ratio)) => Ok(Some((child, *ratio))),
        (..) => vortex_bail!("Chunked array compression tree must have a child iff it has a ratio"),
    }
}
