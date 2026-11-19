// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse encoding for null-dominated float arrays.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt as _;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::SparseScheme as IntSparseScheme;
use crate::schemes::sparse_chunk_offsets::ChunkOffsetsScheme;
use crate::schemes::sparse_chunk_offsets::WithChunkOffsets;
use crate::schemes::sparse_chunk_offsets::compress_chunk_offsets;
use crate::schemes::sparse_chunk_offsets::encode_sparse;
use crate::schemes::sparse_chunk_offsets::refine_sparse;

/// Sparse encoding for null-dominated float arrays.
///
/// This is the same as the integer `SparseScheme`, but we only use this for null-dominated arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NullDominatedSparseScheme;

/// The variant of [`NullDominatedSparseScheme`] that writes patch chunk offsets.
static WITH_CHUNK_OFFSETS: WithChunkOffsets<NullDominatedSparseScheme> =
    WithChunkOffsets(NullDominatedSparseScheme);

impl Scheme for NullDominatedSparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Sparse.id()]
    }

    /// Children: indices=0.
    fn num_children(&self) -> usize {
        1
    }

    /// The indices of a null-dominated sparse array should not be sparse-encoded again.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![DescendantExclusion {
            excluded: IntSparseScheme.id(),
            children: ChildSelection::All,
        }]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let len = data.array_len() as f64;
        let stats = data.float_stats(exec_ctx);
        let value_count = stats.value_count();

        // All-null arrays should be compressed as constant instead anyways.
        if value_count == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // If the majority (90%) of values is null, this will compress well.
        if stats.null_count() as f64 / len > 0.9 {
            return CompressionEstimate::Verdict(EstimateVerdict::Ratio(len / value_count as f64));
        }

        // Otherwise we don't go this route.
        CompressionEstimate::Verdict(EstimateVerdict::Skip)
    }

    fn refine(&self, allowed: &dyn Fn(&ArrayId) -> bool) -> &dyn Scheme {
        refine_sparse(self, &WITH_CHUNK_OFFSETS, allowed)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.compress_sparse(compressor, data, compress_ctx, exec_ctx, false)
    }
}

impl ChunkOffsetsScheme for NullDominatedSparseScheme {
    fn compress_sparse(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
        with_chunk_offsets: bool,
    ) -> VortexResult<ArrayRef> {
        // We pass None as we only run this pathway for NULL-dominated float arrays.
        let sparse_encoded = encode_sparse(data.array(), None, with_chunk_offsets, exec_ctx)?;

        if let Some(sparse) = sparse_encoded.as_opt::<Sparse>() {
            let indices = sparse
                .patches()
                .indices()
                .clone()
                .execute::<PrimitiveArray>(exec_ctx)?
                .narrow(exec_ctx)?;
            let compressed_indices = compressor.compress_child(
                &indices.into_array(),
                &compress_ctx,
                self.id(),
                0,
                exec_ctx,
            )?;

            let chunk_offsets = compress_chunk_offsets(
                compressor,
                sparse.patches().chunk_offsets().as_ref(),
                &compress_ctx,
                self,
                exec_ctx,
            )?;

            Sparse::try_new_with_chunk_offsets(
                compressed_indices,
                sparse.patches().values().clone(),
                sparse.len(),
                sparse.fill_scalar().clone(),
                chunk_offsets,
                exec_ctx,
            )
            .map(|a| a.into_array())
        } else {
            Ok(sparse_encoded)
        }
    }
}
