// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse encoding for fixed-size lists with many null rows.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_sparse::Sparse;
use vortex_sparse::patch_chunk_offsets;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::IntDictScheme;
use crate::schemes::integer::IntRLEScheme;
use crate::schemes::integer::RunEndScheme;
use crate::schemes::integer::SparseScheme as IntSparseScheme;
use crate::schemes::sparse_chunk_offsets::ChunkOffsetsScheme;
use crate::schemes::sparse_chunk_offsets::WithChunkOffsets;
use crate::schemes::sparse_chunk_offsets::compress_chunk_offsets;
use crate::schemes::sparse_chunk_offsets::refine_sparse;

/// The smallest estimated ratio of the dense list's size to the sparse one's worth encoding for.
const MIN_ESTIMATED_RATIO: f64 = 1.1;

/// Sparse encoding of a fixed-size list that stores only its valid rows.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FixedSizeListSparseScheme;

/// The variant of [`FixedSizeListSparseScheme`] that writes patch chunk offsets.
static WITH_CHUNK_OFFSETS: WithChunkOffsets<FixedSizeListSparseScheme> =
    WithChunkOffsets(FixedSizeListSparseScheme);

impl Scheme for FixedSizeListSparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.fixed_size_list.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::FixedSizeList(_)) && canonical.dtype().is_nullable()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Sparse.id()]
    }

    /// Children: values=0, indices=1.
    fn num_children(&self) -> usize {
        2
    }

    /// Sparse indices (child 1) are monotonically increasing positions with all unique values.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: RunEndScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: IntRLEScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: IntSparseScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    /// Compares the list's size with the valid rows plus their bit-packed positions.
    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let array = data.array();
        let len = array.len();
        let Ok(mask) = validity_mask(array, exec_ctx) else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        let valid = mask.true_count();
        if valid == 0 || valid == len {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let dense_bytes = array.nbytes() as f64;
        let row_bytes = dense_bytes / len as f64;
        let index_bytes = f64::from(usize::BITS - (len - 1).leading_zeros()) / 8.0;
        let sparse_bytes = valid as f64 * (row_bytes + index_bytes);

        let ratio = dense_bytes / sparse_bytes;
        if ratio < MIN_ESTIMATED_RATIO {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(ratio))
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

impl ChunkOffsetsScheme for FixedSizeListSparseScheme {
    fn compress_sparse(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
        with_chunk_offsets: bool,
    ) -> VortexResult<ArrayRef> {
        let array = data.array();
        let mask = validity_mask(array, exec_ctx)?;
        let AllOr::Some(valid_indices) = mask.indices() else {
            // The estimate skips lists without both valid and null rows.
            return Ok(array.clone());
        };

        let values = array.filter(mask.clone())?;
        let compressed_values =
            compressor.compress_child(&values, &compress_ctx, self.id(), 0, exec_ctx)?;

        let indices = valid_indices
            .iter()
            .map(|&idx| idx as u64)
            .collect::<Buffer<u64>>();
        let chunk_offsets = with_chunk_offsets
            .then(|| patch_chunk_offsets(&indices, array.len(), exec_ctx))
            .transpose()?;
        let indices = PrimitiveArray::new(indices, Validity::NonNullable).narrow(exec_ctx)?;
        let compressed_indices = compressor.compress_child(
            &indices.into_array(),
            &compress_ctx,
            self.id(),
            1,
            exec_ctx,
        )?;

        let chunk_offsets = compress_chunk_offsets(
            compressor,
            chunk_offsets.as_ref(),
            &compress_ctx,
            self,
            exec_ctx,
        )?;

        Ok(Sparse::try_new_with_chunk_offsets(
            compressed_indices,
            compressed_values,
            array.len(),
            Scalar::null(array.dtype().clone()),
            chunk_offsets,
            exec_ctx,
        )?
        .into_array())
    }
}

/// Returns which rows of `array` are valid.
fn validity_mask(array: &ArrayRef, exec_ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
    array.validity()?.execute_mask(array.len(), exec_ctx)
}
