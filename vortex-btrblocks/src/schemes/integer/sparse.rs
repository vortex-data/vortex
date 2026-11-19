// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse integer encoding for single-value-dominated arrays.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::scalar::Scalar;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt as _;

use super::IntRLEScheme;
use super::RunEndScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::GenerateStatsOptions;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::sparse_chunk_offsets::ChunkOffsetsScheme;
use crate::schemes::sparse_chunk_offsets::WithChunkOffsets;
use crate::schemes::sparse_chunk_offsets::compress_chunk_offsets;
use crate::schemes::sparse_chunk_offsets::encode_sparse;
use crate::schemes::sparse_chunk_offsets::refine_sparse;

/// Sparse encoding for single-value-dominated arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SparseScheme;

/// The variant of [`SparseScheme`] that writes patch chunk offsets.
static WITH_CHUNK_OFFSETS: WithChunkOffsets<SparseScheme> = WithChunkOffsets(SparseScheme);

impl Scheme for SparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Sparse.id(), Constant.id()]
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions {
            count_distinct_values: true,
        }
    }

    /// Children: values=0, indices=1.
    fn num_children(&self) -> usize {
        2
    }

    /// Sparse indices (child 1) are monotonically increasing positions with all unique values.
    /// Dict, RunEnd, RLE, and Sparse are all pointless on such data.
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
                excluded: SparseScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let len = data.array_len() as f64;
        let stats = data.integer_stats(exec_ctx);
        let value_count = stats.value_count();

        // All-null arrays should be compressed as constant instead anyways.
        if value_count == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // If the majority (90%) of values is null, this will compress well.
        if stats.null_count() as f64 / len > 0.9 {
            return CompressionEstimate::Verdict(EstimateVerdict::Ratio(len / value_count as f64));
        }

        let (_, most_frequent_count) = stats
            .erased()
            .most_frequent_value_and_count()
            .vortex_expect(
                "this must be present since `SparseScheme` declared that we need distinct values",
            );

        // If the most frequent value is the only value, we should compress as constant instead.
        if most_frequent_count == value_count {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        debug_assert!(value_count > most_frequent_count);

        // See if the most frequent value accounts for >= 90% of the set values.
        let freq = most_frequent_count as f64 / value_count as f64;
        if freq < 0.9 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // We only store the positions of the non-top values.
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            value_count as f64 / (value_count - most_frequent_count) as f64,
        ))
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

impl ChunkOffsetsScheme for SparseScheme {
    fn compress_sparse(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
        with_chunk_offsets: bool,
    ) -> VortexResult<ArrayRef> {
        let len = data.array_len();
        let stats = data.integer_stats(exec_ctx);
        let array = data.array();

        let (most_frequent_value, most_frequent_count) = stats
            .erased()
            .most_frequent_value_and_count()
            .vortex_expect(
                "this must be present since `SparseScheme` declared that we need distinct values",
            );

        if most_frequent_count as usize == len {
            // If the most frequent value is the only value, we should compress as constant instead.
            return Ok(ConstantArray::new(
                Scalar::primitive_value(
                    most_frequent_value,
                    most_frequent_value.ptype(),
                    array.dtype().nullability(),
                ),
                len,
            )
            .into_array());
        }

        let sparse_encoded = encode_sparse(
            array,
            Some(Scalar::primitive_value(
                most_frequent_value,
                most_frequent_value.ptype(),
                array.dtype().nullability(),
            )),
            with_chunk_offsets,
            exec_ctx,
        )?;

        if let Some(sparse) = sparse_encoded.as_opt::<Sparse>() {
            let sparse_values_primitive = sparse
                .patches()
                .values()
                .clone()
                .execute::<PrimitiveArray>(exec_ctx)?;
            let compressed_values = compressor.compress_child(
                &sparse_values_primitive.into_array(),
                &compress_ctx,
                self.id(),
                0,
                exec_ctx,
            )?;

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
                1,
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
                compressed_values,
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
