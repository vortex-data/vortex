// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse encoding for null-dominated string arrays.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_error::VortexResult;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt as _;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::IntDictScheme;
use crate::schemes::integer::SparseScheme as IntSparseScheme;

/// Sparse encoding for null-dominated arrays.
///
/// This is the same as the integer `SparseScheme`, but we only use this for null-dominated arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NullDominatedSparseScheme;

impl Scheme for NullDominatedSparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    /// Children: indices=0.
    fn num_children(&self) -> usize {
        1
    }

    /// The indices of a null-dominated sparse array should not be sparse-encoded again.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntSparseScheme.id(),
                children: ChildSelection::All,
            },
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::All,
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
        let stats = data.varbinview_stats(exec_ctx);
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

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // We pass None as we only run this pathway for NULL-dominated string arrays.
        let sparse_encoded = Sparse::encode(data.array(), None, exec_ctx)?;

        if let Some(sparse) = sparse_encoded.as_opt::<Sparse>() {
            // Compress the indices only (not the values for strings).
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

            Sparse::try_new(
                compressed_indices,
                sparse.patches().values().clone(),
                sparse.len(),
                sparse.fill_scalar().clone(),
            )
            .map(|a| a.into_array())
        } else {
            Ok(sparse_encoded)
        }
    }
}
