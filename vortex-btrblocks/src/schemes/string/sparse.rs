// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse encoding for null-dominated string arrays.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::PType;
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

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Sparse.id()]
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
        let len = data.array_len();
        let stats = data.varbinview_stats(exec_ctx);
        let value_count = stats.value_count() as usize;

        // All-null arrays should be compressed as constant instead anyways.
        if value_count == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Only null-dominated arrays (90% null) go this route.
        if stats.null_count() as f64 / len as f64 <= 0.9 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Sparse drops the views of null rows and adds an index per value. It keeps every value's
        // bytes, so long values bound the ratio well below the fraction of rows that are null.
        let Ok(before) = uncompressed_size_in_bytes(data.array(), exec_ctx) else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        let null_views = (len - value_count) * size_of::<BinaryView>();
        let indices = value_count * PType::min_unsigned_ptype_for_value(len as u64).byte_width();
        let after = before.saturating_sub(null_views) + indices;
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(before as f64 / after as f64))
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
                exec_ctx,
            )
            .map(|a| a.into_array())
        } else {
            Ok(sparse_encoded)
        }
    }
}
