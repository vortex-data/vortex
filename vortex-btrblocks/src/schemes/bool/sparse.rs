// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse encoding for bool arrays dominated by a single value.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::scalar::Scalar;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt as _;

use crate::ArrayAndStats;
use crate::BoolStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::IntRLEScheme;
use crate::schemes::integer::RunEndScheme;
use crate::schemes::integer::SparseScheme as IntSparseScheme;

/// Sparse encoding for bool arrays where one value (or null) dominates.
///
/// Stores the positions that differ from the dominant value. This suits validity bitmaps with
/// only a few nulls, and flags that are rarely set.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SparseScheme;

impl Scheme for SparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.bool.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_boolean()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Sparse.id(), Constant.id()]
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
                excluded: IntSparseScheme.id(),
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
        let len = data.array_len();
        let stats = data.bool_stats(exec_ctx);

        // All-null and all-equal arrays are compressed as constant instead.
        let exceptions = exception_count(len, &stats);
        if stats.value_count() == 0 || exceptions == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // A nullable array stores a validity bit next to each value bit, both in the canonical
        // bitmap and in the sparse values.
        let bits_per_value = if stats.null_count() > 0 { 2 } else { 1 };
        // Indices are narrowed and then bitpacked to the width of the largest position.
        let index_bits = (usize::BITS - (len - 1).leading_zeros()).max(1) as usize;

        let canonical_bits = len * bits_per_value;
        let sparse_bits = exceptions * (index_bits + bits_per_value);

        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            canonical_bits as f64 / sparse_bits as f64,
        ))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let stats = data.bool_stats(exec_ctx);
        let array = data.array();

        // `Sparse::encode` switches to a null fill on its own when nulls dominate.
        let fill = Scalar::bool(majority_value(&stats), array.dtype().nullability());
        let sparse_encoded = Sparse::encode(array, Some(fill), exec_ctx)?;

        let Some(sparse) = sparse_encoded.as_opt::<Sparse>() else {
            return Ok(sparse_encoded);
        };

        let compressed_values = compressor.compress_child(
            sparse.patches().values(),
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

        Sparse::try_new(
            compressed_indices,
            compressed_values,
            sparse.len(),
            sparse.fill_scalar().clone(),
        )
        .map(|a| a.into_array())
    }
}

/// Returns the more frequent of the valid values, preferring `true` on a tie.
fn majority_value(stats: &BoolStats) -> bool {
    2 * stats.true_count() >= stats.value_count()
}

/// Returns the number of positions `Sparse::encode` stores as patches.
///
/// This mirrors its fill selection: null when more than 90% of the values are null, otherwise
/// the majority value, in which case every null and minority value is a patch.
fn exception_count(len: usize, stats: &BoolStats) -> usize {
    let value_count = stats.value_count() as usize;
    if stats.null_count() as f64 > 0.9 * len as f64 {
        return value_count;
    }

    let true_count = stats.true_count() as usize;
    let majority_count = true_count.max(value_count - true_count);
    len - majority_count
}
