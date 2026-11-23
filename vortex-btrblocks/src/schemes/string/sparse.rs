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
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::PType;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateScore;
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

/// The number of consecutive values in each sampled run of values.
const SAMPLE_RUN_LEN: usize = 64;

/// The number of runs of values sampled to estimate how well the values compress.
const SAMPLE_RUNS: usize = 16;

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

    /// Children: values=0, indices=1.
    fn num_children(&self) -> usize {
        2
    }

    /// Sparse indices (child 1) are monotonically increasing positions with all unique values, so
    /// Sparse and Dict are pointless on them.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntSparseScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
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

        // Sparse drops the views of null rows and adds an index per value.
        let Ok(before) = uncompressed_size_in_bytes(data.array(), exec_ctx) else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        let null_views = (len - value_count) * size_of::<BinaryView>();
        let values = before.saturating_sub(null_views);
        let indices = value_count * PType::min_unsigned_ptype_for_value(len as u64).byte_width();

        // Without a cascade the values keep all their bytes, so long values bound the ratio well
        // below the fraction of rows that are null.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Ratio(
                before as f64 / (values + indices) as f64,
            ));
        }

        // Otherwise the values shrink by whatever the other string schemes achieve on them, which
        // only compressing some of them can tell.
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |compressor, data, best_so_far, compress_ctx, exec_ctx| {
                // The indices alone bound the ratio, however well the values compress.
                let max_ratio = before as f64 / indices as f64;
                let threshold = best_so_far.and_then(EstimateScore::finite_ratio);
                if threshold.is_some_and(|threshold| max_ratio <= threshold) {
                    return Ok(EstimateVerdict::Skip);
                }

                let sample = sample_values(data.array(), exec_ctx)?;
                let sample_before = uncompressed_size_in_bytes(&sample, exec_ctx)?;
                let sample_after = compressor
                    .compress_child(
                        &sample,
                        &compress_ctx,
                        NullDominatedSparseScheme.id(),
                        0,
                        exec_ctx,
                    )?
                    .nbytes();
                if sample_before == 0 || sample_after == 0 {
                    return Ok(EstimateVerdict::Skip);
                }

                let compressed_values = values as f64 * sample_after as f64 / sample_before as f64;
                Ok(EstimateVerdict::Ratio(
                    before as f64 / (compressed_values + indices as f64),
                ))
            },
        )))
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
            // The values are the valid strings alone, so the other string schemes apply to them.
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
                exec_ctx,
            )
            .map(|a| a.into_array())
        } else {
            Ok(sparse_encoded)
        }
    }
}

/// Returns evenly spaced runs of the valid values of `array`, or all of them if they are few.
fn sample_values(array: &ArrayRef, exec_ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let mask = array.validity()?.execute_mask(array.len(), exec_ctx)?;
    let values = array.filter(mask)?;
    let count = values.len();
    if count <= SAMPLE_RUNS * SAMPLE_RUN_LEN {
        return Ok(values);
    }

    let runs = (0..SAMPLE_RUNS)
        .map(|run| {
            let start = run * count / SAMPLE_RUNS;
            values.slice(start..start + SAMPLE_RUN_LEN)
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(ChunkedArray::try_new(runs, values.dtype().clone())?.into_array())
}
