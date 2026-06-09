// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run-length float encoding.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::schemes::integer::RUN_LENGTH_THRESHOLD;
use crate::schemes::integer::rle_compress;
use crate::schemes::rle_ancestor_exclusions;
use crate::schemes::rle_descendant_exclusions;

/// RLE scheme for float arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatRLEScheme;

impl Scheme for FloatRLEScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.rle"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    /// Children: values=0, indices=1, offsets=2.
    fn num_children(&self) -> usize {
        3
    }

    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        rle_descendant_exclusions()
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        rle_ancestor_exclusions()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // RLE is only useful when we cascade it with another encoding.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        if data.float_stats(exec_ctx).average_run_length() < RUN_LENGTH_THRESHOLD {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        rle_compress(self, compressor, data, compress_ctx, exec_ctx)
    }
}
