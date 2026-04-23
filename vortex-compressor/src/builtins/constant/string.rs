// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Constant encoding for string arrays.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::builtins::constant::compress_constant_array_with_validity;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::DeferredEstimate;
use crate::estimate::EstimateVerdict;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;

/// Constant encoding for string arrays with a single distinct value.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StringConstantScheme;

impl Scheme for StringConstantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.constant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // Constant detection on a sample is a false positive, since the sample being constant does
        // not mean the full array is constant.
        if compress_ctx.is_sample() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let array_len = data.array().len();
        let stats = data.varbinview_stats(exec_ctx);

        // We want to use `Constant` if there are only nulls in the array.
        if stats.value_count() == 0 {
            debug_assert_eq!(stats.null_count(), array_len);
            return CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse);
        }

        // The estimator is exact for small cardinalities, so a constant array (exactly one
        // distinct value) is always reported as 1. An estimate above 1 therefore means the array
        // is genuinely not constant; an estimate of 1 still falls through to the exact check below.
        if stats.estimated_distinct_count().is_some_and(|c| c > 1) {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Otherwise our best bet is to actually check if the array is constant.
        // This is an expensive check, but the alternative of not compressing a constant array is
        // far less preferable.
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, _best_so_far, _ctx, exec_ctx| {
                if is_constant(data.array(), exec_ctx)? {
                    Ok(EstimateVerdict::AlwaysUse)
                } else {
                    Ok(EstimateVerdict::Skip)
                }
            },
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        compress_constant_array_with_validity(data.array(), exec_ctx)
    }
}
