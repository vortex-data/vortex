// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Constant encoding for string arrays.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_error::VortexResult;

use super::is_utf8_string;
use crate::CascadingCompressor;
use crate::builtins::StringConstantScheme;
use crate::builtins::constant::compress_constant_array_with_validity;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::DeferredEstimate;
use crate::estimate::EstimateVerdict;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;

impl Scheme for StringConstantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.constant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_utf8_string(canonical)
    }

    fn expected_compression_ratio(
        &self,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> CompressionEstimate {
        // Constant detection on a sample is a false positive, since the sample being constant does
        // not mean the full array is constant.
        if ctx.is_sample() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let array_len = data.array().len();
        let stats = data.string_stats();

        // We want to use `Constant` if there are only nulls in the array.
        if stats.value_count() == 0 {
            debug_assert_eq!(stats.null_count() as usize, array_len);
            return CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse);
        }

        // Since the estimated distinct count is always going to be less than or equal to the actual
        // distinct count, if this is not equal to 1 the actual is definitely not equal to 1.
        if stats.estimated_distinct_count().is_some_and(|c| c > 1) {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Otherwise our best bet is to actually check if the array is constant.
        // This is an expensive check, but the alternative of not compressing a constant array is
        // far less preferable.
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |compressor, data, _ctx| {
                if is_constant(data.array(), &mut compressor.execution_ctx())? {
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
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        compress_constant_array_with_validity(data.array())
    }
}
