// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Constant encoding for float arrays.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_error::VortexResult;

use super::is_float_primitive;
use crate::CascadingCompressor;
use crate::builtins::FloatConstantScheme;
use crate::builtins::constant::compress_constant_array_with_validity;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::DeferredEstimate;
use crate::estimate::EstimateVerdict;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;

impl Scheme for FloatConstantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.constant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_float_primitive(canonical)
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
        let stats = data.float_stats();

        // Note that we only compute distinct counts if other schemes have requested it.
        if let Some(distinct_count) = stats.distinct_count() {
            if distinct_count > 1 {
                return CompressionEstimate::Verdict(EstimateVerdict::Skip);
            } else {
                debug_assert_eq!(distinct_count, 1);
                return CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse);
            }
        }

        // We want to use `Constant` if there are only nulls in the array.
        if stats.value_count() == 0 {
            debug_assert_eq!(stats.null_count() as usize, array_len);
            return CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse);
        }

        // TODO(connor): Can we be smart here with the max and min like with integers?

        // Otherwise our best bet is to actually check if the array is constant.
        // This is an expensive check, but in practice the distinct count is known because we often
        // include dictionary encoding in our set of schemes, so we rarely call this.
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
