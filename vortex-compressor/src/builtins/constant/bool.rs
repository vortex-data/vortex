// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Constant encoding for bool arrays.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::builtins::BoolConstantScheme;
use crate::builtins::constant::compress_constant_array_with_validity;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::EstimateVerdict;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;

impl Scheme for BoolConstantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.bool.constant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Bool(_))
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
        let stats = data.bool_stats();

        // We want to use `Constant` if there are only nulls in the array.
        if stats.value_count() == 0 {
            debug_assert_eq!(stats.null_count() as usize, array_len);
            return CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse);
        }

        if stats.is_constant() {
            return CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse);
        }

        CompressionEstimate::Verdict(EstimateVerdict::Skip)
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
