// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Constant encoding for integer arrays.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_error::VortexResult;

use super::is_integer_primitive;
use crate::CascadingCompressor;
use crate::builtins::IntConstantScheme;
use crate::builtins::constant::compress_constant_array_with_validity;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::EstimateVerdict;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;

impl Scheme for IntConstantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.constant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
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
        let stats = data.integer_stats();

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

        // Otherwise, use the max and min to determine if there is a single value.
        match stats.erased().max_minus_min().checked_ilog2() {
            Some(_) => CompressionEstimate::Verdict(EstimateVerdict::Skip),
            // If max-min == 0, then we know that there is only 1 value.
            None => CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse),
        }
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
