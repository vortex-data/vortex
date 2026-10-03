// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Entropy-coded bins for integers.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_entropy_bins::EntropyBins;
use vortex_entropy_bins::EntropyBinsConfig;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::schemes::integer::RUN_LENGTH_THRESHOLD;

/// Typical cost of one run under RunEnd in half-bytes: a value plus a delta-coded end, ~12 bits.
const HALF_BYTES_PER_RUN: usize = 3;

/// Entropy-coded bins: pco's bins with a SIMD tANS id stream and variable-width offsets, in
/// independently decodable blocks. Opt-in: add a preset (or your own dials) with
/// [`with_new_scheme`](crate::BtrBlocksCompressorBuilder::with_new_scheme), e.g.
/// `with_new_scheme(&EntropyBinsScheme::BALANCED)`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EntropyBinsScheme {
    /// Layout dials passed to [`EntropyBins::plan`].
    pub config: EntropyBinsConfig,
    /// How much smaller (in percent) than the best other scheme's estimate this scheme's must
    /// be. Entropy-coded blocks decode slower than bit-packing, so the default asks for 10%.
    pub min_gain_percent: u32,
}

impl EntropyBinsScheme {
    /// [`EntropyBinsConfig::BALANCED`], used where it beats other schemes by 10%.
    pub const BALANCED: Self = Self {
        config: EntropyBinsConfig::BALANCED,
        min_gain_percent: 10,
    };

    /// [`EntropyBinsConfig::FAST`], used only where it beats other schemes by 25%.
    pub const FAST: Self = Self {
        config: EntropyBinsConfig::FAST,
        min_gain_percent: 25,
    };

    /// [`EntropyBinsConfig::SMALLEST`], used wherever it is smaller than other schemes.
    pub const SMALLEST: Self = Self {
        config: EntropyBinsConfig::SMALLEST,
        min_gain_percent: 0,
    };
}

impl Default for EntropyBinsScheme {
    fn default() -> Self {
        Self::BALANCED
    }
}

impl Scheme for EntropyBinsScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.entropy_bins"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![EntropyBins.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // Sampling would train the bins on the sample and score that same sample, which
        // overfits clustered samples of sorted or drifting data. The callback scores bins on
        // blocks they were not trained on instead.
        let config = self.config;
        let min_gain = 1.0 + f64::from(self.min_gain_percent) / 100.0;
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |_compressor, data, best_so_far, _ctx, exec_ctx| {
                let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
                let raw = primitive.len() * primitive.ptype().byte_width();
                let estimate = EntropyBins::plan(primitive.as_view(), &config)?.nbytes;
                // RunEnd's sampled estimate cuts runs at every 64-row sample edge, so on
                // run-heavy data it looks worse than it is. Leave such arrays to RunEnd (whose
                // children may still use this scheme) when the runs are clearly cheaper.
                let run_length = data.integer_stats(exec_ctx).average_run_length();
                if run_length >= RUN_LENGTH_THRESHOLD {
                    let runs = primitive.len() / run_length as usize;
                    if estimate > runs * HALF_BYTES_PER_RUN / 2 {
                        return Ok(EstimateVerdict::Skip);
                    }
                }
                let ratio = raw as f64 / estimate.max(1) as f64;
                // Entropy-coded blocks decode slower than bit-packing: require a clear win.
                let threshold = best_so_far.and_then(EstimateScore::finite_ratio);
                if ratio <= 1.0 || threshold.is_some_and(|t| ratio <= t * min_gain) {
                    return Ok(EstimateVerdict::Skip);
                }
                Ok(EstimateVerdict::Ratio(ratio))
            },
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = data.array_as_primitive();
        let plan = EntropyBins::plan(primitive, &self.config)?;
        // Bins that do not fit the encoding's limits leave the array as it is.
        match EntropyBins::from_primitive(primitive, self.config.level, plan.options) {
            Ok(array) => Ok(array.into_array()),
            Err(_) => Ok(primitive.array().clone()),
        }
    }
}
