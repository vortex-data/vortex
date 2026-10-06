// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Entropy-coded bins for integers.

use std::cell::RefCell;
use std::collections::VecDeque;

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_entropy_bins::EntropyBins;
use vortex_entropy_bins::EntropyBinsConfig;
use vortex_entropy_bins::EntropyBinsPlan;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::schemes::integer::RUN_LENGTH_THRESHOLD;

/// Plans kept from recent estimates.
const CACHED_PLANS: usize = 16;

thread_local! {
    /// Plans from recent estimates, so `compress` does not plan the array it was just estimated
    /// on again. A stale hit only costs size: any plan encodes any array correctly.
    static PLANS: RefCell<VecDeque<(PlanKey, EntropyBinsPlan)>> =
        RefCell::new(VecDeque::with_capacity(CACHED_PLANS));
}

/// Identifies an array's values: their buffer, length, type and a few of the values.
#[derive(Clone, Copy, PartialEq, Eq)]
struct PlanKey {
    addr: usize,
    len: usize,
    ptype: PType,
    fingerprint: [i128; 3],
    config: EntropyBinsConfig,
}

fn plan_key(primitive: ArrayView<'_, Primitive>, config: &EntropyBinsConfig) -> PlanKey {
    match_each_integer_ptype!(primitive.ptype(), |T| {
        let values = primitive.as_slice::<T>();
        let at = |i: usize| values.get(i).map_or(0, |&v| i128::from(v));
        PlanKey {
            addr: values.as_ptr().addr(),
            len: values.len(),
            ptype: primitive.ptype(),
            fingerprint: [
                at(0),
                at(values.len() / 2),
                at(values.len().saturating_sub(1)),
            ],
            config: *config,
        }
    })
}

/// [`EntropyBins::plan`], reusing the plan of a recent estimate on the same array.
fn cached_plan(
    primitive: ArrayView<'_, Primitive>,
    config: &EntropyBinsConfig,
) -> VortexResult<EntropyBinsPlan> {
    let key = plan_key(primitive, config);
    if let Some(plan) = PLANS.with_borrow(|plans| {
        plans
            .iter()
            .find_map(|(k, plan)| (*k == key).then_some(*plan))
    }) {
        return Ok(plan);
    }
    let plan = EntropyBins::plan(primitive, config)?;
    PLANS.with_borrow_mut(|plans| {
        if plans.len() == CACHED_PLANS {
            plans.pop_front();
        }
        plans.push_back((key, plan));
    });
    Ok(plan)
}

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
    /// be. This is the scan-speed dial: entropy-coded blocks decode slower than bit-packing, so
    /// raising it trades size for decode throughput (on 64-bit columns, 25% instead of 10%
    /// decodes ~6% faster for ~3% more bytes).
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
                let estimate = cached_plan(primitive.as_view(), &config)?.nbytes;
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
        let plan = cached_plan(primitive, &self.config)?;
        Ok(EntropyBins::from_primitive(primitive, self.config.level, plan.options)?.into_array())
    }
}
