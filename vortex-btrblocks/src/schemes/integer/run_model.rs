// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An opt-in model that ranks RunEnd, RLE and Sparse from bit widths and decode cost.
//!
//! By default the compressor skips RunEnd and RLE below an average run of 4 and Sparse below a
//! 90% top value, then samples the survivors. Both cutoffs ignore value width: wide values pay off
//! with much shorter runs or a less dominant top value. And a 1,024-value sample breaks the runs
//! and sequences that RunEnd children cascade into, so the sample misranks them.
//!
//! [`RunModel`] replaces those estimates with closed-form sizes, in bits:
//!
//! - RunEnd: `runs × (value bits + position bits)`.
//! - RLE: `(runs + chunks) × value bits + len × index bits`, where the index width covers the runs
//!   in one 1,024-value chunk.
//! - Sparse: `patches × (value bits + position bits)`.
//!
//! With a bandwidth set, each estimate is also charged its decode time beyond bit-packing's,
//! converted into the bytes that time could have transferred. RunEnd decodes per run, so it costs
//! far more than bit-packing on short runs; RLE decodes close to bit-packing's speed. A size-only
//! model ([`RunModel::size_only`]) picks the smallest output whatever it costs to decode.
//!
//! Enable it with
//! [`BtrBlocksCompressorBuilder::with_run_model`](crate::BtrBlocksCompressorBuilder::with_run_model).

use std::sync::LazyLock;

use parking_lot::Mutex;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;

use super::IntRLEScheme;
use super::RunEndScheme;
use super::SparseScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::GenerateStatsOptions;
use crate::Scheme;

/// Values in one FastLanes chunk, the unit RLE indexes within.
const RLE_CHUNK_LEN: f64 = 1024.0;

/// Closed-form size and decode-cost model for RunEnd, RLE and Sparse.
///
/// The default decode costs were measured on 65,536-value i64 arrays on a 4-core x86 machine.
/// They are coarse: set them from your own measurements where it matters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunModel {
    /// Bytes per second read from storage. `None` ranks by size alone.
    pub bandwidth: Option<f64>,
    /// Bit-packing's decode time per value, the baseline the other costs are charged against.
    pub baseline_ns_per_value: f64,
    /// RunEnd's decode time per run.
    pub runend_ns_per_run: f64,
    /// RLE's decode time per value.
    pub rle_ns_per_value: f64,
    /// Sparse's decode time per value, for filling the array with the top value.
    pub sparse_ns_per_value: f64,
    /// Sparse's decode time per patch.
    pub sparse_ns_per_patch: f64,
    /// RLE is skipped below this average run length. Production uses 4.
    pub rle_min_average_run: f64,
    /// RunEnd is skipped below this average run length. Production uses 4.
    pub runend_min_average_run: f64,
}

impl Default for RunModel {
    fn default() -> Self {
        Self::size_only()
    }
}

impl RunModel {
    /// Ranks by size alone, whatever the decode cost.
    pub const fn size_only() -> Self {
        Self {
            bandwidth: None,
            baseline_ns_per_value: 0.35,
            runend_ns_per_run: 4.5,
            rle_ns_per_value: 0.6,
            sparse_ns_per_value: 0.35,
            sparse_ns_per_patch: 2.0,
            rle_min_average_run: 2.0,
            runend_min_average_run: 1.0,
        }
    }

    /// Ranks by `bytes / bandwidth + decode time`, with `bandwidth` in bytes per second.
    pub const fn for_bandwidth(bandwidth: f64) -> Self {
        let mut model = Self::size_only();
        model.bandwidth = Some(bandwidth);
        model
    }

    /// Converts decode time beyond bit-packing's into the bits that time could have transferred.
    fn penalty_bits(&self, extra_ns: f64) -> f64 {
        match self.bandwidth {
            Some(bandwidth) => extra_ns * 1e-9 * bandwidth * 8.0,
            None => 0.0,
        }
    }

    fn verdict(&self, canonical_bits: f64, model_bits: f64, extra_ns: f64) -> CompressionEstimate {
        let cost = (model_bits + self.penalty_bits(extra_ns)).max(1.0);
        let ratio = canonical_bits / cost;
        if ratio > 1.0 {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(ratio))
        } else {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }
    }
}

/// Bits needed for values `0..=max`.
fn bits_for_max(max: u64) -> f64 {
    f64::from(u64::BITS - max.leading_zeros()).max(1.0)
}

/// Bits needed to tell `count` values apart.
fn bits_for_count(count: f64) -> f64 {
    if count <= 1.0 {
        1.0
    } else {
        count.log2().ceil()
    }
}

/// Which scheme a [`Modeled`] wraps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    RunEnd,
    Rle,
    Sparse,
}

/// The stats every model estimate reads.
struct Inputs {
    len: f64,
    canonical_bits: f64,
    value_count: f64,
    null_count: u32,
    value_bits: f64,
    position_bits: f64,
    runs: f64,
}

impl Inputs {
    fn from(data: &ArrayAndStats, exec_ctx: &mut ExecutionCtx) -> Option<Self> {
        let width = data.array_as_primitive().ptype().bit_width() as f64;
        let len = data.array_len() as f64;
        let stats = data.integer_stats(exec_ctx);
        if stats.value_count() == 0 {
            return None;
        }
        Some(Self {
            len,
            canonical_bits: len * width,
            value_count: f64::from(stats.value_count()),
            null_count: stats.null_count(),
            value_bits: bits_for_max(stats.erased().max_minus_min()),
            position_bits: bits_for_max(data.array_len() as u64),
            runs: f64::from(stats.run_count()),
        })
    }
}

/// A RunEnd, RLE or Sparse scheme whose estimate comes from a [`RunModel`].
///
/// It shares the wrapped scheme's [`SchemeId`](crate::SchemeId), so exclusion rules that name the
/// wrapped scheme apply to it unchanged, and it compresses exactly as the wrapped scheme does.
#[derive(Debug)]
pub(crate) struct Modeled {
    inner: &'static dyn Scheme,
    kind: Kind,
    model: RunModel,
}

impl Modeled {
    fn runend(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let Some(i) = Inputs::from(data, exec_ctx) else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        // Null runs are not counted in the stats, so leave nullable data to the wrapped scheme.
        if i.null_count > 0 {
            return self
                .inner
                .expected_compression_ratio(data, compress_ctx, exec_ctx);
        }
        if i.value_count < self.model.runend_min_average_run * i.runs {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let bits = i.runs * (i.value_bits + i.position_bits);
        let extra_ns =
            i.runs * self.model.runend_ns_per_run - i.len * self.model.baseline_ns_per_value;
        self.model.verdict(i.canonical_bits, bits, extra_ns)
    }

    fn rle(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // RLE is only useful when it cascades into its children.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let Some(i) = Inputs::from(data, exec_ctx) else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        if i.null_count > 0 {
            return self
                .inner
                .expected_compression_ratio(data, compress_ctx, exec_ctx);
        }
        if i.value_count < self.model.rle_min_average_run * i.runs {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let chunks = (i.len / RLE_CHUNK_LEN).ceil();
        // Runs vary between chunks; the indices pack at the widest chunk's width.
        let runs_per_chunk = i.runs / chunks * 1.25;
        let bits = (i.runs + chunks) * i.value_bits
            + i.len * bits_for_count(runs_per_chunk)
            + chunks * 16.0;
        let extra_ns = i.len * (self.model.rle_ns_per_value - self.model.baseline_ns_per_value);
        self.model.verdict(i.canonical_bits, bits, extra_ns)
    }

    fn sparse(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let Some(i) = Inputs::from(data, exec_ctx) else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        // Null-dominated arrays keep the wrapped scheme's estimate.
        if f64::from(i.null_count) / i.len > 0.9 {
            return self
                .inner
                .expected_compression_ratio(data, compress_ctx, exec_ctx);
        }
        let Some((_, top)) = data.integer_stats(exec_ctx).most_frequent_value_and_count() else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        let top = f64::from(top);
        // A single value compresses as Constant instead.
        if top >= i.value_count {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        // The fill value is not null, so every null is stored as a patch too.
        let patches = i.len - top;
        let bits = patches * (i.value_bits + i.position_bits);
        let extra_ns = i.len * (self.model.sparse_ns_per_value - self.model.baseline_ns_per_value)
            + patches * self.model.sparse_ns_per_patch;
        self.model.verdict(i.canonical_bits, bits, extra_ns)
    }
}

impl Scheme for Modeled {
    fn scheme_name(&self) -> &'static str {
        self.inner.scheme_name()
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        self.inner.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        self.inner.produced_encodings()
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        self.inner.stats_options()
    }

    fn num_children(&self) -> usize {
        self.inner.num_children()
    }

    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        self.inner.descendant_exclusions()
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        self.inner.ancestor_exclusions()
    }

    fn fixed_nbytes(&self, compressed: &ArrayRef) -> u64 {
        self.inner.fixed_nbytes(compressed)
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        match self.kind {
            Kind::RunEnd => self.runend(data, compress_ctx, exec_ctx),
            Kind::Rle => self.rle(data, compress_ctx, exec_ctx),
            Kind::Sparse => self.sparse(data, compress_ctx, exec_ctx),
        }
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.inner
            .compress(compressor, data, compress_ctx, exec_ctx)
    }
}

/// The modeled RunEnd, RLE and Sparse schemes, in that order.
type ModeledSchemes = [&'static dyn Scheme; 3];

/// Modeled schemes built so far, one set per distinct model.
///
/// Schemes are registered as `&'static`, so each model's schemes are leaked once and reused by
/// every later builder with an equal model.
static MODELED: LazyLock<Mutex<Vec<(RunModel, ModeledSchemes)>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// Returns the modeled RunEnd, RLE and Sparse schemes for `model`.
pub(crate) fn modeled_schemes(model: RunModel) -> ModeledSchemes {
    let mut cache = MODELED.lock();
    if let Some((_, schemes)) = cache.iter().find(|(m, _)| *m == model) {
        return *schemes;
    }
    let leak = |inner: &'static dyn Scheme, kind| -> &'static dyn Scheme {
        Box::leak(Box::new(Modeled { inner, kind, model }))
    };
    let schemes = [
        leak(&RunEndScheme, Kind::RunEnd),
        leak(&IntRLEScheme, Kind::Rle),
        leak(&SparseScheme, Kind::Sparse),
    ];
    cache.push((model, schemes));
    schemes
}
