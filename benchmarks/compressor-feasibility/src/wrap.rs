// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheme wrappers that change how the stock compressor chooses, without changing its code.
//!
//! A wrapper delegates everything to the scheme it wraps except `expected_compression_ratio`.
//! Replacement wrappers keep the wrapped scheme's id, so exclusion rules and registration order
//! behave exactly as in production. The forcing wrapper has its own id and only acts at the root.

use std::fmt;

use vortex_array::ArrayId;
use vortex_array::Canonical;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::serde::SerializeOptions;
use vortex_array::ExecutionCtx;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::CompressorContext;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::stats::ArrayAndStats;
use vortex_compressor::stats::GenerateStatsOptions;
use vortex_error::VortexResult;

/// How a wrapper changes the wrapped scheme's estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Always use the scheme at the root; never below it.
    ForceRoot,
    /// Replace "sample me" with a closed-form estimate from the stats.
    NoSample,
    /// Replace heuristic skips with sampling, so the sample decides.
    CapOff,
    /// Replace "sample me" (and optionally closed-form ratios) with a custom sample estimate.
    Estimate(SamplePolicy),
    /// Estimate RunEnd and Sparse from a size model over stats instead of caps and samples.
    SizeModel,
    /// Record the production estimate at the root without changing it.
    Spy,
}

/// Root estimates recorded by [`Mode::Spy`]: (scheme, kind, ratio).
pub static ESTIMATES: parking_lot::Mutex<Vec<(String, &'static str, f64)>> =
    parking_lot::Mutex::new(Vec::new());

fn spy_record(name: &str, kind: &'static str, ratio: f64) {
    ESTIMATES.lock().push((name.to_string(), kind, ratio));
}

/// How a custom sample estimate is taken and measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplePolicy {
    /// Number of slices, spread evenly over the array.
    pub slices: usize,
    /// Values per slice.
    pub slice_len: usize,
    /// Measure sizes as serialized bytes (buffers plus metadata) instead of buffer bytes.
    pub serialized: bool,
    /// Let a sample that compresses to zero buffer bytes compete instead of disqualifying it.
    pub zero_ok: bool,
    /// Also replace closed-form ratio verdicts (Sparse, Dict, FOR) with the sample estimate.
    pub all: bool,
}

/// The session used to measure serialized sizes inside estimates.
pub static SESSION: std::sync::OnceLock<vortex::session::VortexSession> = std::sync::OnceLock::new();

/// A scheme with a changed estimate.
pub struct Wrapped {
    inner: &'static dyn Scheme,
    name: &'static str,
    mode: Mode,
}

impl fmt::Debug for Wrapped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Wrapped")
            .field("name", &self.name)
            .field("mode", &self.mode)
            .finish()
    }
}

impl Wrapped {
    /// Leaks a wrapper so it can be registered like the built-in statics.
    pub fn leak(inner: &'static dyn Scheme, mode: Mode) -> &'static dyn Scheme {
        let name = match mode {
            Mode::ForceRoot => {
                let name: &'static str =
                    Box::leak(format!("forced/{}", inner.scheme_name()).into_boxed_str());
                name
            }
            Mode::NoSample | Mode::CapOff | Mode::Estimate(_) | Mode::SizeModel | Mode::Spy => {
                inner.scheme_name()
            }
        };
        Box::leak(Box::new(Self { inner, name, mode }))
    }

    fn base_name(&self) -> &'static str {
        self.inner.scheme_name()
    }
}

impl Scheme for Wrapped {
    fn scheme_name(&self) -> &'static str {
        self.name
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        self.inner.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        self.inner.produced_encodings()
    }

    fn refine(&self, allowed: &dyn Fn(&ArrayId) -> bool) -> &dyn Scheme {
        let refined = self.inner.refine(allowed);
        if std::ptr::addr_eq(refined, self.inner) {
            self
        } else {
            Self::leak(refined, self.mode)
        }
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

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        match self.mode {
            Mode::ForceRoot => {
                if compress_ctx.cascade_history().is_empty() && !compress_ctx.is_sample() {
                    CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
                } else {
                    CompressionEstimate::Verdict(EstimateVerdict::Skip)
                }
            }
            Mode::NoSample => {
                let finished = compress_ctx.finished_cascading();
                match self
                    .inner
                    .expected_compression_ratio(data, compress_ctx, exec_ctx)
                {
                    CompressionEstimate::Deferred(DeferredEstimate::Sample) => {
                        CompressionEstimate::Verdict(closed_form(
                            self.base_name(),
                            data,
                            finished,
                            exec_ctx,
                        ))
                    }
                    other => other,
                }
            }
            Mode::Estimate(policy) => {
                let inner = self.inner;
                match inner.expected_compression_ratio(data, compress_ctx, exec_ctx) {
                    CompressionEstimate::Deferred(DeferredEstimate::Sample) => {
                        estimate_with(inner, policy)
                    }
                    CompressionEstimate::Verdict(EstimateVerdict::Ratio(_)) if policy.all => {
                        estimate_with(inner, policy)
                    }
                    other => other,
                }
            }
            Mode::Spy => {
                let root = compress_ctx.cascade_history().is_empty() && !compress_ctx.is_sample();
                let name = self.base_name();
                let estimate = self.inner.expected_compression_ratio(
                    data,
                    compress_ctx.clone(),
                    exec_ctx,
                );
                if !root {
                    return estimate;
                }
                match estimate {
                    CompressionEstimate::Verdict(EstimateVerdict::Ratio(r)) => {
                        spy_record(name, "closed_form", r);
                        estimate
                    }
                    CompressionEstimate::Verdict(EstimateVerdict::Skip) => {
                        spy_record(name, "skip", f64::NAN);
                        estimate
                    }
                    CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse) => {
                        spy_record(name, "always", f64::INFINITY);
                        estimate
                    }
                    CompressionEstimate::Deferred(DeferredEstimate::Sample) => {
                        // Reproduce the production sample (16 x 64, buffer bytes) to record its ratio.
                        let policy = SamplePolicy {
                            slices: 16,
                            slice_len: 64,
                            serialized: false,
                            zero_ok: false,
                            all: false,
                        };
                        let ratio = sample_ratio(self.inner, policy, data, compress_ctx, exec_ctx);
                        spy_record(name, "sample", ratio.unwrap_or(f64::NAN));
                        CompressionEstimate::Deferred(DeferredEstimate::Sample)
                    }
                    CompressionEstimate::Deferred(DeferredEstimate::Callback(callback)) => {
                        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
                            move |compressor, data, best, ctx, exec| {
                                let verdict = callback(compressor, data, best, ctx, exec)?;
                                match verdict {
                                    EstimateVerdict::Ratio(r) => spy_record(name, "callback", r),
                                    EstimateVerdict::Skip => spy_record(name, "skip", f64::NAN),
                                    EstimateVerdict::AlwaysUse => {
                                        spy_record(name, "always", f64::INFINITY)
                                    }
                                }
                                Ok(verdict)
                            },
                        )))
                    }
                }
            }
            Mode::SizeModel => {
                let name = self.base_name();
                if name.contains("runend") || name.contains("sparse") {
                    size_model(name, data, exec_ctx)
                } else {
                    self.inner
                        .expected_compression_ratio(data, compress_ctx, exec_ctx)
                }
            }
            Mode::CapOff => {
                let finished = compress_ctx.finished_cascading();
                let estimate = self
                    .inner
                    .expected_compression_ratio(data, compress_ctx, exec_ctx);
                match estimate {
                    CompressionEstimate::Verdict(EstimateVerdict::Skip)
                        if cap_applies(self.base_name(), data, finished, exec_ctx) =>
                    {
                        CompressionEstimate::Deferred(DeferredEstimate::Sample)
                    }
                    other => other,
                }
            }
        }
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.inner.compress(compressor, data, compress_ctx, exec_ctx)
    }
}

fn bits(value: u64) -> f64 {
    f64::from(u64::BITS - value.leading_zeros())
}

/// A closed-form ratio for the schemes that only know how to sample.
fn closed_form(
    name: &str,
    data: &ArrayAndStats,
    finished_cascading: bool,
    exec_ctx: &mut ExecutionCtx,
) -> EstimateVerdict {
    let width = f64::from(data.array_as_primitive().ptype().bit_width() as u32);
    let len = data.array_len() as u64;
    let stats = data.integer_stats(exec_ctx);
    let erased = stats.erased();
    let ratio = if name.contains("bitpack") {
        erased
            .max_ilog2()
            .map(|log| width / f64::from(log + 1))
            .unwrap_or(width)
    } else if name.contains("zigzag") {
        if finished_cascading {
            return EstimateVerdict::Skip;
        }
        width / (bits(erased.max_minus_min()) + 1.0).max(1.0)
    } else if name.contains("run_end") || name.contains("runend") {
        let run = f64::from(stats.average_run_length().max(1));
        run * width / (width + bits(len))
    } else if name.contains("rle") {
        if finished_cascading {
            return EstimateVerdict::Skip;
        }
        let run = f64::from(stats.average_run_length().max(1));
        run * width / (width + 16.0)
    } else {
        return EstimateVerdict::Skip;
    };
    if ratio > 1.0 {
        EstimateVerdict::Ratio(ratio)
    } else {
        EstimateVerdict::Skip
    }
}

/// Whether a skip from this scheme came from a tunable cap rather than a hard limit.
fn cap_applies(
    name: &str,
    data: &ArrayAndStats,
    finished_cascading: bool,
    exec_ctx: &mut ExecutionCtx,
) -> bool {
    let stats = data.integer_stats(exec_ctx);
    if stats.value_count() == 0 {
        return false;
    }
    if name.contains("dict") {
        true
    } else if name.contains("sparse") {
        stats
            .most_frequent_value_and_count()
            .is_some_and(|(_, top)| top < stats.value_count())
    } else if name.contains("rle") {
        !finished_cascading
    } else {
        name.contains("run_end") || name.contains("runend")
    }
}

fn estimate_with(inner: &'static dyn Scheme, policy: SamplePolicy) -> CompressionEstimate {
    CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
        move |compressor, data, _best, compress_ctx, exec_ctx| {
            let array = data.array();
            let sample = if compress_ctx.is_sample() {
                array.clone()
            } else {
                take_sample(array, policy, exec_ctx)?
            };
            let sample_data = ArrayAndStats::new(sample, inner.stats_options());
            let Ok(compressed) =
                inner.compress(compressor, &sample_data, compress_ctx.with_sampling(), exec_ctx)
            else {
                return Ok(EstimateVerdict::Skip);
            };
            let (before, after) = if policy.serialized {
                (size_of(sample_data.array()), size_of(&compressed))
            } else {
                (sample_data.array().nbytes(), compressed.nbytes())
            };
            Ok(match after {
                0 if policy.zero_ok => EstimateVerdict::Ratio(before as f64),
                0 => EstimateVerdict::Skip,
                _ => EstimateVerdict::Ratio(before as f64 / after as f64),
            })
        },
    )))
}

/// The ratio a sample estimate would report, or `None` if the sample fails to compress.
fn sample_ratio(
    inner: &'static dyn Scheme,
    policy: SamplePolicy,
    data: &ArrayAndStats,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> Option<f64> {
    let compressor = SPY_COMPRESSOR.get()?;
    let sample = take_sample(data.array(), policy, exec_ctx).ok()?;
    let sample_data = ArrayAndStats::new(sample, inner.stats_options());
    let compressed = inner
        .compress(compressor, &sample_data, compress_ctx.with_sampling(), exec_ctx)
        .ok()?;
    let after = compressed.nbytes();
    (after > 0).then(|| sample_data.array().nbytes() as f64 / after as f64)
}

/// The production compressor the spy compresses its samples with.
pub static SPY_COMPRESSOR: std::sync::OnceLock<CascadingCompressor> = std::sync::OnceLock::new();

fn take_sample(
    array: &ArrayRef,
    policy: SamplePolicy,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = array.len();
    let total = policy.slices * policy.slice_len;
    if len <= total {
        return Ok(array.clone());
    }
    let partition = len / policy.slices;
    let chunks = (0..policy.slices)
        .map(|i| {
            // The middle of each partition, so slices are spread evenly and deterministic.
            let start = i * partition + (partition.saturating_sub(policy.slice_len)) / 2;
            array.slice(start..start + policy.slice_len)
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let chunked = ChunkedArray::try_new(chunks, array.dtype().clone())?.into_array();
    Ok(chunked.execute::<Canonical>(exec_ctx)?.into_array())
}

fn size_of(array: &ArrayRef) -> u64 {
    let Some(session) = SESSION.get() else {
        return array.nbytes();
    };
    array
        .serialize(&ArrayContext::empty(), session, &SerializeOptions::default())
        .map(|buffers| buffers.iter().map(|b| b.len() as u64).sum())
        .unwrap_or_else(|_| array.nbytes())
}

/// Estimates RunEnd and Sparse from the bits their outputs need, rather than from caps.
///
/// RunEnd stores one value and one end per run: values bit-packed at the value range's width,
/// ends frame-of-reference packed at roughly the width of the array length. Sparse stores the
/// non-top values and their positions. Both are compared against the canonical width.
fn size_model(name: &str, data: &ArrayAndStats, exec_ctx: &mut ExecutionCtx) -> CompressionEstimate {
    let width = f64::from(data.array_as_primitive().ptype().bit_width() as u32);
    let len = data.array_len() as f64;
    let stats = data.integer_stats(exec_ctx);
    let values = f64::from(stats.value_count());
    if values == 0.0 {
        return CompressionEstimate::Verdict(EstimateVerdict::Skip);
    }
    let value_bits = bits(stats.erased().max_minus_min()).max(1.0);
    let position_bits = bits(data.array_len() as u64);
    let canonical_bits = len * width;
    let model_bits = if name.contains("runend") {
        // `average_run_length` is truncated to an integer, so count runs exactly instead.
        let primitive = data.array_as_primitive();
        let runs = vortex_array::match_each_integer_ptype!(primitive.ptype(), |T| {
            let slice = primitive.as_slice::<T>();
            1 + slice.windows(2).filter(|pair| pair[0] != pair[1]).count()
        }) as f64;
        runs * (value_bits + position_bits)
    } else {
        let Some((_, top)) = stats.most_frequent_value_and_count() else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        let top = f64::from(top);
        if top >= values {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        // Sparse fills with the most common value, or with null when most values are null.
        let nulls = len - values;
        let patches = if nulls > top { values } else { values - top };
        patches * (value_bits + position_bits)
    };
    let ratio = canonical_bits / model_bits.max(1.0);
    if ratio > 1.0 {
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(ratio))
    } else {
        CompressionEstimate::Verdict(EstimateVerdict::Skip)
    }
}
