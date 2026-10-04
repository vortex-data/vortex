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
use vortex_array::ArrayRef;
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
}

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
            Mode::NoSample | Mode::CapOff => inner.scheme_name(),
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
