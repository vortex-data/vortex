// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression ratio estimation types and sampling-based estimation.

use std::fmt;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::ctx::CompressorContext;
use crate::sample::SAMPLE_SIZE;
use crate::sample::sample;
use crate::sample::sample_count_approx_one_percent;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::trace;

/// Closure type for [`DeferredEstimate::Callback`].
///
/// The compressor calls this with the same arguments it would pass to sampling, plus the best
/// [`EstimateScore`] observed so far (if any). The closure must resolve directly to a terminal
/// [`EstimateVerdict`].
///
/// The `best_so_far` threshold is an early-exit hint. If your scheme's maximum achievable
/// compression ratio is not strictly greater than
/// `best_so_far.and_then(EstimateScore::finite_ratio)`, you should return
/// [`EstimateVerdict::Skip`]. Returning a ratio equal to the threshold is permitted but will
/// lose to the prior best due to strict `>` tie-breaking in the selector. Use the threshold
/// only as an early-exit hint, never to perform additional work.
#[rustfmt::skip]
pub type EstimateFn = dyn FnOnce(
        &CascadingCompressor,
        &ArrayAndStats,
        Option<EstimateScore>,
        CompressorContext,
        &mut ExecutionCtx,
    ) -> VortexResult<EstimateVerdict>
    + Send
    + Sync;

/// The result of a [`Scheme`]'s compression ratio estimation.
///
/// This type is returned by [`Scheme::expected_compression_ratio`] to tell the compressor how
/// promising this scheme is for a given array without performing any expensive work.
///
/// [`CompressionEstimate::Verdict`] means the scheme already knows the terminal answer.
/// [`CompressionEstimate::Deferred`] means the compressor must do extra work before the scheme can
/// produce a terminal answer.
#[derive(Debug)]
pub enum CompressionEstimate {
    /// The scheme already knows the terminal estimation verdict.
    Verdict(EstimateVerdict),

    /// The compressor must perform deferred work to resolve the terminal estimation verdict.
    Deferred(DeferredEstimate),
}

/// The terminal answer to a compression estimate request.
#[derive(Debug)]
pub enum EstimateVerdict {
    /// Do not use this scheme for this array.
    Skip,

    /// Always use this scheme, as it is definitively the best choice.
    ///
    /// Some examples include constant detection, decimal byte parts, and temporal decomposition.
    ///
    /// The compressor will select this scheme immediately without evaluating further candidates.
    /// Schemes that return `AlwaysUse` must be mutually exclusive per canonical type (enforced by
    /// [`Scheme::matches`]), otherwise the winner depends silently on registration order.
    ///
    /// [`Scheme::matches`]: crate::scheme::Scheme::matches
    AlwaysUse,

    /// The estimated compression ratio. This must be greater than `1.0` to be considered by the
    /// compressor, otherwise it is worse than the canonical encoding.
    Ratio(f64),
}

/// Deferred work that can resolve to a terminal [`EstimateVerdict`].
pub enum DeferredEstimate {
    /// The scheme cannot cheaply estimate its ratio, so the compressor should compress a small
    /// sample to determine effectiveness.
    Sample,

    /// A fallible estimation requiring a custom expensive computation.
    ///
    /// Use this only when the scheme needs to perform trial encoding or other costly checks to
    /// determine its compression ratio. The callback returns an [`EstimateVerdict`] directly, so
    /// it cannot request more sampling or another deferred callback.
    ///
    /// The compressor evaluates all immediate [`CompressionEstimate::Verdict`] results before
    /// invoking any deferred callback, and passes the best [`EstimateScore`] observed so far to
    /// the callback. This lets the callback return [`EstimateVerdict::Skip`] without performing
    /// expensive work when its maximum achievable ratio cannot beat the current best. See
    /// [`EstimateFn`] for the full contract.
    Callback(Box<EstimateFn>),
}

/// Ranked estimate used for comparing non-terminal compression candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EstimateScore {
    /// A finite compression ratio. Higher means a smaller amount of data, so it is better.
    FiniteCompression(f64),
    /// Trial compression produced a 0-byte output.
    ///
    /// This has no finite ratio and is not eligible for scheme selection.
    ///
    /// TODO(connor): A zero-byte sample usually means the sampler happened to hit an all-null
    /// sample. Improve this logic so we can distinguish real zero-byte wins from sampling artifacts.
    ZeroBytes,
}

impl EstimateScore {
    /// Converts measured sample sizes into a ranked estimate.
    pub(super) fn from_sample_sizes(before_nbytes: u64, after_nbytes: u64) -> Self {
        if after_nbytes == 0 {
            Self::ZeroBytes
        } else {
            Self::FiniteCompression(before_nbytes as f64 / after_nbytes as f64)
        }
    }

    /// Returns the finite compression ratio, or [`None`] for the zero-byte special case.
    ///
    /// Callers comparing a scheme's maximum achievable ratio against a "best so far" threshold
    /// should use this to extract a numeric value from an [`EstimateScore`].
    pub fn finite_ratio(self) -> Option<f64> {
        match self {
            Self::FiniteCompression(ratio) => Some(ratio),
            Self::ZeroBytes => None,
        }
    }

    /// Returns whether this estimate is eligible to compete.
    fn is_valid(self) -> bool {
        match self {
            Self::FiniteCompression(ratio) => {
                ratio.is_finite() && !ratio.is_subnormal() && ratio > 1.0
            }
            Self::ZeroBytes => false,
        }
    }

    /// Returns whether this estimate beats another valid estimate.
    fn beats(self, other: Self) -> bool {
        match (self, other) {
            (Self::ZeroBytes, _) => false,
            (Self::FiniteCompression(_), Self::ZeroBytes) => true,
            (Self::FiniteCompression(ratio), Self::FiniteCompression(best_ratio)) => {
                ratio > best_ratio
            }
        }
    }
}

/// Winner estimate carried from scheme selection into result tracing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum WinnerEstimate {
    /// The scheme must be used immediately.
    AlwaysUse,
    /// The scheme won by a ranked estimate.
    Score(EstimateScore),
}

impl WinnerEstimate {
    /// Returns the traceable numeric ratio for the winning estimate.
    pub(super) fn trace_ratio(self) -> Option<f64> {
        match self {
            Self::AlwaysUse => None,
            Self::Score(score) => score.finite_ratio(),
        }
    }
}

/// Returns `true` if `score` beats the current best estimate.
pub(super) fn is_better_score(
    score: EstimateScore,
    best: Option<&(&'static dyn Scheme, EstimateScore)>,
) -> bool {
    score.is_valid() && best.is_none_or(|(_, best_score)| score.beats(*best_score))
}

/// Estimates compression ratio by compressing a ~1% sample of the data.
///
/// Creates a new [`ArrayAndStats`] for the sample so that stats are generated from the sample, not
/// the full array.
///
/// # Errors
///
/// Returns an error if sample compression fails.
pub(super) fn estimate_compression_ratio_with_sampling<S: Scheme + ?Sized>(
    compressor: &CascadingCompressor,
    scheme: &S,
    array: &ArrayRef,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<EstimateScore> {
    let sample_array = if compress_ctx.is_sample() {
        array.clone()
    } else {
        let sample_count = sample_count_approx_one_percent(array.len());
        // `ArrayAndStats` expects a canonical array (so that it can easily compute lazy stats).
        let canonical: Canonical = sample(array, SAMPLE_SIZE, sample_count).execute(exec_ctx)?;
        canonical.into_array()
    };

    let sample_data = ArrayAndStats::new(sample_array, scheme.stats_options());
    let error_ctx = trace::enabled_error_context(&compress_ctx);
    let sample_ctx = compress_ctx.with_sampling();

    let compressed = match scheme.compress(compressor, &sample_data, sample_ctx, exec_ctx) {
        Ok(compressed) => compressed,
        Err(err) => {
            trace::sample_compress_failed(scheme.id(), error_ctx.as_ref(), &err);
            return Err(err);
        }
    };

    let after = compressed.nbytes();
    let before = sample_data.array().nbytes();

    let score = EstimateScore::from_sample_sizes(before, after);

    if matches!(score, EstimateScore::ZeroBytes) {
        trace::zero_byte_sample_result(scheme.id(), before);
    }

    Ok(score)
}

impl fmt::Debug for DeferredEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeferredEstimate::Sample => write!(f, "Sample"),
            DeferredEstimate::Callback(_) => write!(f, "Callback(..)"),
        }
    }
}
