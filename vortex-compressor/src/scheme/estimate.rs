// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression ratio estimation types returned by schemes.

use std::fmt;

use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::scheme::CompressorContext;
use crate::stats::ArrayAndStats;

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

/// The result of a [`Scheme`](crate::scheme::Scheme)'s compression ratio estimation.
///
/// This type is returned by
/// [`Scheme::expected_compression_ratio`](crate::scheme::Scheme::expected_compression_ratio) to
/// tell the compressor how promising this scheme is for a given array without performing any
/// expensive work.
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
    /// Some examples include decimal byte parts and temporal decomposition.
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
    pub(crate) fn from_sample_sizes(before_nbytes: u64, after_nbytes: u64) -> Self {
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
    pub(crate) fn is_valid(self) -> bool {
        match self {
            Self::FiniteCompression(ratio) => {
                ratio.is_finite() && !ratio.is_subnormal() && ratio > 1.0
            }
            Self::ZeroBytes => false,
        }
    }

    /// Returns whether this estimate beats another valid estimate.
    pub(crate) fn beats(self, other: Self) -> bool {
        match (self, other) {
            (Self::ZeroBytes, _) => false,
            (Self::FiniteCompression(_), Self::ZeroBytes) => true,
            (Self::FiniteCompression(ratio), Self::FiniteCompression(best_ratio)) => {
                ratio > best_ratio
            }
        }
    }
}

impl fmt::Debug for DeferredEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeferredEstimate::Sample => write!(f, "Sample"),
            DeferredEstimate::Callback(_) => write!(f, "Callback(..)"),
        }
    }
}
