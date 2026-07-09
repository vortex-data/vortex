// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression ratio estimation types and sampling-based estimation.

use std::fmt;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::candidate::Candidate;
use crate::cost::Cost;
use crate::cost::CostModel;
use crate::ctx::CompressorContext;
use crate::sample::SAMPLE_SIZE;
use crate::sample::sample;
use crate::sample::sample_count_approx_one_percent;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;
use crate::trace;

/// Closure type for [`DeferredEstimate::Callback`].
///
/// The compressor calls this with the same arguments it would pass to sampling, plus a
/// [`SkipThreshold`] handle wrapping the best candidate observed so far. The closure must
/// resolve directly to a terminal [`EstimateVerdict`].
///
/// The threshold is an early-exit hint. If your scheme knows its maximum achievable
/// compression ratio, ask [`SkipThreshold::best_case_ratio_cannot_win`] before doing
/// expensive work and return [`EstimateVerdict::Skip`] when it answers `true`. Returning an
/// estimate that merely ties the threshold is permitted but will lose to the prior best,
/// since displacing the running best requires a strictly better (lower-cost) candidate. Use
/// the threshold only as an early-exit hint, never to perform additional work.
#[rustfmt::skip]
pub type EstimateFn = dyn FnOnce(
        &CascadingCompressor,
        &ArrayAndStats,
        SkipThreshold,
        CompressorContext,
        &mut ExecutionCtx,
    ) -> VortexResult<EstimateVerdict>
    + Send
    + Sync;

/// Early-exit threshold handle passed to [`DeferredEstimate::Callback`] closures.
///
/// Owned and constructed by the compressor during selection; wraps the best candidate
/// observed so far together with the compressor's [`CostModel`] and the selection-site
/// facts needed to price a hypothetical best-case candidate. Schemes keep their side of the
/// bargain — knowing their own best case — and ask the handle whether that best case could
/// still win.
pub struct SkipThreshold {
    /// The best candidate so far: its cost and its ranked estimate. `None` when no
    /// candidate has been stored yet.
    best: Option<(Cost, EstimateScore)>,

    /// The compressor's cost model.
    model: Arc<dyn CostModel>,

    /// The scheme whose deferred estimate is being resolved.
    scheme: &'static dyn Scheme,

    /// Uncompressed size in bytes of the array under selection.
    input_nbytes: u64,

    /// Number of values in the array under selection.
    n_values: u64,

    /// The cascade ancestry `(scheme_id, child_index)` of the selection site.
    cascade: Vec<(SchemeId, usize)>,
}

impl SkipThreshold {
    /// Creates a threshold handle for `scheme` at a selection site.
    ///
    /// Normally the compressor builds these during selection. The constructor is public so
    /// scheme implementors can unit-test their callbacks' skip decisions.
    pub fn new(
        best: Option<(Cost, EstimateScore)>,
        model: Arc<dyn CostModel>,
        scheme: &'static dyn Scheme,
        input_nbytes: u64,
        n_values: u64,
        cascade: Vec<(SchemeId, usize)>,
    ) -> Self {
        Self {
            best,
            model,
            scheme,
            input_nbytes,
            n_values,
            cascade,
        }
    }

    /// Returns `true` if a candidate achieving `max_ratio` — the best case the calling
    /// scheme could possibly report — still could not displace the best candidate observed
    /// so far, so the scheme should return [`EstimateVerdict::Skip`] without doing expensive
    /// work.
    ///
    /// The handle prices a hypothetical candidate carrying `max_ratio` under the
    /// compressor's cost model and compares it against the best cost so far; displacement
    /// requires a strictly lower cost. Under the default [`SizeCost`] model this reduces to
    /// exactly the historical `max_ratio <= best_ratio` skip. Returns `false` when there is
    /// no best candidate yet or the best case cannot be priced.
    ///
    /// [`SizeCost`]: crate::cost::SizeCost
    pub fn best_case_ratio_cannot_win(&self, max_ratio: f64) -> bool {
        let Some((best_cost, _)) = self.best else {
            return false;
        };
        let best_case = Candidate {
            scheme: self.scheme,
            score: EstimateScore::FiniteCompression(max_ratio),
            input_nbytes: self.input_nbytes,
            n_values: self.n_values,
            sampled: None,
            cascade: self.cascade.clone(),
        };
        self.model
            .cost(&best_case)
            .is_some_and(|cost| cost >= best_cost)
    }

    /// The best finite compression ratio observed so far, if any — the pre-cost-model view
    /// of the threshold, kept for callbacks that reason in ratio space.
    pub fn best_ratio(&self) -> Option<f64> {
        self.best.and_then(|(_, score)| score.finite_ratio())
    }
}

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
    /// invoking any deferred callback, and passes a [`SkipThreshold`] wrapping the best
    /// candidate observed so far. This lets the callback return [`EstimateVerdict::Skip`]
    /// without performing expensive work when its maximum achievable estimate cannot win. See
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
}

/// Winner estimate carried from scheme selection into result tracing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum WinnerEstimate {
    /// The scheme must be used immediately.
    AlwaysUse,
    /// The scheme won by a ranked estimate, priced by the compressor's cost model.
    Score {
        /// The winning candidate's ranked estimate.
        score: EstimateScore,
        /// The winning candidate's cost under the compressor's cost model.
        cost: Cost,
    },
}

impl WinnerEstimate {
    /// Returns the traceable numeric ratio for the winning estimate.
    pub(super) fn trace_ratio(self) -> Option<f64> {
        match self {
            Self::AlwaysUse => None,
            Self::Score { score, .. } => score.finite_ratio(),
        }
    }

    /// Returns the traceable cost for the winning estimate.
    pub(super) fn trace_cost(self) -> Option<f64> {
        match self {
            Self::AlwaysUse => None,
            Self::Score { cost, .. } => Some(cost.value()),
        }
    }
}

/// A sampling-based estimate: the ranked score together with the compressed sample array it
/// was measured on.
pub(crate) struct SampledEstimate {
    /// The ranked estimate measured on the sample.
    pub(crate) score: EstimateScore,

    /// The compressed sample array. Its encoding tree is the best available prediction of
    /// the full-array encoding tree.
    pub(crate) sampled: ArrayRef,
}

/// Estimates compression ratio by compressing a ~1% sample of the data.
///
/// Creates a new [`ArrayAndStats`] for the sample so that stats are generated from the sample, not
/// the full array.
///
/// Returns the compressed sample alongside its score so the selection loop can keep the
/// sample's encoding tree for the winning candidate.
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
) -> VortexResult<SampledEstimate> {
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

    Ok(SampledEstimate {
        score,
        sampled: compressed,
    })
}

impl fmt::Debug for DeferredEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeferredEstimate::Sample => write!(f, "Sample"),
            DeferredEstimate::Callback(_) => write!(f, "Callback(..)"),
        }
    }
}
