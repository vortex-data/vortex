// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression ratio estimation types and sampling-based estimation.

use std::fmt;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::ctx::CompressorContext;
use crate::sample::SAMPLE_SIZE;
use crate::sample::sample;
use crate::sample::sample_count_approx_one_percent;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;

/// Closure type for [`CompressionEstimate::Estimate`]. The compressor calls this with the same
/// arguments it would pass to sampling.
#[rustfmt::skip]
pub type EstimateFn = dyn FnOnce(
        &CascadingCompressor,
        &mut ArrayAndStats,
        CompressorContext,
    ) -> VortexResult<CompressionEstimate>
    + Send
    + Sync;

// TODO(connor): We should make use of the fact that some checks are cheap and some checks are
// expensive (sample or estimate variants).
/// The result of a [`Scheme`]'s compression ratio estimation.
///
/// This type is returned by [`Scheme::expected_compression_ratio`] to tell the compressor how
/// promising this scheme is for a given array without performing any expensive work.
///
/// All expensive or fallible operations (sampling, trial encoding) are deferred to the compressor
/// via the [`Sample`](CompressionEstimate::Sample) and [`Estimate`](CompressionEstimate::Estimate)
/// variants.
///
/// [`Sample`]: CompressionEstimate::Sample
/// [`Estimate`]: CompressionEstimate::Estimate
pub enum CompressionEstimate {
    /// Do not use this scheme for this array.
    Skip,

    /// Always use this scheme, as we know it is definitively the best choice.
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

    /// The scheme cannot cheaply estimate its ratio, so the compressor should compress a small
    /// sample to determine effectiveness.
    Sample,

    /// A fallible estimation requiring a custom expensive computation. The compressor will call the
    /// closure and handle the result.
    ///
    /// Use this only when the scheme needs to perform trial encoding or other costly checks to
    /// determine its compression ratio.
    ///
    /// The estimation function must **not** return a [`Sample`](CompressionEstimate::Sample) or
    /// [`Estimate`](CompressionEstimate::Estimate) variant to ensure the estimation process is
    /// bounded.
    Estimate(Box<EstimateFn>),
}

/// Returns `true` if `ratio` is a valid compression ratio (> 1.0, finite, not subnormal) that
/// beats the current best.
pub(super) fn is_better_ratio(ratio: f64, best: &Option<(&'static dyn Scheme, f64)>) -> bool {
    ratio.is_finite() && !ratio.is_subnormal() && ratio > 1.0 && best.is_none_or(|(_, r)| ratio > r)
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
    scheme: &S,
    compressor: &CascadingCompressor,
    array: &ArrayRef,
    ctx: CompressorContext,
) -> VortexResult<f64> {
    let sample_array = if ctx.is_sample() {
        array.clone()
    } else {
        let source_len = array.len();
        let sample_count = sample_count_approx_one_percent(source_len);

        tracing::trace!(
            "Sampling {} values out of {}",
            SAMPLE_SIZE as u64 * sample_count as u64,
            source_len
        );

        // `ArrayAndStats` expects a canonical array (so that it can easily compute lazy stats).
        let canonical: Canonical =
            sample(array, SAMPLE_SIZE, sample_count).execute(&mut compressor.execution_ctx())?;
        canonical.into_array()
    };

    let mut sample_data = ArrayAndStats::new(sample_array, scheme.stats_options());
    let sample_ctx = ctx.with_sampling();

    let after = scheme
        .compress(compressor, &mut sample_data, sample_ctx)?
        .nbytes();
    let before = sample_data.array().nbytes();

    // TODO(connor): Issue https://github.com/vortex-data/vortex/issues/7268.
    // if after == 0 {
    //     tracing::warn!(
    //         scheme = %scheme.id(),
    //         "sample compressed to 0 bytes, which should only happen for constant arrays",
    //     );
    // }

    let ratio = before as f64 / after as f64;

    tracing::debug!("estimate_compression_ratio_with_sampling(compressor={scheme:#?}) = {ratio}",);

    Ok(ratio)
}

impl fmt::Debug for CompressionEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressionEstimate::Skip => write!(f, "Skip"),
            CompressionEstimate::AlwaysUse => write!(f, "AlwaysUse"),
            CompressionEstimate::Ratio(r) => f.debug_tuple("Ratio").field(r).finish(),
            CompressionEstimate::Sample => write!(f, "Sample"),
            CompressionEstimate::Estimate(_) => write!(f, "Estimate(..)"),
        }
    }
}
