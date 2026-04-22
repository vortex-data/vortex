// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Internal tracing helpers for compressor observability.

use std::fmt;

use crate::ctx::CompressorContext;
use crate::scheme::SchemeId;

/// Shared tracing target for compressor decisions and coarse cascade structure.
pub(super) const TARGET_TRACE: &str = "vortex_compressor::encode";

/// Builds the top-level compression span.
#[inline]
pub(super) fn compress_span(
    len: usize,
    dtype: &impl fmt::Display,
    before_nbytes: u64,
) -> tracing::Span {
    tracing::debug_span!(
        target: TARGET_TRACE,
        "compress",
        len,
        dtype = %dtype,
        before_nbytes,
        after_nbytes = tracing::field::Empty,
        ratio = tracing::field::Empty,
    )
}

/// Builds a span covering one deferred per-scheme evaluation (sample or callback).
///
/// Lets timeline tooling (e.g. Perfetto) attribute wall-clock time to individual schemes during
/// the candidate-selection phase, which the instant `sample.result` event cannot express.
#[inline]
pub(super) fn scheme_eval_span(scheme: SchemeId) -> tracing::Span {
    tracing::debug_span!(
        target: TARGET_TRACE,
        "scheme_eval",
        scheme = %scheme,
    )
}

/// Builds a span covering the winning scheme's full-array compression.
///
/// Separates final-encode time from the sampling/selection time captured by `scheme_eval`,
/// so the two phases can be compared per call.
#[inline]
pub(super) fn winner_compress_span(scheme: SchemeId) -> tracing::Span {
    tracing::debug_span!(
        target: TARGET_TRACE,
        "winner_compress",
        scheme = %scheme,
    )
}

/// Records the final output size and, when finite, the top-level compression ratio.
#[inline]
pub(super) fn record_compress_outcome(span: &tracing::Span, before_nbytes: u64, after_nbytes: u64) {
    span.record("after_nbytes", after_nbytes);
    if after_nbytes != 0 {
        span.record("ratio", before_nbytes as f64 / after_nbytes as f64);
    }
}

/// Emits the MAX_CASCADE short-circuit event.
#[inline]
pub(super) fn cascade_exhausted(parent: SchemeId, child_index: usize) {
    tracing::debug!(
        target: TARGET_TRACE,
        parent = %parent,
        child_index,
        "cascade_exhausted",
    );
}

/// Captures the context needed for error tracing only when ERROR logs are enabled.
#[inline]
pub(super) fn enabled_error_context(ctx: &CompressorContext) -> Option<CompressorContext> {
    tracing::enabled!(target: TARGET_TRACE, tracing::Level::ERROR).then(|| ctx.clone())
}

/// Emits a compression-failure event for a winning scheme.
#[inline]
pub(super) fn scheme_compress_failed(
    scheme: SchemeId,
    before_nbytes: u64,
    ctx: Option<&CompressorContext>,
    err: &impl fmt::Display,
) {
    if let Some(ctx) = ctx {
        tracing::error!(
            target: TARGET_TRACE,
            scheme = %scheme,
            before_nbytes,
            cascade_path = %ctx.cascade_path(),
            cascade_depth = ctx.cascade_depth(),
            error = %err,
            "scheme.compress_failed",
        );
    }
}

/// Emits the leaf compression result event.
#[inline]
#[allow(
    clippy::cognitive_complexity,
    reason = "tracing sometimes triggers this"
)]
pub(super) fn scheme_compress_result(
    scheme: SchemeId,
    before_nbytes: u64,
    after_nbytes: u64,
    estimated_ratio: Option<f64>,
    actual_ratio: Option<f64>,
    accepted: bool,
) {
    match (estimated_ratio, actual_ratio) {
        (Some(estimated_ratio), Some(actual_ratio)) => {
            tracing::debug!(
                target: TARGET_TRACE,
                scheme = %scheme,
                before_nbytes,
                after_nbytes,
                estimated_ratio,
                actual_ratio,
                accepted,
                "scheme.compress_result",
            );
        }
        (Some(estimated_ratio), None) => {
            tracing::debug!(
                target: TARGET_TRACE,
                scheme = %scheme,
                before_nbytes,
                after_nbytes,
                estimated_ratio,
                accepted,
                "scheme.compress_result",
            );
        }
        (None, Some(actual_ratio)) => {
            tracing::debug!(
                target: TARGET_TRACE,
                scheme = %scheme,
                before_nbytes,
                after_nbytes,
                actual_ratio,
                accepted,
                "scheme.compress_result",
            );
        }
        (None, None) => {
            tracing::debug!(
                target: TARGET_TRACE,
                scheme = %scheme,
                before_nbytes,
                after_nbytes,
                accepted,
                "scheme.compress_result",
            );
        }
    }
}

/// Emits a sampling-failure event.
#[inline]
pub(super) fn sample_compress_failed(
    scheme: SchemeId,
    ctx: Option<&CompressorContext>,
    err: &impl fmt::Display,
) {
    if let Some(ctx) = ctx {
        tracing::error!(
            target: TARGET_TRACE,
            scheme = %scheme,
            cascade_path = %ctx.cascade_path(),
            cascade_depth = ctx.cascade_depth(),
            error = %err,
            "sample.compress_failed",
        );
    }
}

/// Emits the sampling result event.
#[inline]
pub(super) fn sample_result(
    scheme: SchemeId,
    sampled_before: u64,
    sampled_after: u64,
    sampled_ratio: Option<f64>,
) {
    if let Some(sampled_ratio) = sampled_ratio {
        tracing::debug!(
            target: TARGET_TRACE,
            scheme = %scheme,
            sampled_before,
            sampled_after,
            sampled_ratio,
            "sample.result",
        );
    } else {
        tracing::debug!(
            target: TARGET_TRACE,
            scheme = %scheme,
            sampled_before,
            sampled_after,
            "sample.result",
        );
    }
}
