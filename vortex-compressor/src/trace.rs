// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Internal tracing helpers for compressor observability.

use std::fmt;

use crate::ctx::CompressorContext;
use crate::scheme::SchemeId;

/// Shared tracing target for compressor decisions and coarse cascade structure.
pub(super) const TARGET_TRACE: &str = "vortex_compressor::encode";

/// Builds the top-level compression span.
///
/// `input_nbytes` is known up front; `compressed_nbytes` / `compression_ratio` are filled in by
/// [`record_compress_outcome`] once the cascade returns.
#[inline]
pub(super) fn compress_span(
    len: usize,
    dtype: &impl fmt::Display,
    before_nbytes: u64,
) -> tracing::Span {
    tracing::debug_span!(
        target: TARGET_TRACE,
        "compress",
        array_len = len,
        dtype = %dtype,
        input_nbytes = before_nbytes,
        compressed_nbytes = tracing::field::Empty,
        compression_ratio = tracing::field::Empty,
    )
}

/// Builds a span covering on-demand materialization of a cached stats type.
///
/// Child of whatever span is active when a stats accessor first fires. Typically that's
/// [`verdict_pass_span`]; entering this span disambiguates stats cost from the rest of Pass 1.
/// `kind` is usually `std::any::type_name::<T>()` so the args identify which group was generated
/// (e.g. `IntegerStats`, `FloatStats`).
#[inline]
pub(super) fn generate_stats_span(kind: &'static str) -> tracing::Span {
    tracing::debug_span!(
        target: TARGET_TRACE,
        "generate_stats",
        stats_kind = kind,
    )
}

/// Builds a span covering Pass 1 of scheme selection (the cheap-verdict pass).
///
/// Stats batches merged across eligible schemes are materialized lazily by the first
/// `expected_compression_ratio` call that touches them. Grouping those calls under one span makes
/// the stats cost (and unexpectedly slow verdicts) visible independently of per-candidate sampling.
#[inline]
pub(super) fn verdict_pass_span() -> tracing::Span {
    tracing::debug_span!(
        target: TARGET_TRACE,
        "verdict_pass",
    )
}

/// Builds a span covering one deferred per-scheme evaluation (sample or callback).
///
/// `scheme_candidate` is the scheme being evaluated, not necessarily chosen.
#[inline]
pub(super) fn scheme_eval_span(scheme: SchemeId) -> tracing::Span {
    tracing::debug_span!(
        target: TARGET_TRACE,
        "scheme_eval",
        scheme_candidate = %scheme,
    )
}

/// Emits the sampling result event for zero-byte sample outputs.
#[inline]
pub(super) fn zero_byte_sample_result(scheme: SchemeId, sampled_before: u64) {
    tracing::debug!(
        target: TARGET_TRACE,
        scheme = %scheme,
        sampled_before,
        sampled_after = 0_u64,
        "sample.result",
    );
}

/// Builds a span covering the winning scheme's full-array compression.
///
/// `scheme_chosen` and `input_nbytes` are known up front. `compressed_nbytes`,
/// `estimated_ratio`, `achieved_ratio`, and `accepted` are filled in by
/// [`record_winner_compress_result`] once the encode completes.
#[inline]
pub(super) fn winner_compress_span(scheme: SchemeId, before_nbytes: u64) -> tracing::Span {
    tracing::debug_span!(
        target: TARGET_TRACE,
        "winner_compress",
        scheme_chosen = %scheme,
        input_nbytes = before_nbytes,
        compressed_nbytes = tracing::field::Empty,
        estimated_ratio = tracing::field::Empty,
        achieved_ratio = tracing::field::Empty,
        accepted = tracing::field::Empty,
    )
}

/// Records the outcome of a winning-scheme compression on the current `winner_compress` span.
#[inline]
pub(super) fn record_winner_compress_result(
    compressed_nbytes: u64,
    estimated_ratio: Option<f64>,
    achieved_ratio: Option<f64>,
    accepted: bool,
) {
    let span = tracing::Span::current();
    span.record("compressed_nbytes", compressed_nbytes);
    if let Some(r) = estimated_ratio {
        span.record("estimated_ratio", r);
    }
    if let Some(r) = achieved_ratio {
        span.record("achieved_ratio", r);
    }
    span.record("accepted", accepted);
}

/// Records the final output size and, when finite, the top-level compression ratio.
#[inline]
pub(super) fn record_compress_outcome(
    span: &tracing::Span,
    input_nbytes: u64,
    compressed_nbytes: u64,
) {
    span.record("compressed_nbytes", compressed_nbytes);
    if compressed_nbytes != 0 {
        span.record(
            "compression_ratio",
            input_nbytes as f64 / compressed_nbytes as f64,
        );
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
