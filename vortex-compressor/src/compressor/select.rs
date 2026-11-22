// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheme selection: estimating each eligible scheme and choosing the winner.

use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use super::ROOT_SCHEME_ID;
use super::sample::estimate_compression_ratio_with_sampling;
use crate::CascadingCompressor;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::EstimateScore;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::trace;

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
fn is_better_score(score: EstimateScore, best: Option<EstimateScore>) -> bool {
    score.is_valid() && best.is_none_or(|best| score.beats(best))
}

/// Records a valid `score` as a candidate and as the running best if it beats it.
fn push_score(
    best: &mut Option<EstimateScore>,
    ranked: &mut Vec<(&'static dyn Scheme, EstimateScore)>,
    scheme: &'static dyn Scheme,
    score: EstimateScore,
) {
    if !score.is_valid() {
        return;
    }
    if is_better_score(score, *best) {
        *best = Some(score);
    }
    ranked.push((scheme, score));
}

/// Orders `ranked` best first, keeping evaluation order among equal scores.
fn sort_ranked(ranked: &mut [(&'static dyn Scheme, EstimateScore)]) {
    // Every ranked score is valid, so it has a finite ratio greater than one.
    ranked.sort_by(|(_, a), (_, b)| {
        let a = a.finite_ratio().unwrap_or(f64::NEG_INFINITY);
        let b = b.finite_ratio().unwrap_or(f64::NEG_INFINITY);
        b.total_cmp(&a)
    });
}

/// Puts an [`EstimateVerdict::AlwaysUse`] scheme ahead of the candidates ranked before it.
fn always_use_first(
    scheme: &'static dyn Scheme,
    mut ranked: Vec<(&'static dyn Scheme, EstimateScore)>,
) -> Vec<(&'static dyn Scheme, WinnerEstimate)> {
    sort_ranked(&mut ranked);
    std::iter::once((scheme, WinnerEstimate::AlwaysUse))
        .chain(
            ranked
                .into_iter()
                .map(|(scheme, score)| (scheme, WinnerEstimate::Score(score))),
        )
        .collect()
}

impl CascadingCompressor {
    /// Calls [`expected_compression_ratio`] on each candidate and returns every scheme expected
    /// to beat the canonical encoding, best first, along with its winner estimate. The list is
    /// empty if no scheme is expected to beat it.
    ///
    /// The caller compresses with the first scheme and falls back to the next one when an
    /// estimate turns out to be too optimistic, so the order matters beyond the first entry.
    ///
    /// Selection runs in two passes. Pass 1 evaluates every immediate
    /// [`CompressionEstimate::Verdict`] and tracks the running best. [`Scheme`]s returning
    /// [`CompressionEstimate::Deferred`] are stashed for pass 2 so that we do not make any
    /// expensive computations if we don't have to.
    ///
    /// The first scheme returning [`EstimateVerdict::AlwaysUse`] in pass 1 is the default. It wins
    /// at once when every other candidate skips. Otherwise it is sampled in pass 2 like a
    /// [`DeferredEstimate::Sample`] and ranked by that estimate. If its sample is not a valid
    /// estimate, it is ranked last, so it is still tried when every other candidate fails.
    ///
    /// Pass 2 evaluates the deferred work. Every [`DeferredEstimate::Sample`] runs before any
    /// [`DeferredEstimate::Callback`], and each callback receives the current best
    /// [`EstimateScore`] as an early-exit hint so it can return [`EstimateVerdict::Skip`] without
    /// doing expensive work when it cannot beat the threshold. A callback that skips is not
    /// ranked, so it is never a fallback.
    ///
    /// A callback returning [`EstimateVerdict::AlwaysUse`] stops evaluation: that scheme comes
    /// first, followed by the schemes ranked before it. Remaining schemes are not evaluated.
    ///
    /// Ties are broken by evaluation order: immediate verdicts, then samples, then the default, then
    /// callbacks, each in registration order.
    ///
    /// [`expected_compression_ratio`]: Scheme::expected_compression_ratio
    pub(super) fn rank_schemes(
        &self,
        schemes: &[&'static dyn Scheme],
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<(&'static dyn Scheme, WinnerEstimate)>> {
        let mut best: Option<EstimateScore> = None;
        let mut ranked: Vec<(&'static dyn Scheme, EstimateScore)> = Vec::new();
        let mut deferred: Vec<(&'static dyn Scheme, DeferredEstimate)> = Vec::new();
        // The first `AlwaysUse` scheme is the default, which the other candidates have to beat.
        let mut default: Option<&'static dyn Scheme> = None;

        // Pass 1: evaluate every immediate verdict. Stash deferred work for pass 2.
        {
            let _verdict_pass = trace::verdict_pass_span().entered();
            for &scheme in schemes {
                match scheme.expected_compression_ratio(data, compress_ctx.clone(), exec_ctx) {
                    CompressionEstimate::Verdict(EstimateVerdict::Skip) => {}
                    CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse) => {
                        default.get_or_insert(scheme);
                    }
                    CompressionEstimate::Verdict(EstimateVerdict::Ratio(ratio)) => {
                        push_score(
                            &mut best,
                            &mut ranked,
                            scheme,
                            EstimateScore::FiniteCompression(ratio),
                        );
                    }
                    CompressionEstimate::Deferred(deferred_estimate) => {
                        deferred.push((scheme, deferred_estimate));
                    }
                }
            }
        }

        if let Some(default) = default {
            // Without other candidates, the default wins without being estimated.
            if best.is_none() && deferred.is_empty() {
                return Ok(vec![(default, WinnerEstimate::AlwaysUse)]);
            }
            // Otherwise it competes on a sample, after the candidates that already have scores, so
            // that it has to beat them.
            deferred.push((default, DeferredEstimate::Sample));
        }

        // Pass 2: run deferred work. Callbacks receive the current best as a threshold so they can
        // short-circuit with `Skip` when they cannot beat it, so samples run first to tighten it.
        deferred.sort_by_key(|(_, estimate)| matches!(estimate, DeferredEstimate::Callback(_)));
        for (scheme, deferred_estimate) in deferred {
            let _span = trace::scheme_eval_span(scheme.id()).entered();
            match deferred_estimate {
                DeferredEstimate::Sample => {
                    let score = estimate_compression_ratio_with_sampling(
                        self,
                        scheme,
                        data,
                        compress_ctx.clone(),
                        exec_ctx,
                    )?;
                    push_score(&mut best, &mut ranked, scheme, score);
                }
                DeferredEstimate::Callback(callback) => {
                    let threshold = best;
                    match callback(self, data, threshold, compress_ctx.clone(), exec_ctx)? {
                        EstimateVerdict::Skip => {}
                        EstimateVerdict::AlwaysUse => {
                            return Ok(always_use_first(scheme, ranked));
                        }
                        EstimateVerdict::Ratio(ratio) => {
                            push_score(
                                &mut best,
                                &mut ranked,
                                scheme,
                                EstimateScore::FiniteCompression(ratio),
                            );
                        }
                    }
                }
            }
        }

        sort_ranked(&mut ranked);
        let mut ranked: Vec<_> = ranked
            .into_iter()
            .map(|(scheme, score)| (scheme, WinnerEstimate::Score(score)))
            .collect();
        // A default without a valid estimate, e.g. because every sample holds only nulls, is the
        // last resort, and the only candidate when nothing else has a valid estimate.
        if let Some(default) = default
            && !ranked.iter().any(|(scheme, _)| scheme.id() == default.id())
        {
            ranked.push((default, WinnerEstimate::AlwaysUse));
        }
        Ok(ranked)
    }

    // TODO(connor): Lots of room for optimization here.
    /// Returns `true` if the candidate scheme should be excluded based on the cascade history and
    /// exclusion rules.
    pub(super) fn is_excluded(&self, candidate: &dyn Scheme, ctx: &CompressorContext) -> bool {
        let id = candidate.id();
        let history = ctx.cascade_history();

        // Self-exclusion: no scheme appears twice in any chain.
        if history.iter().any(|&(sid, _)| sid == id) {
            return true;
        }

        let mut iter = history.iter().copied().peekable();

        // The root entry is always first in the history (if present). Check if the root has
        // excluded us.
        if let Some((_, child_idx)) = iter.next_if(|&(sid, _)| sid == ROOT_SCHEME_ID)
            && self
                .root_exclusions
                .iter()
                .any(|rule| rule.excluded == id && rule.children.contains(child_idx))
        {
            return true;
        }

        // Push rules: Check if any of our ancestors have excluded us.
        for (ancestor_id, child_idx) in iter {
            if let Some(ancestor) = self.schemes.iter().find(|s| s.id() == ancestor_id)
                && ancestor
                    .descendant_exclusions()
                    .iter()
                    .any(|rule| rule.excluded == id && rule.children.contains(child_idx))
            {
                return true;
            }
        }

        // Pull rules: Check if we have excluded ourselves because of our ancestors.
        for rule in candidate.ancestor_exclusions() {
            if history
                .iter()
                .any(|(sid, cidx)| *sid == rule.ancestor && rule.children.contains(*cidx))
            {
                return true;
            }
        }

        false
    }
}
