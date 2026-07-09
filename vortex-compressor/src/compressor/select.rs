// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheme selection: estimating each eligible scheme and choosing the winner.

use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use super::ROOT_SCHEME_ID;
use super::sample::estimate_compression_ratio_with_sampling;
use crate::CascadingCompressor;
use crate::candidate::Candidate;
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

/// Returns `true` if `score` beats the current best candidate.
fn is_better_score(score: EstimateScore, best: Option<&Candidate>) -> bool {
    score.is_valid() && best.is_none_or(|candidate| score.beats(candidate.score))
}

impl CascadingCompressor {
    /// Calls [`expected_compression_ratio`] on each candidate and returns the winning scheme along
    /// with its resolved winner estimate, or `None` if no scheme beats the canonical encoding.
    ///
    /// Selection runs in two passes. Pass 1 evaluates every immediate
    /// [`CompressionEstimate::Verdict`] and tracks the running best. [`Scheme`]s returning
    /// [`CompressionEstimate::Deferred`] are stashed for pass 2 so that we do not make any
    /// expensive computations if we don't have to.
    ///
    /// Pass 2 evaluates the deferred work and, for each [`DeferredEstimate::Callback`], passes the
    /// current best [`EstimateScore`] as an early-exit hint so the callback can return
    /// [`EstimateVerdict::Skip`] without doing expensive work when it cannot beat the threshold.
    ///
    /// Ties are broken by registration order within each pass.
    ///
    /// [`expected_compression_ratio`]: Scheme::expected_compression_ratio
    pub(super) fn choose_best_scheme(
        &self,
        schemes: &[&'static dyn Scheme],
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<(&'static dyn Scheme, WinnerEstimate)>> {
        let mut best: Option<Candidate> = None;
        let mut deferred: Vec<(&'static dyn Scheme, DeferredEstimate)> = Vec::new();

        let input_nbytes = data.array().nbytes();
        let n_values = data.array().len() as u64;
        let new_candidate = |scheme, score, sampled| Candidate {
            scheme,
            score,
            input_nbytes,
            n_values,
            sampled,
            cascade: compress_ctx.cascade_history().to_vec(),
        };

        // Pass 1: evaluate every immediate verdict. Stash deferred work for pass 2.
        {
            let _verdict_pass = trace::verdict_pass_span().entered();
            for &scheme in schemes {
                match scheme.expected_compression_ratio(data, compress_ctx.clone(), exec_ctx) {
                    CompressionEstimate::Verdict(EstimateVerdict::Skip) => {}
                    CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse) => {
                        return Ok(Some((scheme, WinnerEstimate::AlwaysUse)));
                    }
                    CompressionEstimate::Verdict(EstimateVerdict::Ratio(ratio)) => {
                        let score = EstimateScore::FiniteCompression(ratio);

                        if is_better_score(score, best.as_ref()) {
                            best = Some(new_candidate(scheme, score, None));
                        }
                    }
                    CompressionEstimate::Deferred(deferred_estimate) => {
                        deferred.push((scheme, deferred_estimate));
                    }
                }
            }
        }

        // Pass 2: run deferred work. Callbacks receive the current best as a threshold so they can
        // short-circuit with `Skip` when they cannot beat it.
        for (scheme, deferred_estimate) in deferred {
            let _span = trace::scheme_eval_span(scheme.id()).entered();
            let threshold: Option<EstimateScore> = best.as_ref().map(|candidate| candidate.score);
            match deferred_estimate {
                DeferredEstimate::Sample => {
                    let estimate = estimate_compression_ratio_with_sampling(
                        self,
                        scheme,
                        data.array(),
                        compress_ctx.clone(),
                        exec_ctx,
                    )?;

                    if is_better_score(estimate.score, best.as_ref()) {
                        best = Some(new_candidate(
                            scheme,
                            estimate.score,
                            Some(estimate.sampled),
                        ));
                    }
                }
                DeferredEstimate::Callback(callback) => {
                    match callback(self, data, threshold, compress_ctx.clone(), exec_ctx)? {
                        EstimateVerdict::Skip => {}
                        EstimateVerdict::AlwaysUse => {
                            return Ok(Some((scheme, WinnerEstimate::AlwaysUse)));
                        }
                        EstimateVerdict::Ratio(ratio) => {
                            let score = EstimateScore::FiniteCompression(ratio);

                            if is_better_score(score, best.as_ref()) {
                                best = Some(new_candidate(scheme, score, None));
                            }
                        }
                    }
                }
            }
        }

        // The sampled arrays carried by candidates are dropped here; only the winning scheme
        // and its score survive selection.
        Ok(best.map(|candidate| (candidate.scheme, WinnerEstimate::Score(candidate.score))))
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
