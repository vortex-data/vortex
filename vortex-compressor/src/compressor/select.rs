// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheme selection: evaluating each eligible scheme and choosing the winner.

use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use super::ROOT_SCHEME_ID;
use super::sample::evaluate_candidate_with_sampling;
use crate::CascadingCompressor;
use crate::cost::Candidate;
use crate::cost::Cost;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEvaluation;
use crate::scheme::ResolvedEvaluation;
use crate::scheme::Scheme;
use crate::scheme::SchemeEvaluation;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::trace;

/// The selector state retained for the current winner.
///
/// Unlike [`Candidate`], this owns no sampled array or cascade history.
struct BestCandidate {
    /// The currently selected scheme.
    scheme: &'static dyn Scheme,
    /// The model-defined cost used for later comparisons and tracing.
    cost: Cost,
}

/// Selection result carried into winner tracing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum SelectionOutcome {
    /// The scheme must be used immediately.
    AlwaysUse,
    /// The cost assigned to the winning candidate.
    Cost(Cost),
}

impl SelectionOutcome {
    /// Returns the traceable cost for the winning estimate.
    pub(super) fn trace_cost(self) -> Option<f64> {
        match self {
            Self::AlwaysUse => None,
            Self::Cost(cost) => Some(cost.value()),
        }
    }
}

impl CascadingCompressor {
    /// Calls [`Scheme::evaluate`] and returns the winning scheme with its selection outcome, or
    /// `None` if no candidate beats the canonical encoding.
    ///
    /// The compressor's [`CostModel`] computes a cost for each candidate; the winner is the
    /// candidate with the minimum cost, and only candidates with a cost strictly below
    /// [`CostModel::canonical_cost`] are eligible.
    ///
    /// Selection runs in two passes. Pass 1 evaluates every immediate candidate and tracks the
    /// running best cost. Deferred evaluations are stashed for pass 2. Every resolved candidate is
    /// handed to the configured cost model; scheme evaluation never observes the current winner.
    ///
    /// Ties are broken by registration order within each pass (displacing the running best
    /// requires a strictly lower cost).
    ///
    /// [`CostModel`]: crate::cost::CostModel
    /// [`CostModel::canonical_cost`]: crate::cost::CostModel::canonical_cost
    pub(super) fn choose_best_scheme(
        &self,
        schemes: &[&'static dyn Scheme],
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<(&'static dyn Scheme, SelectionOutcome)>> {
        let mut best: Option<BestCandidate> = None;
        let mut deferred: Vec<(&'static dyn Scheme, DeferredEvaluation)> = Vec::new();

        let canonical_cost = self
            .cost_model
            .canonical_cost(data, data.array().len() as u64);

        let cascade = compress_ctx.cascade_history();

        // Pass 1: evaluate every immediate candidate. Stash deferred work for pass 2.
        {
            let _immediate_pass = trace::immediate_evaluation_pass_span().entered();
            for &scheme in schemes {
                match scheme.evaluate(data, compress_ctx.clone(), exec_ctx) {
                    SchemeEvaluation::Skip => {}
                    SchemeEvaluation::AlwaysUse => {
                        return Ok(Some((scheme, SelectionOutcome::AlwaysUse)));
                    }
                    SchemeEvaluation::Candidate(estimate) => {
                        let candidate = Candidate::new(scheme, estimate, data, None, cascade);

                        if let Some(cost) = self.cost_if_better(
                            &candidate,
                            canonical_cost,
                            best.as_ref().map(|b| b.cost),
                        ) {
                            best = Some(BestCandidate { scheme, cost });
                        }
                    }
                    SchemeEvaluation::Deferred(deferred_evaluation) => {
                        deferred.push((scheme, deferred_evaluation));
                    }
                }
            }
        }

        // Pass 2: resolve deferred candidates without exposing current selection state.
        for (scheme, deferred_evaluation) in deferred {
            let _span = trace::scheme_eval_span(scheme.id()).entered();
            match deferred_evaluation {
                DeferredEvaluation::Sample => {
                    let sampled_candidate = evaluate_candidate_with_sampling(
                        self,
                        scheme,
                        data.array(),
                        compress_ctx.clone(),
                        exec_ctx,
                    )?;
                    let candidate = Candidate::new(
                        scheme,
                        sampled_candidate.estimate,
                        data,
                        Some(&sampled_candidate.sampled),
                        cascade,
                    );

                    if let Some(cost) = self.cost_if_better(
                        &candidate,
                        canonical_cost,
                        best.as_ref().map(|b| b.cost),
                    ) {
                        best = Some(BestCandidate { scheme, cost });
                    }
                }
                DeferredEvaluation::Callback(callback) => {
                    match callback(self, data, compress_ctx.clone(), exec_ctx)? {
                        ResolvedEvaluation::Skip => {}
                        ResolvedEvaluation::AlwaysUse => {
                            return Ok(Some((scheme, SelectionOutcome::AlwaysUse)));
                        }
                        ResolvedEvaluation::Candidate(estimate) => {
                            let candidate = Candidate::new(scheme, estimate, data, None, cascade);

                            if let Some(cost) = self.cost_if_better(
                                &candidate,
                                canonical_cost,
                                best.as_ref().map(|b| b.cost),
                            ) {
                                best = Some(BestCandidate { scheme, cost });
                            }
                        }
                    }
                }
            }
        }

        Ok(best.map(|candidate| (candidate.scheme, SelectionOutcome::Cost(candidate.cost))))
    }

    /// Computes a candidate's cost and returns it iff the candidate becomes the new best: the
    /// model must return `Some`, and the cost must be strictly below both the canonical baseline
    /// and the best so far. Strict `<` preserves evaluation-order tie-breaking.
    fn cost_if_better(
        &self,
        candidate: &Candidate<'_>,
        canonical_cost: Cost,
        best_cost: Option<Cost>,
    ) -> Option<Cost> {
        let cost = self.cost_model.cost(candidate)?;
        (cost < canonical_cost && best_cost.is_none_or(|best_cost| cost < best_cost))
            .then_some(cost)
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
