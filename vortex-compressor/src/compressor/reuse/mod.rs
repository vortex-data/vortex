// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reuse of scheme decisions across the chunks of one stream.

use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use super::cascade::exact_uncompressed_size;
use super::plan::CompressionPlan;
use super::plan::Decision;
use super::plan::PlanState;
use super::plan::RecordedPlan;
use super::plan::ratio_within;
use crate::CascadingCompressor;

/// Configures when [`PlanReuse`] replays a plan and when it compares the plan with a new selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanReuseOptions {
    /// The largest deviation, between 0 and 1, of a replayed ratio from the baseline ratio of the
    /// plan. A replayed ratio is accepted when it is between `baseline * (1 - max_ratio_deviation)`
    /// and `baseline / (1 - max_ratio_deviation)`. Outside this range, the compressor also selects
    /// schemes again and keeps the smaller result. The same limit applies to each child position
    /// of the plan.
    ///
    /// Both directions count: a ratio that rises can mean that a different scheme is now better.
    /// The two bounds are reciprocal factors, so a ratio that falls by a factor and a ratio that
    /// rises by the same factor get the same result. For example, with `0.1` and a baseline of
    /// `10.0`, the accepted range is `9.0` to approximately `11.11`.
    pub max_ratio_deviation: f64,

    /// The number of replays before the next chunk also compares the plan with a new selection,
    /// or `None` to compare only when the ratio deviates. This catches a plan that sampling noise
    /// made a poor choice from the start.
    pub challenge_after: Option<usize>,
}

impl Default for PlanReuseOptions {
    fn default() -> Self {
        Self {
            max_ratio_deviation: 0.1,
            challenge_after: Some(8),
        }
    }
}

/// The scheme decisions that one stream of chunks shares.
///
/// The first chunk uses the full scheme selection and records its decisions as a plan. Later chunks
/// replay the plan. When the replayed ratio deviates, and periodically, a chunk also gets a new
/// selection, and the smaller result decides which plan to keep. See [`PlanReuseOptions`].
///
/// Use one `PlanReuse` for each stream of chunks with the same dtype and similar data, for
/// example the chunks of one column. It is safe to share between threads.
#[derive(Debug)]
pub struct PlanReuse {
    /// The limits for a plan.
    options: PlanReuseOptions,
    /// Either the shared cached plan or a detached single-chunk reservation.
    state: Mutex<PlanReuseMode>,
}

impl Default for PlanReuse {
    fn default() -> Self {
        Self::new(PlanReuseOptions::default())
    }
}

/// Internal mode of a [`PlanReuse`].
#[derive(Debug)]
enum PlanReuseMode {
    /// Shared mode: `next_step` and `replace`/`keep` mutate the cached plan directly.
    Shared(Option<CachedPlan>),
    /// Detached single-chunk mode created by [`PlanReuse::next_chunk`]: replays a pre-decided
    /// [`Step`] without mutating the parent and records any [`PlanUpdate`] for
    /// [`PlanReuse::commit`] to apply in chunk order.
    Detached {
        /// The pre-decided step reserved for this chunk.
        step: Option<Step>,
        /// Any plan update recorded while compressing this chunk.
        update: Option<PlanUpdate>,
    },
}

/// Pending state transition recorded by a detached [`PlanReuse`].
#[derive(Debug)]
enum PlanUpdate {
    /// Replace the cached plan with a newly selected plan and its ratio.
    Replace {
        /// The newly selected compression plan.
        plan: CompressionPlan,
        /// The achieved compression ratio for the newly selected plan.
        ratio: f64,
    },
    /// Keep the cached plan after it won a comparison with `ratio`.
    Keep {
        /// The existing cached plan that won the comparison.
        expected: Arc<CompressionPlan>,
        /// Updated plan when a child position was replanned during replay.
        plan: Option<CompressionPlan>,
        /// The refreshed baseline compression ratio.
        ratio: f64,
    },
    /// Persist a replanned child subtree onto the cached plan without changing root baseline or
    /// replay count.
    UpdatePlan {
        /// The existing cached plan this replay was based on.
        expected: Arc<CompressionPlan>,
        /// The updated plan with replanned child decisions.
        plan: CompressionPlan,
    },
}

/// A plan and the data to check it against.
#[derive(Debug, Clone)]
struct CachedPlan {
    /// The decisions to replay.
    plan: Arc<CompressionPlan>,
    /// The ratio of the plan on the last chunk where it was compared with a new selection.
    ratio: f64,
    /// The number of replays since the last comparison.
    replays: usize,
}

/// What the compressor does with the next chunk.
#[derive(Debug)]
enum Step {
    /// Select schemes, because there is no plan.
    Select,
    /// Replay the plan, and compare with a new selection only if the ratio deviates.
    Replay(CachedPlan),
    /// Replay the plan and compare with a new selection.
    Challenge(CachedPlan),
}

impl PlanReuse {
    /// Creates an empty `PlanReuse` with the given options.
    pub fn new(options: PlanReuseOptions) -> Self {
        Self {
            options,
            state: Mutex::new(PlanReuseMode::Shared(None)),
        }
    }

    /// Returns `true` if a plan has been recorded for replay.
    pub fn has_plan(&self) -> bool {
        match &*self.state.lock() {
            PlanReuseMode::Shared(cached) => cached.is_some(),
            PlanReuseMode::Detached { step, .. } => {
                matches!(step, Some(Step::Replay(_) | Step::Challenge(_)))
            }
        }
    }

    /// Reserves the step for the next chunk in stream order and returns a detached `PlanReuse`
    /// for compressing that chunk concurrently.
    ///
    /// Pass the detached value to [`commit`](Self::commit) in chunk order after compression
    /// finishes so plan replacements and ratio updates happen deterministically.
    pub fn next_chunk(&self) -> Self {
        let step = self.next_step();
        Self {
            options: self.options,
            state: Mutex::new(PlanReuseMode::Detached {
                step: Some(step),
                update: None,
            }),
        }
    }

    /// Applies the plan update recorded by a detached `PlanReuse` from
    /// [`next_chunk`](Self::next_chunk).
    pub fn commit(&self, chunk: Self) {
        let PlanReuseMode::Detached { update, .. } = chunk.state.into_inner() else {
            return;
        };
        match update {
            Some(PlanUpdate::Replace { plan, ratio }) => self.replace(plan, ratio),
            Some(PlanUpdate::Keep {
                expected,
                plan,
                ratio,
            }) => self.keep(&expected, plan, ratio),
            Some(PlanUpdate::UpdatePlan { expected, plan }) => self.update_plan(&expected, plan),
            None => {}
        }
    }

    /// Returns what to do with the next chunk, and counts the replay.
    fn next_step(&self) -> Step {
        let mut state = self.state.lock();
        match &mut *state {
            PlanReuseMode::Shared(cached) => {
                let Some(entry) = cached.as_mut() else {
                    return Step::Select;
                };

                if self
                    .options
                    .challenge_after
                    .is_some_and(|limit| entry.replays >= limit)
                {
                    entry.replays = 0;
                    return Step::Challenge(entry.clone());
                }

                entry.replays += 1;
                Step::Replay(entry.clone())
            }
            PlanReuseMode::Detached { step, .. } => step.take().unwrap_or(Step::Select),
        }
    }

    /// Replaces the cached plan. A plan without a scheme at the root, such as the plan of a
    /// constant chunk, tells nothing about the next chunk, so it is ignored.
    fn replace(&self, plan: CompressionPlan, ratio: f64) {
        if !matches!(plan.root_decision(), Some(Decision::Scheme(_))) {
            return;
        }

        let mut state = self.state.lock();
        match &mut *state {
            PlanReuseMode::Shared(cached) => {
                *cached = Some(CachedPlan {
                    plan: Arc::new(plan),
                    ratio,
                    replays: 0,
                });
            }
            PlanReuseMode::Detached { update, .. } => {
                *update = Some(PlanUpdate::Replace { plan, ratio });
            }
        }
    }

    /// Keeps the cached plan after it won a comparison with `ratio`, applying `updated_plan` if
    /// any child position was replanned.
    fn keep(
        &self,
        expected: &Arc<CompressionPlan>,
        updated_plan: Option<CompressionPlan>,
        ratio: f64,
    ) {
        let mut state = self.state.lock();
        match &mut *state {
            PlanReuseMode::Shared(cached) => {
                if let Some(entry) = cached.as_mut()
                    && Arc::ptr_eq(&entry.plan, expected)
                {
                    if let Some(plan) = updated_plan {
                        entry.plan = Arc::new(plan);
                    }
                    entry.ratio = ratio;
                    entry.replays = 0;
                }
            }
            PlanReuseMode::Detached { update, .. } => {
                *update = Some(PlanUpdate::Keep {
                    expected: Arc::clone(expected),
                    plan: updated_plan,
                    ratio,
                });
            }
        }
    }

    /// Updates the cached plan in place when a child position was replanned while the root replay
    /// was accepted, preserving the root baseline ratio and replay counter.
    fn update_plan(&self, expected: &Arc<CompressionPlan>, plan: CompressionPlan) {
        let mut state = self.state.lock();
        match &mut *state {
            PlanReuseMode::Shared(cached) => {
                if let Some(entry) = cached.as_mut()
                    && Arc::ptr_eq(&entry.plan, expected)
                {
                    entry.plan = Arc::new(plan);
                }
            }
            PlanReuseMode::Detached { update, .. } => {
                *update = Some(PlanUpdate::UpdatePlan {
                    expected: Arc::clone(expected),
                    plan,
                });
            }
        }
    }

    /// Returns `true` if a replayed `ratio` is close enough to the baseline of `plan`.
    fn accepts(&self, ratio: f64, plan: &CachedPlan) -> bool {
        ratio_within(ratio, plan.ratio, self.options.max_ratio_deviation)
    }
}

impl CascadingCompressor {
    /// Compresses an array, reusing the scheme decisions that `reuse` holds from earlier arrays.
    ///
    /// The first array compresses like [`compress`](Self::compress). See [`PlanReuse`] for how
    /// later arrays replay the decisions.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress_reusing(
        &self,
        array: &ArrayRef,
        reuse: &PlanReuse,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let before_nbytes = exact_uncompressed_size(array, exec_ctx)?;
        let ratio_of =
            |compressed: &ArrayRef| compression_ratio(before_nbytes, compressed.nbytes());

        let (cached, always_challenge) = match reuse.next_step() {
            Step::Select => {
                let (selected, recorded) = self.compress_with_plan(array, None, reuse, exec_ctx)?;
                reuse.replace(recorded.plan, ratio_of(&selected));
                return Ok(selected);
            }
            Step::Replay(cached) => (cached, false),
            Step::Challenge(cached) => (cached, true),
        };

        let (replayed, recorded) =
            self.compress_with_plan(array, Some(Arc::clone(&cached.plan)), reuse, exec_ctx)?;
        let replayed_ratio = ratio_of(&replayed);

        let root_decision = recorded.plan.root_decision();
        if root_decision.is_none() {
            // Chunks short-circuited before scheme selection (e.g. constant or all-null) do not
            // exercise the cached plan and must not trigger re-selection or overwrite its baseline.
            return Ok(replayed);
        }

        if recorded.root_selected {
            // Replay already fell through to full root selection (e.g. the cached root scheme
            // rejected the chunk), so avoid repeating the same selection.
            if matches!(root_decision, Some(Decision::Scheme(_))) {
                reuse.replace(recorded.plan, replayed_ratio);
            }
            return Ok(replayed);
        }

        if matches!(root_decision, Some(Decision::Scheme(_)))
            && root_decision != cached.plan.root_decision()
        {
            reuse.replace(recorded.plan, replayed_ratio);
            return Ok(replayed);
        }

        if !always_challenge && reuse.accepts(replayed_ratio, &cached) {
            if recorded.child_replanned && matches!(root_decision, Some(Decision::Scheme(_))) {
                reuse.update_plan(&cached.plan, recorded.plan);
            }
            return Ok(replayed);
        }

        let (selected, selected_recorded) =
            self.compress_with_plan(array, None, reuse, exec_ctx)?;
        if selected.nbytes() < replayed.nbytes() {
            reuse.replace(selected_recorded.plan, ratio_of(&selected));
            return Ok(selected);
        }

        if matches!(root_decision, Some(Decision::Scheme(_))) {
            reuse.keep(
                &cached.plan,
                recorded.child_replanned.then_some(recorded.plan),
                replayed_ratio,
            );
        }
        Ok(replayed)
    }

    /// Compresses an array, replaying `replay` if given, and returns the recorded decisions and
    /// fallback flags.
    fn compress_with_plan(
        &self,
        array: &ArrayRef,
        replay: Option<Arc<CompressionPlan>>,
        reuse: &PlanReuse,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, RecordedPlan)> {
        let state = Arc::new(PlanState::new(replay, reuse.options.max_ratio_deviation));
        let compressed = self.compress_in_plan(array, Arc::clone(&state), exec_ctx)?;

        // All contexts that held the state are gone after the compression returns.
        let recorded = Arc::into_inner(state)
            .map(PlanState::into_recorded)
            .unwrap_or_default();

        Ok((compressed, recorded))
    }
}

/// Returns the compression ratio, which is infinite for an empty output.
fn compression_ratio(before_nbytes: u64, after_nbytes: u64) -> f64 {
    if after_nbytes == 0 {
        f64::INFINITY
    } else {
        before_nbytes as f64 / after_nbytes as f64
    }
}

#[cfg(test)]
mod tests;
