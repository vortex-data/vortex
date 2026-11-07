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
use super::plan::ratio_within;
use crate::CascadingCompressor;

/// Configures when [`PlanReuse`] replays a plan and when it compares the plan with a new selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanReuseOptions {
    /// The largest relative deviation, between 0 and 1, of a replayed ratio from the baseline
    /// ratio of the plan. Outside this range, the compressor also selects schemes again and keeps
    /// the smaller result. The same limit applies to each child position of the plan.
    ///
    /// Both directions count: a ratio that rises can mean that a different scheme is now better.
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
#[derive(Debug, Default)]
pub struct PlanReuse {
    /// The limits for a plan.
    options: PlanReuseOptions,
    /// The current plan, if a chunk made one.
    cached: Mutex<Option<CachedPlan>>,
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
            cached: Mutex::new(None),
        }
    }

    /// Returns what to do with the next chunk, and counts the replay.
    fn next_step(&self) -> Step {
        let mut cached = self.cached.lock();
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

    /// Replaces the cached plan. A plan without a scheme at the root, such as the plan of a
    /// constant chunk, tells nothing about the next chunk, so it is ignored.
    fn replace(&self, plan: CompressionPlan, ratio: f64) {
        if !matches!(plan.root_decision(), Some(Decision::Scheme(_))) {
            return;
        }

        *self.cached.lock() = Some(CachedPlan {
            plan: Arc::new(plan),
            ratio,
            replays: 0,
        });
    }

    /// Keeps the cached plan after it won a comparison with `ratio`.
    fn keep(&self, plan: &Arc<CompressionPlan>, ratio: f64) {
        let mut cached = self.cached.lock();
        if let Some(entry) = cached.as_mut()
            && Arc::ptr_eq(&entry.plan, plan)
        {
            entry.ratio = ratio;
            entry.replays = 0;
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
                let (selected, plan) = self.compress_with_plan(array, None, reuse, exec_ctx)?;
                reuse.replace(plan, ratio_of(&selected));
                return Ok(selected);
            }
            Step::Replay(cached) => (cached, false),
            Step::Challenge(cached) => (cached, true),
        };

        let (replayed, _) =
            self.compress_with_plan(array, Some(Arc::clone(&cached.plan)), reuse, exec_ctx)?;
        let replayed_ratio = ratio_of(&replayed);

        if !always_challenge && reuse.accepts(replayed_ratio, &cached) {
            return Ok(replayed);
        }

        let (selected, plan) = self.compress_with_plan(array, None, reuse, exec_ctx)?;
        if selected.nbytes() < replayed.nbytes() {
            reuse.replace(plan, ratio_of(&selected));
            return Ok(selected);
        }

        reuse.keep(&cached.plan, replayed_ratio);
        Ok(replayed)
    }

    /// Compresses an array, replaying `replay` if given, and returns the recorded decisions.
    fn compress_with_plan(
        &self,
        array: &ArrayRef,
        replay: Option<Arc<CompressionPlan>>,
        reuse: &PlanReuse,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, CompressionPlan)> {
        let state = Arc::new(PlanState::new(replay, reuse.options.max_ratio_deviation));
        let compressed = self.compress_in_plan(array, Arc::clone(&state), exec_ctx)?;

        // All contexts that held the state are gone after the compression returns.
        let plan = Arc::into_inner(state)
            .map(PlanState::into_recorded)
            .unwrap_or_default();

        Ok((compressed, plan))
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
