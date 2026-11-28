// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recording and replaying the scheme decisions that the compressor makes for one array.
//!
//! A [`CompressionPlan`] maps each cascade position to a decision and the ratio it gave. A replay
//! skips scheme estimation at the positions that the plan covers. A replayed child whose ratio
//! deviates too much also gets a new selection, and the smaller result wins.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::scheme::SchemeId;

/// A cascade position: the `(scheme, child index)` chain from the root.
type CascadePosition = Vec<(SchemeId, usize)>;

/// The decision that the compressor made at one cascade position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// The compressor encoded the array with this scheme.
    Scheme(SchemeId),
    /// No scheme was better than the canonical array.
    Unchanged,
}

/// A decision and the compression ratio that it gave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Planned {
    /// The decision at the position.
    pub(crate) decision: Decision,
    /// The compression ratio that the decision gave.
    pub(crate) ratio: f64,
}

/// The decisions that the compressor made for one array.
#[derive(Debug, Clone, Default)]
pub(crate) struct CompressionPlan {
    /// The decision at each cascade position.
    decisions: FxHashMap<CascadePosition, Planned>,
}

impl CompressionPlan {
    /// Returns the decision at the root position, if any.
    pub(crate) fn root_decision(&self) -> Option<Decision> {
        self.decisions
            .get([].as_slice())
            .map(|planned| planned.decision)
    }

    #[cfg(test)]
    pub(crate) fn decision(&self, position: &[(SchemeId, usize)]) -> Option<Decision> {
        self.decisions.get(position).map(|planned| planned.decision)
    }
}

/// Returns `true` if `ratio` is between `baseline * (1 - max_deviation)` and
/// `baseline / (1 - max_deviation)`.
pub(crate) fn ratio_within(ratio: f64, baseline: f64, max_deviation: f64) -> bool {
    let keep = 1.0 - max_deviation;
    ratio >= baseline * keep && ratio * keep <= baseline
}

/// Outcome of recording a compression pass through [`PlanState`].
#[derive(Debug, Default)]
pub(crate) struct RecordedPlan {
    /// Recorded scheme decisions.
    pub(crate) plan: CompressionPlan,
    /// `true` if any non-root child fell back from the replayed plan.
    pub(crate) child_replanned: bool,
    /// `true` if the root fell through from replay to full selection.
    pub(crate) root_selected: bool,
}

/// The plan to replay and the recorder for the current call, carried by a
/// [`CompressorContext`](crate::scheme::CompressorContext).
#[derive(Debug)]
pub(crate) struct PlanState {
    /// The plan to replay, if any.
    replay: Option<Arc<CompressionPlan>>,
    /// See [`PlanReuseOptions::max_ratio_deviation`](crate::PlanReuseOptions).
    max_ratio_deviation: f64,
    /// The decisions of the current call.
    recorded: Mutex<CompressionPlan>,
    /// Set when any child fell back from `replay` to full selection.
    child_replanned: AtomicBool,
    /// Set when the root fell through from `replay` to full selection.
    root_selected: AtomicBool,
}

impl PlanState {
    /// Creates a state that replays `replay`, if given, and records all decisions.
    pub(crate) fn new(replay: Option<Arc<CompressionPlan>>, max_ratio_deviation: f64) -> Self {
        Self {
            replay,
            max_ratio_deviation,
            recorded: Mutex::new(CompressionPlan::default()),
            child_replanned: AtomicBool::new(false),
            root_selected: AtomicBool::new(false),
        }
    }

    /// Returns `true` if this state is replaying a cached plan.
    pub(crate) fn has_replay(&self) -> bool {
        self.replay.is_some()
    }

    /// Marks that a child position fell back from the replayed plan.
    pub(crate) fn mark_child_replanned(&self) {
        self.child_replanned.store(true, Ordering::Relaxed);
    }

    /// Marks that the root position fell through from replay to full selection.
    pub(crate) fn mark_root_selected(&self) {
        self.root_selected.store(true, Ordering::Relaxed);
    }

    /// Returns the replayed decision at `position`, if any.
    pub(crate) fn planned(&self, position: &[(SchemeId, usize)]) -> Option<Planned> {
        self.replay
            .as_ref()
            .and_then(|plan| plan.decisions.get(position).copied())
    }

    /// Returns `true` if a replayed `ratio` is close enough to the `planned` ratio.
    pub(crate) fn accepts(&self, ratio: f64, planned: &Planned) -> bool {
        ratio_within(ratio, planned.ratio, self.max_ratio_deviation)
    }

    /// Records `decision` and its `ratio` at `position`.
    pub(crate) fn record(&self, position: &[(SchemeId, usize)], decision: Decision, ratio: f64) {
        self.recorded
            .lock()
            .decisions
            .insert(position.to_vec(), Planned { decision, ratio });
    }

    /// Removes and returns all recorded decisions at or beneath `prefix`.
    pub(crate) fn take_subtree(
        &self,
        prefix: &[(SchemeId, usize)],
    ) -> FxHashMap<CascadePosition, Planned> {
        let mut recorded = self.recorded.lock();
        let mut taken = FxHashMap::default();
        recorded.decisions.retain(|pos, planned| {
            if pos.starts_with(prefix) {
                taken.insert(pos.clone(), *planned);
                false
            } else {
                true
            }
        });
        taken
    }

    /// Replaces all recorded decisions at or beneath `prefix` with `snapshot`.
    pub(crate) fn restore_subtree(
        &self,
        prefix: &[(SchemeId, usize)],
        snapshot: FxHashMap<CascadePosition, Planned>,
    ) {
        let mut recorded = self.recorded.lock();
        recorded.decisions.retain(|pos, _| !pos.starts_with(prefix));
        recorded.decisions.extend(snapshot);
    }

    /// Consumes the state and returns the recorded plan and fallback flags.
    pub(crate) fn into_recorded(self) -> RecordedPlan {
        RecordedPlan {
            plan: self.recorded.into_inner(),
            child_replanned: self.child_replanned.into_inner(),
            root_selected: self.root_selected.into_inner(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::ratio_within;

    #[rstest]
    #[case::inside_lower_bound(9.01, 10.0, true)]
    #[case::below_lower_bound(8.99, 10.0, false)]
    #[case::inside_upper_bound(11.1, 10.0, true)]
    #[case::above_upper_bound(11.12, 10.0, false)]
    #[case::infinite_ratio_and_baseline(f64::INFINITY, f64::INFINITY, true)]
    #[case::infinite_ratio_finite_baseline(f64::INFINITY, 10.0, false)]
    #[case::finite_ratio_infinite_baseline(10.0, f64::INFINITY, false)]
    fn ratio_within_uses_reciprocal_bounds(
        #[case] ratio: f64,
        #[case] baseline: f64,
        #[case] expected: bool,
    ) {
        assert_eq!(ratio_within(ratio, baseline, 0.1), expected);
    }
}
