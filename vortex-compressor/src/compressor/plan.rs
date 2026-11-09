// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recording and replaying the scheme decisions that the compressor makes for one array.
//!
//! A [`CompressionPlan`] maps each cascade position to a decision and the ratio it gave. A replay
//! skips scheme estimation at the positions that the plan covers. A replayed child whose ratio
//! deviates too much also gets a new selection, and the smaller result wins.

use std::sync::Arc;

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
}

/// Returns `true` if `ratio` is between `baseline * (1 - max_deviation)` and
/// `baseline / (1 - max_deviation)`.
pub(crate) fn ratio_within(ratio: f64, baseline: f64, max_deviation: f64) -> bool {
    let keep = 1.0 - max_deviation;
    ratio >= baseline * keep && ratio * keep <= baseline
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
}

impl PlanState {
    /// Creates a state that replays `replay`, if given, and records all decisions.
    pub(crate) fn new(replay: Option<Arc<CompressionPlan>>, max_ratio_deviation: f64) -> Self {
        Self {
            replay,
            max_ratio_deviation,
            recorded: Mutex::new(CompressionPlan::default()),
        }
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

    /// Consumes the state and returns the recorded plan.
    pub(crate) fn into_recorded(self) -> CompressionPlan {
        self.recorded.into_inner()
    }
}
