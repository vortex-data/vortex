// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recording and replaying the scheme decisions that the compressor makes for one array.
//!
//! A [`CompressionPlan`] maps each cascade position (the chain of `(scheme, child)` pairs from the
//! root) to the decision that the compressor made there. When the compressor replays a plan, it
//! skips scheme estimation at each position that the plan covers and uses the recorded decision.
//! Positions that the plan does not cover use the full selection, and the new decisions are
//! recorded.
//!
//! The plan also records the compression ratio at each position. When the ratio of a replayed
//! child position deviates too much from the recorded ratio, the compressor also selects schemes
//! again at that position and keeps the smaller result. Thus a change in the data of one child,
//! which the ratio of the whole array can hide, still gets a new decision.

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
    /// The compression ratio of the decision at the position.
    pub(crate) ratio: f64,
}

/// The decisions that the compressor made for one array, keyed by cascade position.
#[derive(Debug, Clone, Default)]
pub(crate) struct CompressionPlan {
    /// The decision at each cascade position.
    decisions: FxHashMap<CascadePosition, Planned>,
}

impl CompressionPlan {
    /// Returns the decision at the root position, if the compressor made one.
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

/// The plan state that a [`CompressorContext`](crate::scheme::CompressorContext) carries.
///
/// It holds an optional plan to replay and a recorder for the decisions of the current call.
#[derive(Debug)]
pub(crate) struct PlanState {
    /// The plan to replay, if any.
    replay: Option<Arc<CompressionPlan>>,
    /// The largest relative deviation of the ratio at a replayed child position from the ratio
    /// in the plan, before the compressor selects schemes again at that position.
    max_ratio_deviation: f64,
    /// The decisions of the current call.
    recorded: Mutex<CompressionPlan>,
}

impl PlanState {
    /// Creates a state that replays `replay` (if given) and records all decisions.
    pub(crate) fn new(replay: Option<Arc<CompressionPlan>>, max_ratio_deviation: f64) -> Self {
        Self {
            replay,
            max_ratio_deviation,
            recorded: Mutex::new(CompressionPlan::default()),
        }
    }

    /// Returns the replayed decision at `position`, if the plan has one.
    pub(crate) fn planned(&self, position: &[(SchemeId, usize)]) -> Option<Planned> {
        self.replay
            .as_ref()
            .and_then(|plan| plan.decisions.get(position).copied())
    }

    /// Returns `true` if a replayed `ratio` is close enough to the `planned` ratio.
    pub(crate) fn accepts(&self, ratio: f64, planned: &Planned) -> bool {
        ratio_within(ratio, planned.ratio, self.max_ratio_deviation)
    }

    /// Records `decision`, which gave the compression `ratio`, at `position`.
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
