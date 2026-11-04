// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optional reads composed ahead of a continuation planner.

use std::ops::Range;

use rustc_hash::FxHashSet;
use vortex_error::VortexResult;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_session::VortexSession;

use super::SegmentLocation;
use crate::plan::physical::PlanRef;
use crate::plan::physical::pipeline::Scan;
use crate::segments::SegmentId;

/// Publishes optional reads, then runs `next` in the same root and I/O session.
///
/// The hints do not wait for completion. In particular, projection reads can overlap a filter
/// continuation without starting projection compute before its selected rows are known.
pub(super) struct PrefetchPlanner {
    scope: WorkScope,
    requests: Option<IoBatch>,
    next: Option<Box<dyn Planner>>,
}

impl PrefetchPlanner {
    pub(super) fn new(scope: WorkScope, requests: IoBatch, next: Box<dyn Planner>) -> Self {
        Self {
            scope,
            requests: Some(requests),
            next: Some(next),
        }
    }
}

impl IoConsumer for PrefetchPlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for PrefetchPlanner {
    fn state(&self) -> State {
        if self.next.is_some() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if let Some(requests) = self.requests.take()
            && !requests.is_empty()
        {
            return Ok(PlannerOutput::NeedsIO(requests));
        }
        Ok(match self.next.take() {
            Some(next) => PlannerOutput::Planner(self.scope.clone(), next),
            None => PlannerOutput::Done,
        })
    }
}

/// The segments `plans` read over `rows` under `mask`, each once, in id order.
pub(super) fn segments_read<'a>(
    session: &VortexSession,
    plans: impl Iterator<Item = &'a PlanRef>,
    rows: Range<u64>,
    mask: &Mask,
) -> VortexResult<Vec<SegmentId>> {
    let mut segments = Vec::new();
    for plan in plans {
        segments.extend(Scan::segments(session, plan, rows.clone(), mask)?);
    }
    segments.sort_unstable();
    segments.dedup();
    Ok(segments)
}

/// Publishes a hint for each segment not in `seen`. Call with stronger hints first.
pub(super) fn hint(
    locations: &[SegmentLocation],
    next_hint: &mut u32,
    intent: IoIntent,
    segments: &[SegmentId],
    seen: &mut FxHashSet<SegmentId>,
    batch: &mut IoBatch,
) {
    for &segment in segments {
        if !seen.insert(segment) {
            continue;
        }
        batch.push(IoRequest {
            intent,
            request: IoRequestId(*next_hint),
            target: locations[*segment as usize].target(),
        });
        *next_hint -= 1;
    }
}
