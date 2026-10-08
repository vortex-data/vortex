// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use tracing::debug;
use vortex_io::request::IoRequest;
use vortex_io::request::IoTarget;
use vortex_io::request::trace::next_id;
use vortex_io::request::trace::timestamp_ns;

use super::Item;
use super::Output;
use super::Work;
use crate::planning::morsel::MorselOutput;
use crate::planning::planner::PlannerOutput;

pub(super) struct Trace {
    pub(super) id: u64,
}

impl Trace {
    pub(super) fn new() -> Option<Self> {
        if !tracing::enabled!(target: "vortex_scan::driver", tracing::Level::DEBUG) {
            return None;
        }
        let trace = Self { id: next_id() };
        trace.event("start", 0, 0);
        Some(trace)
    }

    pub(super) fn event(&self, event: &str, ready: usize, parked: usize) {
        debug!(target: "vortex_scan::driver", run = self.id, ts_ns = timestamp_ns(), event,
            ready, parked, "scan driver");
    }

    pub(super) fn work(&self, event: &str, work: &Work) {
        let (kind, stage) = match &work.item {
            Item::Planner(planner) => ("planner", planner.trace_name()),
            Item::Morsel(morsel) => ("morsel", morsel.trace_name()),
        };
        debug!(target: "vortex_scan::driver", run = self.id, ts_ns = timestamp_ns(), event,
            owner = work.id.0, root = work.root.0, file = work.scope.file_ordinal,
            row_start = work.scope.rows.start, row_end = work.scope.rows.end, kind, stage,
            outstanding = work.outstanding.len(), state = ?work.state(), "scan driver");
    }

    pub(super) fn compute(&self, work: &Work, start_ns: u64, output: &Output) {
        let (outcome, rows) = match output {
            Output::Planner(PlannerOutput::Done) | Output::Morsel(MorselOutput::Done) => {
                ("done", 0)
            }
            Output::Planner(PlannerOutput::Continue) | Output::Morsel(MorselOutput::Continue) => {
                ("continue", 0)
            }
            Output::Planner(PlannerOutput::NeedsIO(_))
            | Output::Morsel(MorselOutput::NeedsIO(_)) => ("io", 0),
            Output::Planner(PlannerOutput::Planner(..)) => ("planner", 0),
            Output::Planner(PlannerOutput::Morsel(..)) => ("morsel", 0),
            Output::Morsel(MorselOutput::Batch(array)) => ("batch", array.len()),
        };
        debug!(target: "vortex_scan::driver", run = self.id, ts_ns = timestamp_ns(),
            event = "compute", owner = work.id.0, root = work.root.0, start_ns, outcome, rows,
            state = ?work.state(), "scan driver");
    }

    pub(super) fn request(&self, work: &Work, request: &IoRequest) {
        let (target, offset, length) = match request.target {
            IoTarget::Range { offset, len, .. } => ("range", offset, len),
            IoTarget::Size => ("size", 0, 0),
        };
        debug!(target: "vortex_scan::driver", run = self.id, ts_ns = timestamp_ns(),
            event = "request", owner = work.id.0, root = work.root.0,
            request = request.request.0, intent = ?request.intent, target, offset, length,
            "scan driver");
    }
}
