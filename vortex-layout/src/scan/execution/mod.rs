// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan execution over physical plans through the planner I/O and compute protocol.
//!
//! A `ProjectionPlanner` emits one `ProjectionMorsel` for each fixed-size work chunk. The morsel
//! runs the physical projection plan over the chunk's selected rows through
//! the pipeline [`Scan`](crate::plan::physical::pipeline::Scan). Every segment read is published
//! through a `ScanAdaptor`, so the planning driver can advance other work while reads wait.

mod adaptor;
mod projection;

use std::sync::Arc;

use vortex_buffer::Alignment;
use vortex_io::request::IoTarget;
use vortex_scan::planning::next::next_fn;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::planning::selection::SelectedRows;
use vortex_scan::planning::selection::SelectionPlanner;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use self::projection::ProjectionPlanner;
use crate::plan::physical::PlanRef;
use crate::plan::physical::pipeline::ValueCache;

/// Where a segment's bytes live in its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentLocation {
    /// Byte offset of the segment from the start of the source.
    pub offset: u64,
    /// Length of the segment in bytes.
    pub length: u32,
    /// Alignment the segment's bytes must have once delivered.
    pub alignment: Alignment,
}

impl SegmentLocation {
    /// The byte range a read of the segment asks for.
    fn target(&self) -> IoTarget {
        IoTarget::Range {
            offset: self.offset,
            len: self.length as usize,
            alignment: self.alignment,
        }
    }
}

/// What every split of a scan shares: the projection plan and where its segments live.
#[derive(Clone)]
pub(super) struct ScanPlans {
    /// The session used for decoding and expression evaluation.
    pub(super) session: VortexSession,
    /// The location of every segment, indexed by segment id.
    pub(super) locations: Arc<[SegmentLocation]>,
    /// The optimized projection plan, over the scan's root row domain.
    pub(super) projection: PlanRef,
    /// The global row index of the root row domain's first row.
    pub(super) row_offset: u64,
    /// Whole values, such as dictionary values, shared by the splits of this scan. Their plans
    /// always read the same complete row domain, independently of a split's selection.
    pub(super) values: ValueCache,
}

/// Selects the rows of a fixed work chunk before handing them to its projection planner.
pub(super) fn plan_projection(
    plans: ScanPlans,
    scope: WorkScope,
    selection: Arc<Selection>,
) -> Box<dyn Planner> {
    Box::new(SelectionPlanner::new(
        scope,
        selection,
        next_fn(move |selected| Ok(ProjectionPlanner::new(plans.clone(), selected))),
    ))
}
