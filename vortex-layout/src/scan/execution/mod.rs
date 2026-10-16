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
use vortex_io::segments::SegmentSourceId;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
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

/// What every split of a scan shares: the projection plan and where its segments live.
#[derive(Clone)]
pub(super) struct ScanPlans {
    /// The session used for decoding and expression evaluation.
    pub(super) session: VortexSession,
    /// The location of every segment, indexed by segment id.
    pub(super) locations: Arc<[SegmentLocation]>,
    /// The file's id in the segment IO service its splits read through.
    pub(super) source: SegmentSourceId,
    /// The optimized projection plan, over the scan's root row domain.
    pub(super) projection: PlanRef,
    /// The global row index of the root row domain's first row.
    pub(super) row_offset: u64,
    /// Whole values, such as dictionary values, shared by the splits of this scan. Their plans
    /// always read the same complete row domain, independently of a split's selection.
    pub(super) values: ValueCache,
}

/// The rows of a split selected for projection.
struct SelectedRows {
    /// The split the rows belong to.
    scope: WorkScope,
    /// Which rows of `scope.rows` are selected.
    mask: Mask,
}

/// The work for one split: project the rows of `scope` selected by `mask`.
pub(super) fn plan_projection(plans: ScanPlans, scope: WorkScope, mask: Mask) -> Box<dyn Planner> {
    Box::new(ProjectionPlanner::new(plans, SelectedRows { scope, mask }))
}
