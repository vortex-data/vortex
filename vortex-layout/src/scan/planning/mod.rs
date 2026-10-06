// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Layout-level planners and morsels for the planning protocol.
//!
//! A split is evaluated over physical plans: an [`AnnouncePlanner`] tells the IO service which
//! segments the split is likely to read, a pruning [`FilterPlanner`] drops rows whose zone
//! statistics prove the filter false, a [`FilterPlanner`] runs the filter plan over the rows that
//! remain, and a [`ProjectionPlanner`] emits a [`ProjectionMorsel`] that runs the projection plan
//! over the rows that survive. Each is a separate stage, and the evaluating ones execute through
//! the plan exec graph; [`plan_split`] composes them for one split.
//!
//! Planners that decide which splits to read, and that know where each segment lives in a file,
//! belong to the crates that own those sources.

mod announce;
mod filter;
mod graph;
mod projection;

use std::sync::Arc;

pub use announce::AnnouncePlanner;
pub use filter::FilterPlanner;
pub use filter::FilterPlans;
pub use filter::SelectedRows;
pub use projection::ProjectionMorsel;
pub use projection::ProjectionPlanner;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_io::request::IoTarget;
use vortex_mask::Mask;
use vortex_scan::planning::next::Next;
use vortex_scan::planning::next::next_fn;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_session::VortexSession;

use crate::plan::PlanRef;
use crate::plan::exec::DecodeCache;

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
    pub fn target(&self) -> IoTarget {
        IoTarget::Range {
            offset: self.offset,
            len: self.length as usize,
            alignment: self.alignment,
        }
    }
}

/// What every split of a scan shares: the projection plan and where its segments live.
#[derive(Clone)]
pub struct ScanPlans {
    /// The session used for decoding and expression evaluation.
    pub session: VortexSession,
    /// The location of every segment, indexed by segment id.
    pub locations: Arc<[SegmentLocation]>,
    /// The optimized projection plan, over the scan's root row domain.
    pub projection: PlanRef,
    /// Where the chunks the projection reads start, over the root row domain, sorted. A split's
    /// projection is cut at each of these, so every projection split reads at most one chunk of
    /// every column.
    pub projection_starts: Arc<[u64]>,
    /// The global row index of the root row domain's first row.
    pub row_offset: u64,
    /// Segments already decoded by the graphs of one split. [`plan_split`] gives each split its
    /// own.
    pub decoded: DecodeCache,
}

/// The work for one split: announce its likely reads, prune the rows of `scope` selected by
/// `mask` with zone statistics, filter the rows that remain, then project the rows that survive.
/// The pruning and filter stages are skipped when their plan is absent.
pub fn plan_split(
    plans: ScanPlans,
    pruning: Option<PlanRef>,
    filter: Option<FilterPlans>,
    scope: WorkScope,
    mask: Mask,
) -> VortexResult<Box<dyn Planner>> {
    Ok(Box::new(AnnouncePlanner::for_split(
        plans,
        pruning,
        filter,
        SelectedRows { scope, mask },
    )))
}

/// Builds execution stages only after the announcement. Split construction and early IO
/// announcements happen before tasks run; allocating continuations and the decode cache here
/// lets the split's execution task do that work.
fn plan_selected(
    plans: ScanPlans,
    pruning: Option<PlanRef>,
    filter: Option<FilterPlans>,
    selected: SelectedRows,
) -> VortexResult<Box<dyn Planner>> {
    let plans = ScanPlans {
        decoded: DecodeCache::default(),
        ..plans
    };
    if pruning.is_none() && filter.is_none() {
        return Ok(Box::new(ProjectionPlanner::new(plans, selected)));
    }
    let project: Next<SelectedRows> = {
        let plans = plans.clone();
        next_fn(move |selected| Ok(ProjectionPlanner::new(plans.clone(), selected)))
    };
    let Some(pruning) = pruning else {
        return Ok(match filter {
            Some(filter) => Box::new(FilterPlanner::new(
                plans,
                filter,
                selected.scope,
                selected.mask,
                project,
            )),
            None => Box::new(ProjectionPlanner::new(plans, selected)),
        });
    };
    let next = match filter {
        None => project,
        Some(filter) => {
            let plans = plans.clone();
            next_fn(move |selected: SelectedRows| {
                Ok(FilterPlanner::new(
                    plans.clone(),
                    filter.clone(),
                    selected.scope,
                    selected.mask,
                    Arc::clone(&project),
                ))
            })
        }
    };
    Ok(Box::new(FilterPlanner::pruning(
        plans,
        pruning,
        selected.scope,
        selected.mask,
        next,
    )))
}
