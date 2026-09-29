// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Layout-level planners and morsels for the planning protocol.
//!
//! Two ways to evaluate a split live here:
//!
//! - Over physical plans: a pruning [`FilterPlanner`] drops rows whose zone statistics prove the
//!   filter false, a [`FilterPlanner`] runs the filter plan over the rows that remain, and a
//!   [`ProjectionPlanner`] emits a [`ProjectionMorsel`] that runs the projection plan over the rows
//!   that survive. Each is a separate stage executing through the plan exec graph; [`plan_split`]
//!   composes them for one split.
//! - Over a [`LayoutReader`](crate::LayoutReader): a [`SplitMorsel`] polls the reader's filter and
//!   projection futures against a [`PollingSegmentSource`].
//!
//! Planners that decide which splits to read, and that know where each segment lives in a file,
//! belong to the crates that own those sources.

mod filter;
mod graph;
mod projection;
mod segments;
mod split_morsel;

use std::sync::Arc;

pub use filter::FilterPlanner;
pub use filter::FilterPlans;
pub use filter::SelectedRows;
pub use projection::ProjectionMorsel;
pub use projection::ProjectionPlanner;
pub use segments::PollingSegmentSource;
pub use segments::SegmentLocation;
pub use split_morsel::SplitMorsel;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::planning::next::Next;
use vortex_scan::planning::next::PendingPlanner;
use vortex_scan::planning::next::next_fn;
use vortex_scan::planning::next::pending;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_session::VortexSession;

use crate::plan::PlanRef;
use crate::plan::exec::DecodeCache;

/// What every split of a scan shares: the projection plan and where its segments live.
#[derive(Clone)]
pub struct ScanPlans {
    /// The session used for decoding and expression evaluation.
    pub session: VortexSession,
    /// The location of every segment, indexed by segment id.
    pub locations: Arc<[SegmentLocation]>,
    /// The optimized projection plan, over the scan's root row domain.
    pub projection: PlanRef,
    /// The global row index of the root row domain's first row.
    pub row_offset: u64,
    /// Segments already decoded by the graphs of one split. [`plan_split`] gives each split its
    /// own.
    pub decoded: DecodeCache,
}

/// The work for one split: prune the rows of `scope` selected by `mask` with zone statistics,
/// filter the rows that remain, then project the rows that survive. Each stage is skipped when
/// its plan is absent.
pub fn plan_split(
    plans: ScanPlans,
    pruning: Option<PlanRef>,
    filter: Option<FilterPlans>,
    scope: WorkScope,
    mask: Mask,
) -> VortexResult<Box<dyn PendingPlanner>> {
    // The split's pruning, filter, and projection graphs share one cache, so a segment several of
    // them read is fetched and decoded once.
    let plans = ScanPlans {
        decoded: DecodeCache::default(),
        ..plans
    };
    let project: Next<SelectedRows> = {
        let plans = plans.clone();
        next_fn(move |selected| Ok(ProjectionPlanner::new(plans.clone(), selected)))
    };
    let filter: Next<SelectedRows> = match filter {
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
    let Some(pruning) = pruning else {
        return filter(SelectedRows { scope, mask });
    };
    Ok(pending(move || {
        Ok(
            Box::new(FilterPlanner::pruning(plans, pruning, scope, mask, filter))
                as Box<dyn Planner>,
        )
    }))
}
