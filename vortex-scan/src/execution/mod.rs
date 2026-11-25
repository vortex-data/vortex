// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The planners of a V2 scan, and the prepared plans they run.
//!
//! A [`PreparedScan`] holds a scan's plans over one file. An engine composes the planners over it
//! and drives them: a [`PrunePlanner`] evaluates the filter's pruning plans over the scan's rows and
//! emits the planner it was built to continue into, with the rows they keep; a [`ChunkPlanner`]
//! cuts those rows into chunks and spawns a `FilterPlanner` or a `ProjectionPlanner` per
//! chunk; a filter planner runs its chunk's conjuncts and emits a projection planner for the rows
//! that survive; a projection planner emits one morsel per piece of its rows. A planner only
//! says what it needs, compute or reads; where either runs is the engine's driver's decision.

pub mod cache;
pub mod driver;
pub mod filter_expr;
pub(crate) mod io;
pub mod operator;
pub(crate) mod operators;
pub mod pipeline;
pub mod planning;
mod retain;
pub(crate) mod zone_map;

mod adaptor;
mod chunk;
mod filter;
mod parts;
mod projection;
mod prune;
mod selection;

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_io::request::IoService;
use vortex_io::request::IoTarget;
pub use vortex_io::segments::SegmentLocation;
use vortex_mask::Mask;
use vortex_session::VortexSession;

pub use self::chunk::ChunkPlanner;
pub(crate) use self::chunk::FILTERED_ROWS_PER_CHUNK;
#[cfg(test)]
pub(crate) use self::chunk::MAX_ROWS_PER_CHUNK;
pub(crate) use self::chunk::chunk_rows;
#[cfg(test)]
pub(crate) use self::chunk::row_chunks;
pub(crate) use self::chunk::rows_per_chunk;
#[cfg(test)]
pub(crate) use self::chunk::scan_chunks;
use self::filter::FilterPlanner;
pub use self::filter::FilterPlans;
pub use self::filter::FilterPrep;
pub use self::filter::SharedFilterPrep;
pub(crate) use self::parts::part_planner;
pub(crate) use self::parts::static_kept;
use self::projection::ProjectionPlanner;
pub use self::prune::PruneNext;
pub use self::prune::PrunePlanner;
use crate::execution::cache::ScanCache;
use crate::execution::io::SegmentRanges;
use crate::execution::io::SegmentScanIo;
use crate::execution::pipeline::ValueCache;
use crate::plan::PlanRef;
use crate::planning::planner::Planner;
use crate::planning::planner::WorkScope;
use crate::selection::Selection;
use crate::source::SegmentSource;

/// The byte range a read of the segment at `location` asks for.
fn segment_target(location: &SegmentLocation) -> IoTarget {
    IoTarget::Range {
        offset: location.offset,
        len: location.length as usize,
        alignment: location.alignment,
    }
}

/// A scan's plans over one file, built once from the file's footer and read by every planner of
/// the scan. Nothing here touches a segment, and no planner plans again.
pub struct PreparedScan {
    /// The session used for decoding and expression evaluation.
    pub(crate) session: VortexSession,
    /// The rows of the root row domain the scan covers.
    pub(crate) rows: Range<u64>,
    /// The rows of `rows` the scan selects.
    pub(crate) selection: Selection,
    /// Fetches segment bytes by id.
    pub(crate) segments: Arc<dyn SegmentSource>,
    /// The segments' byte ranges, back to their ids.
    pub(crate) ranges: SegmentRanges,
    /// Held for the scan's life, so scans of the file that start meanwhile share it.
    #[expect(dead_code, reason = "kept alive, never read")]
    pub(crate) cache: ScanCache,
    /// The location of every segment, indexed by segment id.
    pub(crate) locations: Arc<[SegmentLocation]>,
    /// The optimized projection plan, over the scan's root row domain.
    pub(crate) projection: PlanRef,
    /// The filter's conjuncts, their plans and their pruning plans.
    pub(crate) filter: Option<Arc<FilterPlans>>,
    /// The global row index of the root row domain's first row.
    pub(crate) row_offset: u64,
    /// Whole values, such as dictionary values, shared by the chunks of this scan. Their plans
    /// always read the same complete row domain, independently of a chunk's selection.
    pub(crate) values: ValueCache,
    /// The rows of one chunk, the unit of work a chunk planner emits.
    pub(crate) chunk_rows: usize,
}

impl PreparedScan {
    /// The rows of the file's row domain the scan covers.
    pub fn rows(&self) -> &Range<u64> {
        &self.rows
    }

    /// The rows of [`rows`](Self::rows) the scan selects.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    /// The filter's plans, when the scan has a filter.
    pub fn filter(&self) -> Option<&FilterPlans> {
        self.filter.as_deref()
    }

    /// The points at which the scan's rows are cut into parts, into at least `min_parts` where
    /// they hold enough blocks: [`split::split_points`](crate::split::split_points) over the
    /// chunk boundaries of the columns the scan reads, each part measured by its rows in `kept`,
    /// the rows of the scan its static conjuncts keep. Reads nothing.
    pub fn part_points(&self, min_parts: usize, kept: Option<&Mask>) -> VortexResult<Vec<u64>> {
        let surviving = kept.map(|kept| crate::split::kept_ranges(&self.rows, kept));
        Ok(crate::split::split_points(
            self.rows.clone(),
            &parts::chunk_boundaries(self)?,
            surviving.as_deref(),
            min_parts,
        ))
    }

    /// The rows of the scan its filter's static conjuncts keep, by the zones their pruning plans
    /// keep, or `None` when no static conjunct has a pruning plan. Reads the zone maps.
    pub async fn static_kept(self: &Arc<Self>) -> VortexResult<Option<Mask>> {
        static_kept(self, &self.io()).await
    }

    /// The service a driver takes IO sessions from: every read a planner of the scan requests is
    /// served from it by segment.
    pub fn io(&self) -> Arc<dyn IoService> {
        Arc::new(SegmentScanIo::new(
            Arc::clone(&self.segments),
            Arc::clone(&self.ranges),
        ))
    }

    /// The planner of one chunk's selected rows: its conjuncts when the scan has a filter, else
    /// its projection.
    fn split_planner(self: &Arc<Self>, selected: SelectedRows) -> Box<dyn Planner> {
        if self.filter.is_some() {
            Box::new(FilterPlanner::new(Arc::clone(self), selected))
        } else {
            Box::new(ProjectionPlanner::new(Arc::clone(self), selected))
        }
    }
}

/// The rows a planner works on: a chunk, and which of its rows are selected.
struct SelectedRows {
    /// The chunk the rows belong to.
    scope: WorkScope,
    /// Which rows of `scope.rows` are selected.
    mask: Mask,
}

#[cfg(test)]
mod tests;
