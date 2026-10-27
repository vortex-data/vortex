// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The planners of a V2 scan, and the prepared plans they run.
//!
//! A [`PreparedScan`] holds a scan's plans over one file. An engine composes the planners over it
//! and drives them: a [`PrunePlanner`] evaluates the filter's pruning plans over the scan's rows and
//! emits the planner it was built to continue into, with the rows they keep; a [`ChunkPlanner`]
//! cuts those rows into chunks and spawns a [`FilterPlanner`] or a [`ProjectionPlanner`] per
//! chunk; a filter planner runs its chunk's conjuncts and emits a projection planner for the rows
//! that survive; a projection planner emits one morsel per piece of its rows. A planner only
//! says what it needs, compute or reads; where either runs is the engine's driver's decision.

mod adaptor;
mod chunk;
mod filter;
mod projection;
mod prune;
mod selection;

use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_io::segments::IoHandle;
use vortex_io::segments::IoIntent;
use vortex_io::segments::IoKey;
use vortex_io::segments::IoOwnerId;
use vortex_io::segments::IoRequest;
use vortex_io::segments::RegisteredSource;
use vortex_io::segments::SegmentIoService;
use vortex_io::segments::SegmentSourceId;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

pub use self::chunk::ChunkPlanner;
pub(crate) use self::chunk::FILTERED_ROWS_PER_CHUNK;
pub(crate) use self::chunk::MAX_ROWS_PER_CHUNK;
pub(crate) use self::chunk::ROW_BLOCK_SIZE;
#[cfg(test)]
pub(crate) use self::chunk::row_chunks;
pub(crate) use self::chunk::rows_per_chunk;
use self::filter::FilterPlanner;
pub use self::filter::FilterPlans;
use self::projection::ProjectionPlanner;
pub use self::prune::PruneNext;
pub use self::prune::PrunePlanner;
use crate::plan::physical::PlanRef;
use crate::plan::physical::pipeline::ValueCache;
use crate::scan::v2::ScanCache;
use crate::segments::SegmentId;

/// The owner a scan's own announcements are submitted as. Nothing waits on them.
const SCAN_OWNER: IoOwnerId = IoOwnerId(u64::MAX);

/// Where a segment's bytes live in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentLocation {
    /// Byte offset of the segment from the start of the source.
    pub offset: u64,
    /// Length of the segment in bytes.
    pub length: u32,
    /// Alignment the segment's bytes must have once delivered.
    pub alignment: Alignment,
}

/// A scan's plans over one file, built once from the file's footer and read by every planner of
/// the scan. Nothing here touches a segment, and no planner plans again.
pub struct PreparedScan {
    /// The session used for decoding and expression evaluation.
    pub(super) session: VortexSession,
    /// The rows of the root row domain the scan covers.
    pub(super) rows: Range<u64>,
    /// The rows of `rows` the scan selects.
    pub(super) selection: Selection,
    /// The file's registration with the segment IO service that serves the scan's reads, held
    /// for the scan's life.
    pub(super) segment_io: RegisteredSource,
    /// Held for the scan's life, so scans of the file that start meanwhile share it.
    #[expect(dead_code, reason = "kept alive, never read")]
    pub(super) cache: ScanCache,
    /// The location of every segment, indexed by segment id.
    pub(super) locations: Arc<[SegmentLocation]>,
    /// The optimized projection plan, over the scan's root row domain.
    pub(super) projection: PlanRef,
    /// The filter's conjuncts, their plans and their pruning plans.
    pub(super) filter: Option<Arc<FilterPlans>>,
    /// The global row index of the root row domain's first row.
    pub(super) row_offset: u64,
    /// Whole values, such as dictionary values, shared by the chunks of this scan. Their plans
    /// always read the same complete row domain, independently of a chunk's selection.
    pub(super) values: ValueCache,
    /// The rows of one chunk, the unit of work a chunk planner emits.
    pub(super) chunk_rows: usize,
    /// Announcements of the scan's reads, held while any planner of the scan lives.
    pub(super) announcements: Mutex<Vec<IoHandle>>,
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

    /// The service a driver takes IO sessions from: every read a planner of the scan requests is
    /// served from it by segment.
    pub fn io(&self) -> Arc<dyn SegmentIoService> {
        Arc::clone(self.segment_io.service())
    }

    /// The file's id in [`io`](Self::io), which the scan's requests name.
    pub(super) fn source(&self) -> SegmentSourceId {
        self.segment_io.id()
    }

    /// Announces `segments` for the rest of the scan, so the IO service can widen the reads of
    /// every chunk with them. The announcements read nothing, and are held while the scan lives:
    /// each chunk runs on a driver of its own, and announcing per chunk would let one chunk's
    /// read coalesce only with its own segments.
    pub(super) fn announce(&self, segments: &[SegmentId]) -> VortexResult<()> {
        if segments.is_empty() {
            return Ok(());
        }
        let batch: Vec<_> = segments
            .iter()
            .map(|segment| {
                IoRequest::new(IoKey::segment(self.source(), **segment), IoIntent::Announce)
            })
            .collect();
        let handles = self.io().session().submit(SCAN_OWNER, &batch)?;
        self.announcements.lock().extend(handles);
        Ok(())
    }

    /// The planner of one chunk's selected rows: its conjuncts when the scan has a filter, else
    /// its projection.
    pub(super) fn split_planner(self: &Arc<Self>, selected: SelectedRows) -> Box<dyn Planner> {
        if self.filter.is_some() {
            Box::new(FilterPlanner::new(Arc::clone(self), selected))
        } else {
            Box::new(ProjectionPlanner::new(Arc::clone(self), selected))
        }
    }
}

/// The rows a planner works on: a chunk, and which of its rows are selected.
pub(super) struct SelectedRows {
    /// The chunk the rows belong to.
    scope: WorkScope,
    /// Which rows of `scope.rows` are selected.
    mask: Mask,
}
