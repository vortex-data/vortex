// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The planners of a V2 scan, and the prepared plans they run.
//!
//! A [`PreparedScan`] holds a scan's plans over one file and makes its planners. The planners
//! form a chain, each emitting the next through the planning protocol: a [`PrunePlanner`]
//! proves the filter's zone maps over the scan's rows and emits one [`ChunkPlanner`] for the rows
//! they keep; the chunk planner cuts those rows into chunks and emits a [`FilterPlanner`] or a
//! [`ProjectionPlanner`] per chunk; a filter planner runs its chunk's conjuncts and emits a
//! projection planner for the rows that survive; a projection planner emits one morsel per piece
//! of its rows. A stage with nothing to do is never made.

mod adaptor;
mod chunk;
mod filter;
mod projection;
mod prune;
mod selection;

use std::ops::Range;
use std::sync::Arc;

use vortex_buffer::Alignment;
use vortex_io::request::IoTarget;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

pub(in crate::scan) use self::chunk::ChunkPlanner;
pub(crate) use self::chunk::MAX_ROWS_PER_CHUNK;
pub(crate) use self::chunk::ROW_BLOCK_SIZE;
pub(crate) use self::chunk::filtered_rows_per_chunk;
#[cfg(test)]
pub(crate) use self::chunk::row_chunks;
pub(crate) use self::chunk::rows_per_chunk;
pub(crate) use self::chunk::total_bytes;
use self::filter::FilterPlanner;
pub(super) use self::filter::FilterPlans;
use self::projection::ProjectionPlanner;
use self::prune::PrunePlanner;
use crate::plan::physical::PlanRef;
use crate::plan::physical::pipeline::ValueCache;

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

/// A scan's plans over one file, built once from the file's footer and read by every planner of
/// the scan. Nothing here touches a segment, and no planner plans again.
pub(crate) struct PreparedScan {
    /// The session used for decoding and expression evaluation.
    pub(super) session: VortexSession,
    /// The location of every segment, indexed by segment id.
    pub(super) locations: Arc<[SegmentLocation]>,
    /// The optimized projection plan, over the scan's root row domain.
    pub(super) projection: PlanRef,
    /// The filter's conjuncts, their plans and their proofs.
    pub(super) filter: Option<Arc<FilterPlans>>,
    /// The global row index of the root row domain's first row.
    pub(super) row_offset: u64,
    /// Whole values, such as dictionary values, shared by the chunks of this scan. Their plans
    /// always read the same complete row domain, independently of a chunk's selection.
    pub(super) values: ValueCache,
    /// The rows of one chunk, the unit of work a chunk planner emits.
    pub(super) chunk_rows: usize,
    /// Optional full-file boundaries supplied by the integration.
    pub(super) natural_splits: Option<Arc<[u64]>>,
}

impl PreparedScan {
    /// The first planner a scan of `rows` needs. Every later planner is emitted by the one before
    /// it: the zone proofs run first when the filter has any, then one planner per chunk of the
    /// rows they keep.
    pub(crate) fn planner(
        self: &Arc<Self>,
        rows: Range<u64>,
        selection: Selection,
    ) -> Box<dyn Planner> {
        let proofs = self
            .filter
            .as_ref()
            .is_some_and(|filter| filter.proofs().next().is_some());
        if proofs && !rows.is_empty() {
            let scan = Arc::clone(self);
            let chunks = rows.clone();
            Box::new(PrunePlanner::new(
                Arc::clone(self),
                rows,
                Box::new(move |kept| Box::new(scan.chunk_planner(chunks, selection, Some(kept)))),
            ))
        } else {
            Box::new(self.chunk_planner(rows, selection, None))
        }
    }

    /// The planner cutting `rows` into chunks, planning only the chunks `selection` and `kept`
    /// leave something of.
    pub(super) fn chunk_planner(
        self: &Arc<Self>,
        rows: Range<u64>,
        selection: Selection,
        kept: Option<Mask>,
    ) -> ChunkPlanner {
        ChunkPlanner::new(Arc::clone(self), rows, selection, kept)
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
