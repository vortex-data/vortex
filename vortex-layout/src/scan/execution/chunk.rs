// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cuts a scan's rows into chunks and plans each one.

use std::ops::Range;
use std::sync::Arc;
use std::vec::IntoIter;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::driver::RowClaim;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;

use super::PreparedScan;
use super::SegmentLocation;
use super::selection::plan_selection;

/// The row grid every chunk boundary lies on.
pub(crate) const ROW_BLOCK_SIZE: usize = 8192;
/// The rows of a projection chunk.
pub(crate) const MAX_ROWS_PER_CHUNK: usize = 8 * ROW_BLOCK_SIZE;
/// The rows of a filtered chunk, balancing conjunct setup against parallelism.
pub(crate) const FILTERED_ROWS_PER_CHUNK: usize = 16 * ROW_BLOCK_SIZE;
const TARGET_BYTES_PER_CHUNK: u64 = 8 * 1024 * 1024;

/// Limit estimated bytes per chunk without changing the writer-aligned row granularity.
pub(crate) fn rows_per_chunk(row_count: u64, locations: &[SegmentLocation]) -> usize {
    if row_count <= MAX_ROWS_PER_CHUNK as u64 {
        return MAX_ROWS_PER_CHUNK;
    }
    let bytes: u64 = locations
        .iter()
        .map(|location| u64::from(location.length))
        .sum();
    let bytes_per_row = bytes.div_ceil(row_count).max(1);
    let blocks = TARGET_BYTES_PER_CHUNK / bytes_per_row / ROW_BLOCK_SIZE as u64;
    blocks.clamp(1, 8) as usize * ROW_BLOCK_SIZE
}

/// Fixed boundaries in the file's row domain, clipped to the requested range.
pub(crate) fn row_chunks(rows: Range<u64>, rows_per_chunk: usize) -> Vec<Range<u64>> {
    if rows.is_empty() {
        return Vec::new();
    }
    let first = rows.start / rows_per_chunk as u64 * rows_per_chunk as u64;
    (first..rows.end)
        .step_by(rows_per_chunk)
        .map(|start| {
            start.max(rows.start)..start.saturating_add(rows_per_chunk as u64).min(rows.end)
        })
        .collect()
}

/// The chunks a scan of `rows` plans: [`row_chunks`], with a chunk at either edge of `rows` that
/// is under half of `rows_per_chunk` joined to its neighbour. An engine cuts a file into
/// partitions whose edges fall off the chunk grid, and a short edge chunk would cost a whole
/// chunk's planning and conjunct setup for a few rows.
pub(crate) fn scan_chunks(rows: Range<u64>, rows_per_chunk: usize) -> Vec<Range<u64>> {
    let mut chunks = row_chunks(rows, rows_per_chunk);
    let short = |chunk: &Range<u64>| chunk.end - chunk.start < rows_per_chunk as u64 / 2;
    if chunks.len() > 1 && chunks.last().is_some_and(short) {
        let last = chunks.pop().vortex_expect("more than one chunk");
        chunks.last_mut().vortex_expect("more than one chunk").end = last.end;
    }
    if chunks.len() > 1 && short(&chunks[0]) {
        let first = chunks.remove(0);
        chunks[0].start = first.start;
    }
    chunks
}

/// Emits one child planner per chunk of a scan's rows, in row order. A chunk none of whose
/// rows is selected and kept is never planned.
pub struct ChunkPlanner {
    scan: Arc<PreparedScan>,
    rows: Range<u64>,
    /// The suffix not yet handed to child planners, including unselected rows.
    claim: RowClaim,
    chunks: IntoIter<Range<u64>>,
    selection: Selection,
    /// The rows zone pruning kept, over `rows`, or `None` when nothing was pruned.
    kept: Option<Mask>,
    independent: bool,
}

impl ChunkPlanner {
    /// Plans chunks of the claimed rows, skipping rows excluded by `selection` or `kept`.
    /// The pruning mask `kept`, when present, covers the entire claim.
    pub fn new(
        scan: Arc<PreparedScan>,
        claim: RowClaim,
        selection: Selection,
        kept: Option<Mask>,
    ) -> Self {
        let chunks = scan_chunks(claim.rows().clone(), scan.chunk_rows);
        Self::with_chunks(scan, claim, chunks, selection, kept)
    }

    /// The planner of `chunks`, which tile the claim in row order, for a driver that cuts the rows
    /// itself, such as on the file's own chunk boundaries.
    pub fn with_chunks(
        scan: Arc<PreparedScan>,
        claim: RowClaim,
        chunks: Vec<Range<u64>>,
        selection: Selection,
        kept: Option<Mask>,
    ) -> Self {
        let rows = claim.rows().clone();
        debug_assert!(
            chunks.first().is_none_or(|chunk| chunk.start == rows.start)
                && chunks.last().is_none_or(|chunk| chunk.end == rows.end)
                && chunks.windows(2).all(|pair| pair[0].end == pair[1].start),
            "chunks must tile the rows in order"
        );
        Self {
            scan,
            rows,
            claim,
            chunks: chunks.into_iter(),
            selection,
            kept,
            independent: false,
        }
    }

    /// Emits independent chunks so the driver can schedule their compute and reads concurrently.
    /// Drivers that already assign work to their own workers keep the default child planners.
    pub fn with_independent_chunks(mut self) -> Self {
        self.independent = true;
        self
    }
}

impl IoConsumer for ChunkPlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for ChunkPlanner {
    fn state(&self) -> State {
        if self.chunks.as_slice().is_empty() {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let kept = self.kept.as_ref().map(|kept| (kept, self.rows.start));
        for chunk in self.chunks.by_ref() {
            let claim = self.claim.split_to(chunk.end);
            let scope = WorkScope {
                file_ordinal: 0,
                rows: chunk,
            };
            if let Some((scope, planner)) =
                plan_selection(&self.scan, scope, claim, &self.selection, kept)
            {
                return Ok(if self.independent {
                    PlannerOutput::Spawn(scope, planner)
                } else {
                    PlannerOutput::Planner(scope, planner)
                });
            }
        }
        Ok(PlannerOutput::Done)
    }
}
