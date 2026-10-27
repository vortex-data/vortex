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
pub(crate) const MAX_ROWS_PER_CHUNK: usize = 32 * ROW_BLOCK_SIZE;
/// The rows of a filtered chunk: large, so a conjunct's reads coalesce and its mask is cheap.
pub(crate) const FILTERED_ROWS_PER_CHUNK: usize = 128 * ROW_BLOCK_SIZE;
// Filtered scans retain smaller work items for selective conjuncts.
const MIN_FILTERED_ROWS_PER_CHUNK: usize = 8 * ROW_BLOCK_SIZE;
const TARGET_BYTES_PER_CHUNK: u64 = 8 * 1024 * 1024;

/// Limit estimated bytes per chunk without changing the writer-aligned row granularity.
pub(crate) fn rows_per_chunk(row_count: u64, locations: &[SegmentLocation]) -> usize {
    if row_count <= MAX_ROWS_PER_CHUNK as u64 {
        return MAX_ROWS_PER_CHUNK;
    }
    let blocks = blocks_per_target_bytes(total_bytes(locations), row_count);
    blocks.clamp(1, (MAX_ROWS_PER_CHUNK / ROW_BLOCK_SIZE) as u64) as usize * ROW_BLOCK_SIZE
}

/// Rows per chunk of a filtered scan over `rows` rows of the file.
///
/// Large chunks amortize filter evaluation over cheap projections. A projection that reads many
/// bytes per row, or a row range that would otherwise give fewer chunks than threads, gets smaller
/// chunks so the surviving rows still spread across the runtime's threads.
pub(crate) fn filtered_rows_per_chunk(
    rows: u64,
    row_count: u64,
    projected_bytes: u64,
    parallelism: usize,
) -> usize {
    let max_blocks = (FILTERED_ROWS_PER_CHUNK / ROW_BLOCK_SIZE) as u64;
    let min_blocks = (MIN_FILTERED_ROWS_PER_CHUNK / ROW_BLOCK_SIZE) as u64;
    let by_bytes = blocks_per_target_bytes(projected_bytes, row_count).max(1);
    let by_threads = (rows / parallelism.max(1) as u64 / ROW_BLOCK_SIZE as u64).max(min_blocks);
    let blocks = by_bytes.min(by_threads).min(max_blocks);
    usize::try_from(blocks).vortex_expect("at most FILTERED_ROWS_PER_CHUNK blocks") * ROW_BLOCK_SIZE
}

/// The row blocks that hold [`TARGET_BYTES_PER_CHUNK`] when `bytes` spread over `row_count` rows.
fn blocks_per_target_bytes(bytes: u64, row_count: u64) -> u64 {
    let bytes_per_row = bytes.div_ceil(row_count.max(1)).max(1);
    TARGET_BYTES_PER_CHUNK / bytes_per_row / ROW_BLOCK_SIZE as u64
}

/// The serialized bytes of every segment in `locations`.
pub(crate) fn total_bytes(locations: &[SegmentLocation]) -> u64 {
    locations
        .iter()
        .map(|location| u64::from(location.length))
        .sum()
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

/// Emits one planner per chunk of a scan's rows, in row order, each spawned as independent so
/// the driver may run it beside the others. A chunk none of whose rows is selected and kept is
/// never planned.
pub struct ChunkPlanner {
    scan: Arc<PreparedScan>,
    rows: Range<u64>,
    chunks: IntoIter<Range<u64>>,
    selection: Selection,
    /// The rows zone pruning kept, over `rows`, or `None` when nothing was pruned.
    kept: Option<Mask>,
}

impl ChunkPlanner {
    /// The planner of the chunks of `rows`, planning only what `selection` and `kept`, the rows
    /// zone pruning left over `rows`, leave something of.
    pub fn new(
        scan: Arc<PreparedScan>,
        rows: Range<u64>,
        selection: Selection,
        kept: Option<Mask>,
    ) -> Self {
        let chunks = row_chunks(rows.clone(), scan.chunk_rows).into_iter();
        Self {
            scan,
            rows,
            chunks,
            selection,
            kept,
        }
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
            let scope = WorkScope {
                file_ordinal: 0,
                rows: chunk,
            };
            if let Some((scope, planner)) = plan_selection(&self.scan, scope, &self.selection, kept)
            {
                return Ok(PlannerOutput::Spawn(scope, planner));
            }
        }
        Ok(PlannerOutput::Done)
    }
}
