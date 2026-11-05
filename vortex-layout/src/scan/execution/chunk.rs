// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cuts a scan's rows into chunks and plans each one.

use std::ops::Range;
use std::sync::Arc;
use std::vec::IntoIter;

use rustc_hash::FxHashSet;
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
use crate::plan::physical::PlanByPtr;
use crate::plan::physical::PlanRef;
use crate::plan::physical::SegmentScan;

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

/// The fewest rows of a chunk sized by [`rows_for_bytes`].
pub(crate) const MIN_ROWS_PER_CHUNK: usize = ROW_BLOCK_SIZE / 8;
/// The fewest chunks [`rows_for_bytes`] cuts a scan of rows wider than a row block's bytes
/// into, so a scan that is one of an engine's partitions keeps several chunks in flight.
const MIN_CHUNKS_PER_SCAN: u64 = 8;

/// The rows of a chunk of a scan of `scan_rows` rows that reads `bytes` over a file of
/// `row_count` rows: whole row blocks of about [`TARGET_BYTES_PER_CHUNK`], at most `max_rows`.
///
/// Rows so wide that a row block holds more than the target, such as long lists, are cut finer:
/// into whole multiples of [`MIN_ROWS_PER_CHUNK`], and into at least [`MIN_CHUNKS_PER_SCAN`]
/// chunks of the scan.
pub(crate) fn rows_for_bytes(row_count: u64, scan_rows: u64, bytes: u64, max_rows: usize) -> usize {
    if row_count == 0 {
        return max_rows;
    }
    let block = ROW_BLOCK_SIZE as u64;
    let rows = TARGET_BYTES_PER_CHUNK / bytes.div_ceil(row_count).max(1);
    if rows >= block {
        return usize::try_from(rows / block * block)
            .unwrap_or(usize::MAX)
            .min(max_rows);
    }
    let min = MIN_ROWS_PER_CHUNK as u64;
    let rows = rows.min(scan_rows / MIN_CHUNKS_PER_SCAN);
    usize::try_from((rows / min * min).max(min))
        .unwrap_or(usize::MAX)
        .min(max_rows)
}

/// The stored bytes of every segment `plans` read, each once.
#[expect(
    clippy::mutable_key_type,
    reason = "plans hash by address, which building their lazy children never changes"
)]
pub(crate) fn plan_bytes<'a>(
    plans: impl Iterator<Item = &'a PlanRef>,
    locations: &[SegmentLocation],
) -> VortexResult<u64> {
    let mut visited = FxHashSet::default();
    let mut segments = FxHashSet::default();
    let mut pending: Vec<PlanRef> = plans.cloned().collect();
    let mut bytes = 0;
    while let Some(plan) = pending.pop() {
        if !visited.insert(PlanByPtr(plan.clone())) {
            continue;
        }
        if let Some(scan) = plan.as_opt::<SegmentScan>()
            && segments.insert(scan.segment_id())
        {
            bytes += locations
                .get(*scan.segment_id() as usize)
                .map_or(0, |location| u64::from(location.length));
        }
        for child in plan.children() {
            pending.push(child?.clone());
        }
    }
    Ok(bytes)
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
        let chunks = scan_chunks(rows.clone(), scan.chunk_rows).into_iter();
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
