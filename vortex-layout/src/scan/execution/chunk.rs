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
use super::PrunePlanner;
use super::SegmentLocation;
use super::selection::plan_selection;
use crate::plan::physical::PlanRef;

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

/// Emits one planner per chunk of a scan's rows, in row order, each spawned as independent so
/// the driver may run it beside the others. A chunk none of whose rows is selected and kept is
/// never planned.
///
/// A dynamic conjunct, such as a TopK's bound, may tighten while the chunks run. A chunk planned
/// after it has changed prunes its rows with the conjunct's zone maps again first, so a chunk the
/// new value excludes reads none of its conjuncts' or projection's segments.
pub struct ChunkPlanner {
    scan: Arc<PreparedScan>,
    rows: Range<u64>,
    chunks: IntoIter<Range<u64>>,
    selection: Selection,
    /// The rows zone pruning kept, over `rows`, or `None` when nothing was pruned.
    kept: Option<Mask>,
    /// The dynamic versions `kept` was pruned at, or `None` when chunks are not pruned again.
    pruned_at: Option<Vec<Option<u64>>>,
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
        let pruned_at = scan
            .filter
            .as_ref()
            .map(|filter| filter.pruning_versions())
            .filter(|versions| versions.iter().any(Option::is_some));
        Self {
            scan,
            rows,
            chunks,
            selection,
            kept,
            pruned_at,
        }
    }

    /// The planner of `chunk` that prunes it with `pruning` before planning what is left.
    /// `None` when zone pruning already excluded every row of the chunk.
    fn prune_again(&self, chunk: Range<u64>, pruning: Vec<PlanRef>) -> Option<Box<dyn Planner>> {
        let len = usize::try_from(chunk.end - chunk.start).vortex_expect("a chunk fits usize");
        let kept = match &self.kept {
            Some(kept) => {
                let offset = usize::try_from(chunk.start - self.rows.start)
                    .vortex_expect("a chunk fits usize");
                kept.slice(offset..offset + len)
            }
            None => Mask::new_true(len),
        };
        if kept.all_false() {
            return None;
        }
        let next = {
            let scan = Arc::clone(&self.scan);
            let selection = self.selection.clone();
            let rows = chunk.clone();
            move |kept: Option<Mask>| -> Box<dyn Planner> {
                Box::new(Self {
                    scan,
                    chunks: vec![rows.clone()].into_iter(),
                    rows,
                    selection,
                    kept,
                    pruned_at: None,
                })
            }
        };
        Some(Box::new(PrunePlanner::again(
            Arc::clone(&self.scan),
            chunk,
            pruning,
            kept,
            Box::new(next),
        )))
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
        while let Some(chunk) = self.chunks.next() {
            let pruning = match (&self.pruned_at, &self.scan.filter) {
                (Some(versions), Some(filter)) => filter.pruning_since(versions),
                _ => Vec::new(),
            };
            if !pruning.is_empty() {
                let scope = WorkScope {
                    file_ordinal: 0,
                    rows: chunk.clone(),
                };
                if let Some(planner) = self.prune_again(chunk, pruning) {
                    return Ok(PlannerOutput::Spawn(scope, planner));
                }
                continue;
            }
            let kept = self.kept.as_ref().map(|kept| (kept, self.rows.start));
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
