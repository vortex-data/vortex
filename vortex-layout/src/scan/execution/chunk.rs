// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cuts a scan's rows into chunks and plans each one.

use std::ops::Range;
use std::sync::Arc;
use std::vec::IntoIter;

use vortex_array::dtype::DType;
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
use crate::plan::physical::Concat;
use crate::plan::physical::PlanRef;

/// The row grid chunk boundaries lie on, except where a stored chunk of a string column ends.
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

/// The chunks a scan of `rows` plans when `boundaries`, sorted root rows, are where stored chunks
/// of its view-typed columns end: [`scan_chunks`] between consecutive kept boundaries. A work
/// chunk that crosses a stored chunk of a string column returns it as several data buffers, which
/// every Arrow kernel downstream then carries, so a boundary is kept whenever both pieces it
/// leaves have at least [`ROW_BLOCK_SIZE`] rows. Shorter stored chunks share a work chunk.
pub(crate) fn scan_chunks_at(
    rows: Range<u64>,
    rows_per_chunk: usize,
    boundaries: &[u64],
) -> Vec<Range<u64>> {
    let min = ROW_BLOCK_SIZE as u64;
    let first = boundaries.partition_point(|&boundary| boundary <= rows.start);
    let mut chunks = Vec::new();
    let mut start = rows.start;
    for &boundary in &boundaries[first..] {
        if boundary.saturating_add(min) > rows.end {
            break;
        }
        if boundary - start >= min {
            chunks.extend(scan_chunks(start..boundary, rows_per_chunk));
            start = boundary;
        }
    }
    chunks.extend(scan_chunks(start..rows.end, rows_per_chunk));
    chunks
}

/// The sorted root rows where a stored chunk of a view-typed column of `plan` ends, found through
/// every child that keeps its parent's row domain.
pub(crate) fn view_chunk_boundaries(plan: &PlanRef) -> VortexResult<Vec<u64>> {
    fn visit(plan: &PlanRef, boundaries: &mut Vec<u64>) -> VortexResult<()> {
        if let Some(concat) = plan.as_opt::<Concat>() {
            if has_views(concat.dtype()) {
                boundaries.extend(concat.row_offsets().iter().skip(1));
            }
            // Visiting a chunk would lower it, for boundaries this scan rarely needs.
            return Ok(());
        }
        for child in plan.children().iter_refs() {
            let child = child?;
            if child.row_count() == plan.row_count() {
                visit(child, boundaries)?;
            }
        }
        Ok(())
    }

    let mut boundaries = Vec::new();
    visit(plan, &mut boundaries)?;
    boundaries.sort_unstable();
    boundaries.dedup();
    Ok(boundaries)
}

/// Whether an Arrow array of `dtype` holds string or binary views.
fn has_views(dtype: &DType) -> bool {
    match dtype {
        DType::Utf8(_) | DType::Binary(_) => true,
        DType::Struct(fields, _) => fields.fields().any(|field| has_views(&field)),
        DType::List(element, _) | DType::FixedSizeList(element, ..) => has_views(element),
        _ => false,
    }
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
        let chunks =
            scan_chunks_at(rows.clone(), scan.chunk_rows, &scan.chunk_boundaries).into_iter();
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
