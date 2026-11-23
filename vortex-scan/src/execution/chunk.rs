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

use super::PreparedScan;
use super::SegmentLocation;
use super::selection::plan_selection;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;
use crate::selection::Selection;

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
            let has_selected = self.selection.row_count_in_range(&chunk) != 0;
            if has_selected && let Some(recorder) = &self.scan.recorder {
                recorder.attempt_file_scan();
            }
            let scope = WorkScope {
                file_ordinal: 0,
                rows: chunk,
            };
            if let Some((scope, planner)) = plan_selection(&self.scan, scope, &self.selection, kept)
            {
                return Ok(PlannerOutput::Spawn(scope, planner));
            }
            if has_selected
                && self.kept.is_some()
                && let Some(recorder) = &self.scan.recorder
            {
                recorder.attempt_split();
                recorder.prune_split();
            }
        }
        Ok(PlannerOutput::Done)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use super::ChunkPlanner;
    use crate::execution::pipeline::synthetic::RowSource;
    use crate::metrics::ScanUsage;
    use crate::plan::LazyPlanRef;
    use crate::planning::planner::Planner;
    use crate::planning::planner::PlannerOutput;
    use crate::prepare::ScanBuilder;
    use crate::selection::Selection;
    use crate::source::ScanInput;
    use crate::source::SegmentFuture;
    use crate::source::SegmentId;
    use crate::source::SegmentSource;

    struct NoSegments;

    impl SegmentSource for NoSegments {
        fn request(&self, _id: SegmentId) -> SegmentFuture {
            Box::pin(async { vortex_bail!("in-memory test source must not request segments") })
        }
    }

    #[test]
    fn pruned_chunks_record_split_pruning_without_file_pruning() -> VortexResult<()> {
        let usage = Arc::new(ScanUsage::default());
        let plan = RowSource::plan(12, vec![4, 4, 4], None, false);
        let mut prepared = ScanBuilder::new(
            VortexSession::empty(),
            ScanInput {
                dtype: plan.dtype().clone(),
                root: LazyPlanRef::ready(plan),
                locations: Arc::new([]),
                segments: Arc::new(NoSegments),
                cache: None,
                usage: Some(Arc::clone(&usage)),
            },
        )
        .prepare()?;
        prepared.chunk_rows = 4;

        // Three 4-row chunks: chunks 0 (0..4) and 2 (8..12) are fully pruned, chunk 1 (4..8) is kept.
        let kept = Mask::from_iter([
            false, false, false, false, true, true, true, true, false, false, false, false,
        ]);
        let mut planner = ChunkPlanner::new(Arc::new(prepared), 0..12, Selection::All, Some(kept));

        let mut spawned_ranges = Vec::new();
        loop {
            match planner.compute()? {
                PlannerOutput::Spawn(scope, _) => spawned_ranges.push(scope.rows),
                PlannerOutput::Done => break,
                other => vortex_bail!("unexpected planner output: {other:?}"),
            }
        }

        assert_eq!(spawned_ranges, vec![4..8]);
        let snap = usage.snapshot();
        assert_eq!(snap.file_scans_attempted, 1);
        assert_eq!(snap.file_scans_pruned, 0);
        assert_eq!(snap.splits_attempted, 2);
        assert_eq!(snap.splits_pruned, 2);
        Ok(())
    }
}
