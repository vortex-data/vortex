// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cuts a scan's rows into chunks and plans each one.

use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::vec::IntoIter;

use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_io::segments::IoKey;
use vortex_io::segments::SourceResult;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;

use super::PreparedScan;
use super::SegmentLocation;
use super::filter::segments_read;
use super::selection::plan_selection;
use super::selection::selected_rows;

/// The row grid every chunk boundary lies on.
pub(crate) const ROW_BLOCK_SIZE: usize = 8192;
/// The rows of a projection chunk.
pub(crate) const MAX_ROWS_PER_CHUNK: usize = 8 * ROW_BLOCK_SIZE;
/// The rows of a filtered chunk: large, so a conjunct's reads coalesce and its mask is cheap.
pub(crate) const FILTERED_ROWS_PER_CHUNK: usize = 16 * MAX_ROWS_PER_CHUNK;
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
    announced: bool,
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
            announced: false,
        }
    }

    /// Announces the filter's and the projection's segments over every chunk still to plan.
    fn announce(&self) -> VortexResult<()> {
        let kept = self.kept.as_ref().map(|kept| (kept, self.rows.start));
        let predicates = self
            .scan
            .filter
            .iter()
            .flat_map(|filter| filter.predicates());
        let plans: Vec<_> = predicates
            .chain(std::iter::once(&self.scan.projection))
            .collect();
        let mut segments = Vec::new();
        for chunk in self.chunks.as_slice() {
            let scope = WorkScope {
                file_ordinal: 0,
                rows: chunk.clone(),
            };
            let Some(selected) = selected_rows(scope, &self.selection, kept) else {
                continue;
            };
            segments.extend(segments_read(
                &self.scan.session,
                plans.iter().copied(),
                selected.scope.rows,
                &selected.mask,
            )?);
        }
        segments.sort_unstable();
        segments.dedup();
        self.scan.announce(&segments)
    }
}

/// Whether a scan announces every chunk's reads before planning the first. Setting
/// `VORTEX_SCAN_EARLY_ANNOUNCE=0` leaves each chunk to announce its own, for comparisons.
fn early_announce() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        !std::env::var("VORTEX_SCAN_EARLY_ANNOUNCE").is_ok_and(|value| value == "0")
    });
    *ENABLED
}

impl Planner for ChunkPlanner {
    fn deliver(&mut self, key: IoKey, _result: SourceResult) {
        vortex_panic!("ChunkPlanner: delivery of {key:?}, though it never fetches");
    }

    fn state(&self) -> State {
        if self.chunks.as_slice().is_empty() {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if !self.announced {
            self.announced = true;
            if early_announce() {
                self.announce()?;
            }
        }
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
