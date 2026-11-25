// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cutting a prepared scan into parts, the units a host's threads take.

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;

use super::ChunkPlanner;
use super::PreparedScan;
use super::PrunePlanner;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;
use crate::selection::Selection;

/// The root planner of a scan cut into parts on `splits`, such as a file's layout split
/// boundaries: it spawns one planner per part, in row order. With a dynamic filter, its pruning
/// plans first over the scan's rows, continuing into the parts of the rows they keep; otherwise,
/// the parts directly.
pub fn part_root(scan: &Arc<PreparedScan>, splits: &[u64], dynamic: bool) -> Box<dyn Planner> {
    let rows = scan.rows().clone();
    let parts = {
        let scan = Arc::clone(scan);
        let parts = part_ranges(rows.clone(), splits);
        let first_row = rows.start;
        move |kept: Option<Mask>| -> Box<dyn Planner> {
            Box::new(PartPlanner::new(scan, parts, first_row, kept))
        }
    };
    if dynamic && scan.filter().is_some() {
        Box::new(PrunePlanner::new(Arc::clone(scan), rows, Box::new(parts)))
    } else {
        parts(None)
    }
}

/// Spawns a planner per part of a scan's rows that zone pruning left something of, in row order.
/// Parts are the file's layout splits, so a layout chunk, and the dictionary it carries, is
/// read by one thread, which lets a host evaluate an expression over that dictionary once.
///
/// With a filter, each part prunes over its own rows before its chunks, on the thread that takes
/// it, like the per-split pruning of a V1 scan, so the pruning of a file's parts runs in parallel.
/// A dynamic filter, such as the bound a top-N pushes down, also prunes the scan as a whole before
/// any part is taken, so a file whose zones it rules out takes no thread. It has tightened by the
/// time a later part is taken, so pruning that part again skips more, from the zone maps the
/// scan-wide pruning read.
struct PartPlanner {
    scan: Arc<PreparedScan>,
    selection: Selection,
    parts: std::vec::IntoIter<Range<u64>>,
    /// The first row of `kept`.
    first_row: u64,
    /// The rows scan-wide pruning kept, from `first_row`, or `None` when nothing was pruned.
    kept: Option<Mask>,
}

impl PartPlanner {
    fn new(
        scan: Arc<PreparedScan>,
        parts: Vec<Range<u64>>,
        first_row: u64,
        kept: Option<Mask>,
    ) -> Self {
        Self {
            selection: scan.selection().clone(),
            scan,
            parts: parts.into_iter(),
            first_row,
            kept,
        }
    }

    /// The rows of `part` that scan-wide pruning kept, or `None` when nothing was pruned.
    fn kept_in(&self, part: &Range<u64>) -> Option<Mask> {
        let kept = self.kept.as_ref()?;
        let offset =
            |row: u64| usize::try_from(row - self.first_row).vortex_expect("a part fits usize");
        Some(kept.slice(offset(part.start)..offset(part.end)))
    }
}

impl IoConsumer for PartPlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for PartPlanner {
    fn state(&self) -> State {
        if self.parts.as_slice().is_empty() {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        while let Some(part) = self.parts.next() {
            let kept = self.kept_in(&part);
            if kept.as_ref().is_some_and(Mask::all_false) {
                continue;
            }
            let chunks = {
                let scan = Arc::clone(&self.scan);
                let selection = self.selection.clone();
                let part = part.clone();
                move |again: Option<Mask>| -> Box<dyn Planner> {
                    let kept = match (kept, again) {
                        (Some(kept), Some(again)) => Some(kept & &again),
                        (kept, again) => kept.or(again),
                    };
                    Box::new(ChunkPlanner::new(scan, part, selection, kept))
                }
            };
            let planner: Box<dyn Planner> = if self.scan.filter().is_some() {
                Box::new(PrunePlanner::new(
                    Arc::clone(&self.scan),
                    part.clone(),
                    Box::new(chunks),
                ))
            } else {
                chunks(None)
            };
            let scope = WorkScope {
                file_ordinal: 0,
                rows: part,
            };
            return Ok(PlannerOutput::Spawn(scope, planner));
        }
        Ok(PlannerOutput::Done)
    }
}

/// The ranges between adjacent `splits`, clipped to `rows`, leaving out empty ones.
fn part_ranges(rows: Range<u64>, splits: &[u64]) -> Vec<Range<u64>> {
    splits
        .windows(2)
        .map(|w| w[0].max(rows.start)..w[1].min(rows.end))
        .filter(|part| part.start < part.end)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;

    use super::part_ranges;

    #[rstest]
    #[case::whole(0..30, &[0, 10, 20, 30], vec![0..10, 10..20, 20..30])]
    #[case::clipped(5..25, &[0, 10, 20, 30], vec![5..10, 10..20, 20..25])]
    #[case::inside_one_split(12..18, &[0, 10, 20, 30], vec![12..18])]
    #[case::empty(5..5, &[0, 10], vec![])]
    fn parts_follow_splits(
        #[case] rows: Range<u64>,
        #[case] splits: &[u64],
        #[case] expected: Vec<Range<u64>>,
    ) {
        assert_eq!(part_ranges(rows, splits), expected);
    }
}
