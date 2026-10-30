// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan for DuckDB.
//!
//! A planner says what it needs: compute, or reads. This driver decides where each runs, and
//! DuckDB's own threads run all of it. Each file is prepared once, and the thread that initializes
//! it cuts the scanned rows into parts on the file's layout splits, the units a V1 scan hands its
//! threads, after pruning the scan as a whole when its filter is dynamic. A thread takes the next
//! part and runs it on a run of its own, inline, waiting only for its reads: with a filter, the
//! part's pruning plans first, continuing into the chunks of the rows pruning kept; without, the
//! chunks directly. Threads join the file only while parts are left, so the threads a file
//! cannot use move on to the next file.

use std::collections::VecDeque;
use std::future;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::io::request::IoConsumer;
use vortex::io::request::IoRequestId;
use vortex::io::request::IoResult;
use vortex::io::request::IoService;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::layout::scan::v2::ChunkPlanner;
use vortex::layout::scan::v2::PreparedScan;
use vortex::layout::scan::v2::PrunePlanner;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::mask::Mask;
use vortex::scan::planning::driver::Progress;
use vortex::scan::planning::driver::Run;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::PlannerOutput;
use vortex::scan::planning::planner::State;
use vortex::scan::planning::planner::WorkScope;
use vortex::scan::selection::Selection;

use crate::RUNTIME;

/// The chunks a thread runs beside each other on its run, so one chunk's reads overlap another's
/// compute.
const CHUNKS_PER_THREAD: usize = 1;

/// One file's V2 scan, shared by the threads scanning it.
pub(crate) struct FileScan {
    scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
    /// The work the root spawned, in row order, that no thread has taken yet.
    queue: Mutex<VecDeque<Work>>,
    /// Set once a thread has taken the last of `queue`.
    exhausted: AtomicBool,
    /// The rows of the chunks taken so far, for DuckDB's progress bar.
    rows_taken: AtomicU64,
}

/// What a thread takes from the file next.
enum Work {
    /// A chunk to run on the thread's run.
    Chunk(WorkScope, Box<dyn Planner>),
    /// A batch the root produced itself.
    Batch(ArrayRef),
}

impl FileScan {
    /// Prepares `builder`, composes its planners, and runs the root until it has spawned the
    /// planner of every part. The root only prunes the scan, for a dynamic filter, and cuts parts,
    /// so this reads zone maps at most.
    ///
    /// Running the root up front leaves the file with a known list of parts, like the splits of a
    /// V1 scan: a thread joins the file only while a part is left, and the threads DuckDB has to
    /// spare move on to the next file instead of waiting for this file's root.
    ///
    /// `splits` are the file's layout split boundaries, as a V1 scan of the same fields cuts it.
    /// `dynamic` says the filter's value changes while the scan runs, as a top-N's bound does.
    pub(crate) fn try_new(
        builder: ScanBuilder,
        splits: &[u64],
        dynamic: bool,
    ) -> VortexResult<Self> {
        let scan = Arc::new(builder.prepare()?);
        let io = scan.io();
        let rows = scan.rows().clone();
        let mut root = Run::new();
        root.admit(
            compose(&scan, splits, dynamic),
            WorkScope {
                file_ordinal: 0,
                rows,
            },
            io.session(),
        );
        let mut queue = VecDeque::new();
        loop {
            match root.advance()? {
                Progress::Batch(batch) => queue.push_back(Work::Batch(batch.array)),
                Progress::Spawn(scope, chunk) => queue.push_back(Work::Chunk(scope, chunk)),
                Progress::RootDone(_) if root.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => break,
                Progress::Waiting => {
                    RUNTIME.block_on(future::poll_fn(|cx| root.poll_completion(cx)))?;
                }
            }
        }
        Ok(Self {
            scan,
            io,
            exhausted: AtomicBool::new(queue.is_empty()),
            queue: Mutex::new(queue),
            rows_taken: AtomicU64::new(0),
        })
    }

    /// Whether every chunk has been taken, so a thread joining the scan finds no work.
    pub(crate) fn is_exhausted(&self) -> bool {
        self.exhausted.load(Ordering::Acquire)
    }

    /// The percentage of the scanned rows whose chunks threads have taken.
    pub(crate) fn progress(&self) -> f64 {
        let rows = self.scan.rows();
        let total = rows.end - rows.start;
        if total == 0 || self.exhausted.load(Ordering::Relaxed) {
            return 100.0;
        }
        100.0 * self.rows_taken.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// The next array of this file for the thread whose run is `run`, or `None` once the file has
    /// no more work for the thread. Keeps the thread's run fed with chunks from the file, and
    /// waits for reads on the calling thread.
    pub(crate) fn next_array(&self, run: &mut Run) -> VortexResult<Option<ArrayRef>> {
        loop {
            while run.live_roots() < CHUNKS_PER_THREAD {
                match self.next_work() {
                    Some(Work::Chunk(scope, chunk)) => {
                        run.admit(chunk, scope, self.io.session());
                    }
                    Some(Work::Batch(array)) => return Ok(Some(array)),
                    None => break,
                }
            }
            match run.advance()? {
                Progress::Batch(batch) => return Ok(Some(batch.array)),
                Progress::Spawn(scope, child) => {
                    run.admit(child, scope, self.io.session());
                }
                Progress::RootDone(_) => {}
                Progress::Idle => return Ok(None),
                Progress::Waiting => {
                    RUNTIME.block_on(future::poll_fn(|cx| run.poll_completion(cx)))?;
                }
            }
        }
    }

    /// Takes the next chunk or batch of the file, or `None` once every one has been taken.
    fn next_work(&self) -> Option<Work> {
        let mut queue = self.queue.lock();
        let work = queue.pop_front()?;
        if queue.is_empty() {
            self.exhausted.store(true, Ordering::Release);
        }
        if let Work::Chunk(scope, _) = &work {
            self.rows_taken
                .fetch_add(scope.rows.end - scope.rows.start, Ordering::Relaxed);
        }
        Some(work)
    }
}

/// The planners of a scan, composed: with a dynamic filter, its pruning plans first over the scan's
/// rows, continuing into the parts of the rows they keep; otherwise, the parts directly.
fn compose(scan: &Arc<PreparedScan>, splits: &[u64], dynamic: bool) -> Box<dyn Planner> {
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
/// exported by one thread, which lets DuckDB evaluate an expression over that dictionary once.
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
                    Box::new(ChunkPlanner::single(scan, part, selection, kept))
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
