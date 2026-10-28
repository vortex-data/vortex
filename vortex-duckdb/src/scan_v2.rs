// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan for DuckDB.
//!
//! The driver prepares each file, prunes its rows, and builds a queue of part planners. DuckDB's
//! workers take one part at a time and drive its entire planner tree to completion. Each part
//! contains several chunks: while one chunk waits for IO, the run can advance another chunk.
//! Planners describe work; the shared queue determines which worker executes it.
//!
//! The thread initializing the file performs whole-file pruning before queueing any parts.
//! DuckDB initializes a file on one thread while its others scan other files. A filter with
//! dynamic comparisons, such as a top-N bound, is pruned again inside each part when a worker
//! takes it. The part then emits its chunk planners as ordinary children of the worker's run.
//!
//! Queued and running parts share the file execution's read coordinator. Each queued planner
//! owns its row claim so reads remain available until all possible consumers have finished.
//!
//! Chunks use the prepared scan's fixed row grid, shared with DataFusion and independent of
//! stored layout boundaries. The driver groups these chunks into a few parts per available thread.

use std::collections::VecDeque;
use std::future;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::io::request::IoService;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::layout::scan::v2::ChunkPlanner;
use vortex::layout::scan::v2::PreparedScan;
use vortex::layout::scan::v2::PrunePlanner;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::mask::Mask;
use vortex::scan::planning::driver::FileReads;
use vortex::scan::planning::driver::Progress;
use vortex::scan::planning::driver::RowClaim;
use vortex::scan::planning::driver::Run;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::WorkScope;
use vortex_utils::parallelism::get_available_parallelism;

use crate::RUNTIME;

/// The parts of a file's rows per thread, so threads that finish early find more work.
const PARTS_PER_THREAD: usize = 4;

/// One file's V2 scan, shared by the threads scanning it.
pub(crate) struct FileScan {
    scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
    /// Part planners in row order that no worker has taken yet.
    work: Mutex<VecDeque<Part>>,
    /// The number of planned parts, taken or not.
    planned: usize,
    /// The threads DuckDB has given the file so far.
    claims: AtomicUsize,
    /// The rows of the parts taken so far, for DuckDB's progress bar.
    rows_taken: AtomicU64,
}

/// One worker assignment, covering a consecutive group of whole chunks.
struct Part {
    scope: WorkScope,
    planner: Box<dyn Planner>,
}

impl FileScan {
    /// Prepares `builder`, prunes its rows, and queues planners for groups of fixed-size chunks.
    /// With `dynamic`, each queued part prunes again when a worker takes it.
    pub(crate) fn try_new(builder: ScanBuilder, dynamic: bool) -> VortexResult<Self> {
        let scan = Arc::new(builder.prepare()?);
        let io = Arc::new(FileReads::new(scan.io(), scan.read_lifetimes()));
        let mut parts = get_available_parallelism().unwrap_or(1) * PARTS_PER_THREAD;
        let mut chunks = scan.row_chunks();
        if scan.sparse_selection() {
            // A sparse selection, such as an engine's fetch of a few rows by id, selects rows in
            // few chunks: each of them is its own part, and the rest are not planned at all, so
            // DuckDB gives the file a thread per chunk that has work, as V1 does with its splits.
            let selection = scan.selection();
            chunks.retain(|chunk| selection.row_count_in_range(chunk) > 0);
            parts = chunks.len();
        }
        let work = plan_parts(&scan, &io, chunks, parts, dynamic)?;
        Ok(Self {
            scan,
            io,
            planned: work.len(),
            work: Mutex::new(work),
            claims: AtomicUsize::new(0),
            rows_taken: AtomicU64::new(0),
        })
    }

    /// Whether the file has work for one more thread: DuckDB gives it one thread per piece of
    /// planned work at most.
    pub(crate) fn claim(&self) -> bool {
        self.claims.fetch_add(1, Ordering::Relaxed) < self.planned
    }

    /// The percentage of the scanned rows whose chunks threads have taken.
    pub(crate) fn progress(&self) -> f64 {
        let rows = self.scan.rows();
        let total = rows.end - rows.start;
        if total == 0 || self.work.lock().is_empty() {
            return 100.0;
        }
        100.0 * self.rows_taken.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// The next array of this file for the thread whose run is `run`, or `None` once the file has
    /// no more work for the thread. Drives one part to completion before taking another, and
    /// waits for reads on the calling thread.
    pub(crate) fn next_array(&self, run: &mut Run) -> VortexResult<Option<ArrayRef>> {
        loop {
            if run.live_roots() == 0 {
                let Some(part) = self.next_part() else {
                    return Ok(None);
                };
                run.admit(part.planner, part.scope, self.io.session());
            }
            match run.advance()? {
                Progress::Batch(batch) => return Ok(Some(batch.array)),
                Progress::Spawn(..) => {
                    vortex_bail!("a queued part must emit ordinary child planners");
                }
                Progress::RootDone(_) => {}
                Progress::Idle => return Ok(None),
                Progress::Waiting => {
                    RUNTIME.block_on(future::poll_fn(|cx| run.poll_completion(cx)))?;
                }
            }
        }
    }

    /// Takes the next part no worker has claimed.
    fn next_part(&self) -> Option<Part> {
        let part = self.work.lock().pop_front();
        if let Some(part) = &part {
            self.rows_taken.fetch_add(
                part.scope.rows.end - part.scope.rows.start,
                Ordering::Relaxed,
            );
        }
        part
    }
}

/// Prunes once, then builds the worker queue without executing any of its parts.
fn plan_parts(
    scan: &Arc<PreparedScan>,
    io: &Arc<FileReads>,
    chunks: Vec<Range<u64>>,
    count: usize,
    dynamic: bool,
) -> VortexResult<VecDeque<Part>> {
    let parts = parts(chunks, count);
    let claim = io.claim(scan.rows().clone());
    if scan.filter().is_none() {
        return Ok(part_planners(scan, claim, parts, None, dynamic));
    }

    let work = Arc::new(Mutex::new(VecDeque::new()));
    let next = {
        let scan = Arc::clone(scan);
        let work = Arc::clone(&work);
        move |kept, claim| {
            *work.lock() = part_planners(&scan, claim, parts, kept, dynamic);
            None
        }
    };
    let rows = scan.rows().clone();
    let pruning = PrunePlanner::new(Arc::clone(scan), claim, Box::new(next));
    let mut run = Run::new();
    run.admit(
        Box::new(pruning),
        WorkScope {
            file_ordinal: 0,
            rows,
        },
        io.session(),
    );
    loop {
        match run.advance()? {
            Progress::RootDone(_) | Progress::Idle => break,
            Progress::Waiting => {
                RUNTIME.block_on(future::poll_fn(|cx| run.poll_completion(cx)))?;
            }
            Progress::Batch(_) | Progress::Spawn(..) => {
                vortex_bail!("whole-file pruning must finish without executing parts");
            }
        }
    }
    let work = std::mem::take(&mut *work.lock());
    Ok(work)
}

/// Builds one planner per surviving part. All children stay within the worker's run.
fn part_planners(
    scan: &Arc<PreparedScan>,
    mut claim: RowClaim,
    parts: Vec<Vec<Range<u64>>>,
    kept: Option<Mask>,
    dynamic: bool,
) -> VecDeque<Part> {
    let mut work = VecDeque::new();
    for chunks in parts {
        let (Some(first), Some(last)) = (chunks.first(), chunks.last()) else {
            continue;
        };
        let rows = first.start..last.end;
        let mut part_claim = claim.split_to(rows.end);
        part_claim.restrict_to(rows.clone());
        let kept = match &kept {
            Some(kept) => {
                let offset = |row: u64| {
                    usize::try_from(row - scan.rows().start).vortex_expect("a part fits usize")
                };
                let kept = kept.slice(offset(rows.start)..offset(rows.end));
                if kept.all_false() {
                    continue;
                }
                Some(kept)
            }
            None => None,
        };
        let scope = WorkScope {
            file_ordinal: 0,
            rows,
        };
        let scan = Arc::clone(scan);
        let selection = scan.selection().clone();
        let planner: Box<dyn Planner> = if dynamic {
            let next_scan = Arc::clone(&scan);
            let next = move |pruned: Option<Mask>, claim: RowClaim| -> Option<Box<dyn Planner>> {
                let kept = match (kept, pruned) {
                    (Some(kept), Some(pruned)) => Some(kept & &pruned),
                    (kept, pruned) => kept.or(pruned),
                };
                Some(Box::new(ChunkPlanner::with_chunks(
                    next_scan, claim, chunks, selection, kept,
                )))
            };
            Box::new(PrunePlanner::new(scan, part_claim, Box::new(next)))
        } else {
            Box::new(ChunkPlanner::with_chunks(
                scan, part_claim, chunks, selection, kept,
            ))
        };
        work.push_back(Part { scope, planner });
    }
    work
}

/// Groups `chunks` into up to `parts` runs of consecutive chunks, of as even a count as they
/// divide into.
fn parts(chunks: Vec<Range<u64>>, parts: usize) -> Vec<Vec<Range<u64>>> {
    if chunks.is_empty() {
        return Vec::new();
    }
    let per_part = chunks.len().div_ceil(parts.max(1));
    let mut chunks = chunks.into_iter().peekable();
    let mut grouped = Vec::new();
    while chunks.peek().is_some() {
        grouped.push(chunks.by_ref().take(per_part).collect());
    }
    grouped
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::parts;

    #[rstest]
    #[case::even(4, 2, vec![2, 2])]
    #[case::ragged(5, 2, vec![3, 2])]
    #[case::more_parts_than_chunks(2, 8, vec![1, 1])]
    #[case::empty(0, 4, vec![])]
    fn parts_group_whole_chunks(
        #[case] chunks: u64,
        #[case] count: usize,
        #[case] expected: Vec<usize>,
    ) {
        let grouped = parts((0..chunks).map(|i| i..i + 1).collect(), count);
        assert_eq!(grouped.iter().map(Vec::len).collect::<Vec<_>>(), expected);
        assert_eq!(
            grouped.concat(),
            (0..chunks).map(|i| i..i + 1).collect::<Vec<_>>()
        );
    }
}
