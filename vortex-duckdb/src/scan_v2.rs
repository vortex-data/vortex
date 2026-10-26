// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan for DuckDB.
//!
//! A planner says what it needs: compute, or reads. This driver decides where each runs, and
//! DuckDB's own threads run all of it. Each file is prepared once and its planners are composed
//! over the prepared scan: with a filter, its pruning plans over the scanned rows first,
//! continuing into the chunks of the rows they keep; without, the chunks directly. The root
//! planner's run is shared by every thread scanning the file. A thread takes the next chunk the
//! root spawns and runs it on a run of its own, inline, waiting only for its reads. Threads keep
//! taking chunks until the root is done, so the file's work spreads over as many threads as
//! DuckDB gives the scan, however its rows divide.
//!
//! Chunks end only on the file's own chunk boundaries, of the columns the scan reads, so no chunk
//! of the file is decoded twice by two chunks of the scan. Spans between boundaries are merged up
//! to the scan's chunk size. The root groups the chunks into parts, a few per thread, and spawns a
//! chunk planner per part. A filtered scan's chunks are up to 1,048,576 rows, so without parts a
//! file of a few million rows would keep only a few threads busy.

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
use vortex_utils::parallelism::get_available_parallelism;

use crate::RUNTIME;

/// The chunks a thread runs beside each other on its run, so one chunk's reads overlap another's
/// compute.
const CHUNKS_PER_THREAD: usize = 1;
/// The parts of a file's rows per thread, so threads that finish early find more work.
const PARTS_PER_THREAD: usize = 4;

/// One file's V2 scan, shared by the threads scanning it.
pub(crate) struct FileScan {
    scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
    /// The run of the root planner, locked while a thread takes the chunk it spawns next.
    root: Mutex<Run>,
    /// Set once the root has spawned its last chunk.
    exhausted: AtomicBool,
    /// The rows of the chunks taken so far, for DuckDB's progress bar.
    rows_taken: AtomicU64,
}

/// What a thread takes from the root next.
enum Work {
    /// A chunk to run on the thread's run.
    Chunk(WorkScope, Box<dyn Planner>),
    /// A batch the root produced itself.
    Batch(ArrayRef),
}

impl FileScan {
    /// Prepares `builder` and composes its planners, cutting chunks on `boundaries`, the sorted
    /// rows the file's chunks of the scanned columns end at. Lowering and planning touch no
    /// segment.
    pub(crate) fn try_new(builder: ScanBuilder, boundaries: &[u64]) -> VortexResult<Self> {
        let scan = Arc::new(builder.prepare()?);
        let io = scan.io();
        let rows = scan.rows().clone();
        let chunks = chunk_ranges(rows.clone(), boundaries, scan.chunk_rows() as u64);
        let mut root = Run::new();
        root.admit(
            compose(&scan, chunks),
            WorkScope {
                file_ordinal: 0,
                rows,
            },
            io.session(),
        );
        Ok(Self {
            scan,
            io,
            root: Mutex::new(root),
            exhausted: AtomicBool::new(false),
            rows_taken: AtomicU64::new(0),
        })
    }

    /// Whether the root has spawned its last chunk, so a thread joining the scan finds no work.
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
    /// no more work for the thread. Keeps the thread's run fed with chunks from the root, and
    /// waits for reads on the calling thread.
    pub(crate) fn next_array(&self, run: &mut Run) -> VortexResult<Option<ArrayRef>> {
        loop {
            while run.live_roots() < CHUNKS_PER_THREAD {
                match self.next_work()? {
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

    /// Advances the root until it spawns a chunk or produces a batch, or `None` once it is done.
    fn next_work(&self) -> VortexResult<Option<Work>> {
        if self.exhausted.load(Ordering::Acquire) {
            return Ok(None);
        }
        let mut root = self.root.lock();
        loop {
            if self.exhausted.load(Ordering::Acquire) {
                return Ok(None);
            }
            match root.advance()? {
                Progress::Batch(batch) => return Ok(Some(Work::Batch(batch.array))),
                Progress::Spawn(scope, chunk) => {
                    self.rows_taken
                        .fetch_add(scope.rows.end - scope.rows.start, Ordering::Relaxed);
                    return Ok(Some(Work::Chunk(scope, chunk)));
                }
                Progress::RootDone(_) if root.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => {
                    self.exhausted.store(true, Ordering::Release);
                    return Ok(None);
                }
                // The root waits for the pruning plans' reads. The threads waiting on the lock
                // would have nothing to run until they arrive.
                Progress::Waiting => {
                    RUNTIME.block_on(future::poll_fn(|cx| root.poll_completion(cx)))?;
                }
            }
        }
    }
}

/// The planners of a scan, composed: with a filter, its pruning plans first over the scan's rows,
/// continuing into the parts of the rows they keep; without, the parts directly.
fn compose(scan: &Arc<PreparedScan>, chunks: Vec<Range<u64>>) -> Box<dyn Planner> {
    let rows = scan.rows().clone();
    let parts = {
        let scan = Arc::clone(scan);
        let rows = rows.clone();
        move |kept: Option<Mask>| -> Box<dyn Planner> {
            let parallelism = get_available_parallelism().unwrap_or(1);
            Box::new(PartPlanner::new(
                scan,
                rows,
                kept,
                parts(chunks, parallelism * PARTS_PER_THREAD),
            ))
        }
    };
    if scan.filter().is_some() {
        Box::new(PrunePlanner::new(Arc::clone(scan), rows, Box::new(parts)))
    } else {
        parts(None)
    }
}

/// Spawns a chunk planner per part of a scan's rows, in row order. A part is a run of whole
/// chunks.
struct PartPlanner {
    scan: Arc<PreparedScan>,
    selection: Selection,
    /// The first row of `kept`.
    first_row: u64,
    /// The rows zone pruning kept, from `first_row`, or `None` when nothing was pruned.
    kept: Option<Mask>,
    /// The chunks of each part, which tile the scan's rows.
    parts: std::vec::IntoIter<Vec<Range<u64>>>,
}

impl PartPlanner {
    fn new(
        scan: Arc<PreparedScan>,
        rows: Range<u64>,
        kept: Option<Mask>,
        parts: Vec<Vec<Range<u64>>>,
    ) -> Self {
        Self {
            selection: scan.selection().clone(),
            scan,
            first_row: rows.start,
            kept,
            parts: parts.into_iter(),
        }
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
        for chunks in self.parts.by_ref() {
            let (Some(first), Some(last)) = (chunks.first(), chunks.last()) else {
                continue;
            };
            let part = first.start..last.end;
            let kept = match &self.kept {
                Some(kept) => {
                    let offset = |row: u64| {
                        usize::try_from(row - self.first_row).vortex_expect("a part fits usize")
                    };
                    let kept = kept.slice(offset(part.start)..offset(part.end));
                    if kept.all_false() {
                        continue;
                    }
                    Some(kept)
                }
                None => None,
            };
            let chunks = ChunkPlanner::with_chunks(
                Arc::clone(&self.scan),
                part.clone(),
                chunks,
                self.selection.clone(),
                kept,
            );
            let scope = WorkScope {
                file_ordinal: 0,
                rows: part,
            };
            return Ok(PlannerOutput::Spawn(scope, Box::new(chunks)));
        }
        Ok(PlannerOutput::Done)
    }
}

/// Cuts `rows` into chunks that end only on `boundaries` or at the end of `rows`, merging the
/// spans between boundaries into chunks of at most `max_rows` rows. A span wider than `max_rows`
/// stays one chunk, so no chunk of the file is split.
fn chunk_ranges(rows: Range<u64>, boundaries: &[u64], max_rows: u64) -> Vec<Range<u64>> {
    let mut chunks = Vec::new();
    let mut start = rows.start;
    let mut end = rows.start;
    let ends = boundaries
        .iter()
        .copied()
        .filter(|&row| row > rows.start && row < rows.end)
        .chain((!rows.is_empty()).then_some(rows.end));
    for row in ends {
        if row - start > max_rows && end > start {
            chunks.push(start..end);
            start = end;
        }
        end = row;
    }
    if end > start {
        chunks.push(start..end);
    }
    chunks
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
    use std::ops::Range;

    use rstest::rstest;

    use super::chunk_ranges;
    use super::parts;

    #[rstest]
    #[case::merged(0..400, &[100, 200, 300, 400], 200, vec![0..200, 200..400])]
    #[case::wide_spans(0..300, &[0, 87, 182, 300], 64, vec![0..87, 87..182, 182..300])]
    #[case::clipped(50..250, &[100, 200, 300], 100, vec![50..100, 100..200, 200..250])]
    #[case::no_boundaries(0..10, &[], 4, vec![0..10])]
    #[case::empty(5..5, &[5], 4, vec![])]
    fn chunks_end_on_boundaries(
        #[case] rows: Range<u64>,
        #[case] boundaries: &[u64],
        #[case] max_rows: u64,
        #[case] expected: Vec<Range<u64>>,
    ) {
        assert_eq!(chunk_ranges(rows, boundaries, max_rows), expected);
    }

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
