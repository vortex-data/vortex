// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan for DuckDB.
//!
//! A planner says what it needs: compute, or reads. This driver decides where each runs, and
//! DuckDB's own threads run all of it. Each file is prepared once. Its root planner cuts the
//! scanned rows into parts and spawns, per part, with a filter, its pruning plans over the part,
//! continuing into the chunks of the rows they keep; without, the part's chunks directly. The
//! root planner's run is shared by every thread scanning the file and reads nothing. A thread
//! takes the next part the root spawns and runs it on a run of its own, inline, waiting only for
//! its reads. Threads keep taking parts until the root is done, so the file's work, its pruning
//! included, spreads over as many threads as DuckDB gives the scan, however its rows divide.
//!
//! The root cuts the scanned rows into parts of whole 65,536-row blocks, a few per thread. A filtered scan's chunks are up to 1,048,576 rows, so without
//! parts a file of a few million rows would keep only a few threads busy. A file too short to give
//! every thread a 65,536-row block, such as a file of wide rows, is cut on 8,192-row blocks
//! instead.

use std::future;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use vortex::array::ArrayRef;
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
const PARTS_PER_THREAD: u64 = 4;
/// The row blocks parts are cut on: the rows of a projection chunk, and of the writer's chunks.
const PART_ROW_BLOCK: u64 = 65_536;
/// The row blocks parts are cut on when too few [`PART_ROW_BLOCK`]s cover the rows to give every
/// part one: the row grid every chunk boundary lies on.
const MIN_PART_ROW_BLOCK: u64 = 8_192;

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
    /// Prepares `builder` and composes its planners. Lowering and planning touch no segment.
    pub(crate) fn try_new(builder: ScanBuilder) -> VortexResult<Self> {
        let scan = Arc::new(builder.prepare()?);
        let io = scan.io();
        let rows = scan.rows().clone();
        let mut root = Run::new();
        root.admit(
            compose(&scan),
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
fn compose(scan: &Arc<PreparedScan>) -> Box<dyn Planner> {
    let parallelism = get_available_parallelism().unwrap_or(1) as u64;
    Box::new(PartPlanner::new(
        Arc::clone(scan),
        scan.rows().clone(),
        parallelism * PARTS_PER_THREAD,
    ))
}

/// Spawns a planner per part of a scan's rows, in row order: with a filter, its pruning plans
/// over the part, continuing into the chunks of the rows they keep; without, the part's chunks.
/// Parts end on whole row blocks, see [`part_ranges`].
///
/// Each part prunes on the thread that takes it, so a file's pruning runs on as many threads as
/// take its parts, and spawning the parts reads nothing: the threads waiting on the root take
/// their parts at once and move on to the next file once the root has spawned the last.
struct PartPlanner {
    scan: Arc<PreparedScan>,
    selection: Selection,
    parts: std::vec::IntoIter<Range<u64>>,
}

impl PartPlanner {
    fn new(scan: Arc<PreparedScan>, rows: Range<u64>, parts: u64) -> Self {
        Self {
            selection: scan.selection().clone(),
            scan,
            parts: part_ranges(rows, parts).into_iter(),
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
        let Some(part) = self.parts.next() else {
            return Ok(PlannerOutput::Done);
        };
        let chunks = {
            let scan = Arc::clone(&self.scan);
            let selection = self.selection.clone();
            let part = part.clone();
            move |kept: Option<Mask>| -> Box<dyn Planner> {
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
        Ok(PlannerOutput::Spawn(scope, planner))
    }
}

/// Cuts `rows` into up to `parts` non-empty ranges of whole row blocks of the file's row domain
/// that tile it, clipped to `rows`. The blocks are [`PART_ROW_BLOCK`]s, or [`MIN_PART_ROW_BLOCK`]s
/// when there are fewer of those than `parts`.
fn part_ranges(rows: Range<u64>, parts: u64) -> Vec<Range<u64>> {
    if rows.is_empty() {
        return Vec::new();
    }
    let parts = parts.max(1);
    let blocks_of = |block: u64| rows.end.div_ceil(block) - rows.start / block;
    let block = if blocks_of(PART_ROW_BLOCK) >= parts {
        PART_ROW_BLOCK
    } else {
        MIN_PART_ROW_BLOCK
    };
    let first_block = rows.start / block;
    let blocks = blocks_of(block);
    let blocks_per_part = blocks.div_ceil(parts);
    (0..blocks.div_ceil(blocks_per_part))
        .map(|part| {
            let start = (first_block + part * blocks_per_part) * block;
            let end = start + blocks_per_part * block;
            start.max(rows.start)..end.min(rows.end)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;

    use super::MIN_PART_ROW_BLOCK;
    use super::PART_ROW_BLOCK;
    use super::part_ranges;

    const B: u64 = PART_ROW_BLOCK;
    const M: u64 = MIN_PART_ROW_BLOCK;

    #[rstest]
    #[case::even(0..4 * B, 2, vec![0..2 * B, 2 * B..4 * B])]
    #[case::ragged_end(0..3 * B + 10, 2, vec![0..2 * B, 2 * B..3 * B + 10])]
    #[case::more_parts_than_blocks(0..2 * B, 8, (0..8).map(|p| 2 * M * p..2 * M * (p + 1)).collect())]
    #[case::wide_rows(0..100_000, 4, vec![0..4 * M, 4 * M..8 * M, 8 * M..12 * M, 12 * M..100_000])]
    #[case::more_parts_than_min_blocks(0..2 * M, 8, vec![0..M, M..2 * M])]
    #[case::small(0..100, 4, vec![0..100])]
    #[case::unaligned_start(B / 2..3 * B, 2, vec![B / 2..2 * B, 2 * B..3 * B])]
    #[case::empty(5..5, 4, vec![])]
    fn parts_end_on_blocks(
        #[case] rows: Range<u64>,
        #[case] parts: u64,
        #[case] expected: Vec<Range<u64>>,
    ) {
        assert_eq!(part_ranges(rows, parts), expected);
    }
}
