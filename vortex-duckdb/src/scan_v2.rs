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
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::error::VortexResult;
use vortex::io::request::IoService;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::scan::planning::driver::Progress;
use vortex::scan::planning::driver::Run;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::WorkScope;
use vortex::scan::prepare::PreparedScan;
use vortex::scan::prepare::part_root;

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
            part_root(&scan, splits, dynamic),
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
