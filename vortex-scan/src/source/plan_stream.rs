// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An opened [`PlanDataSource`](super::PlanDataSource) scan: V2's planners and [`Run`] driver
//! behind [`ScanExec`] and its [`ScanStream`](crate::stream::ScanStream)s.
//!
//! When an unordered scan is made, it prepares each child once and runs its [`part_root`], which
//! cuts the child into parts: the rows between the child's splits for the fields the scan reads,
//! such as a file's layout splits, or the scan's chunks for a child without splits. With a dynamic
//! filter, such as the bound a top-N pushes down, the root first prunes the child as a whole
//! against the bound as it stands, which reads zone maps. Those are the reads the scan needs to
//! know its units of work, so they run in the async [`scan`](crate::DataSource::scan), which the
//! host awaits. Opening the scan reads nothing: it only queues the parts, so no stream waits on
//! another for its work. Each part prunes again against the bound as it stands when a stream
//! takes it.
//!
//! Every stream of the scan takes parts from the one queue, and the scan is exhausted once the
//! last part is taken. Each stream owns a [`Run`], driven on the task that polls the stream, and
//! runs one part on it at a time: the part's chunks run side by side, so one chunk's reads overlap
//! another's compute, batches are returned as they are produced, and reads are awaited with
//! [`Run::poll_completion`]. Nothing is spawned, so a host that polls a stream with `block_on`
//! runs the scan's compute on its own thread.
//!
//! An ordered scan returns its rows in order through its first stream, running one child after
//! another on [`ScanDriver`], which keeps a scan's chunks in order. The scan is exhausted once
//! that stream exists, and later streams return nothing. A host that reads an ordered scan in
//! parallel [splits](crate::DataSourceScan::split) it first.
//!
//! A limit is one [`RowLimit`] for the whole scan: streams stop taking parts once it is spent,
//! which also exhausts the scan, and each stream's output is trimmed to it.

use std::collections::VecDeque;
use std::future;
use std::iter;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_functions::dynamic::DynamicExprUpdates;
use vortex_io::request::IoService;
use vortex_io::session::RuntimeSessionExt;

use super::plan::ScanState;
use super::plan::Unit;
use crate::execution::driver::ScanDriver;
use crate::execution::scan_chunks;
use crate::limit::RowLimit;
use crate::limit::limit_rows;
use crate::planning::driver::Progress;
use crate::planning::driver::Run;
use crate::planning::planner::Planner;
use crate::planning::planner::WorkScope;
use crate::prepare::PreparedScan;
use crate::prepare::part_root;
use crate::stream::Claims;
use crate::stream::ExhaustedFlag;
use crate::stream::ScanExec;

/// The parts of an unordered scan, cut when the scan is made.
pub(super) struct Planned {
    work: Vec<Work>,
}

/// Prepares each of the scan's `units` once and runs its root, awaiting the zone maps a dynamic
/// filter's root reads, and returns the work the roots produce, in unit and row order.
pub(super) async fn plan(state: &ScanState, units: &[Unit]) -> VortexResult<Planned> {
    let mut work = Vec::new();
    for unit in units {
        let rows = state.unit_rows(unit)?;
        if rows.is_empty() {
            continue;
        }
        let (child, root) = prepare(state, unit, rows)?;
        work.extend(run_root(child, root).await?);
    }
    Ok(Planned { work })
}

/// Opens a scan of `units`. An unordered scan reads the parts it `planned`.
pub(super) fn open(
    state: Arc<ScanState>,
    units: Vec<Unit>,
    planned: Option<Planned>,
) -> VortexResult<ScanExec> {
    let dtype = state.dtype.clone();
    let limit = state.limit.map(RowLimit::new);
    let exhausted = ExhaustedFlag::new();
    if state.ordered {
        let arrays = Arc::new(Mutex::new(Some(in_order(state, units))));
        let flag = exhausted.clone();
        return Ok(ScanExec::new(dtype, exhausted, move || {
            // The first stream returns every row, so later ones would find nothing.
            flag.set();
            let arrays = arrays
                .lock()
                .take()
                .unwrap_or_else(|| stream::empty().boxed());
            limited(arrays, limit.clone())
        }));
    }

    let planned = planned.vortex_expect("an unordered scan cuts its parts when it is made");
    if planned.work.is_empty() {
        exhausted.set();
    }
    let queue = Arc::new(Queue {
        total: planned.work.len(),
        work: Mutex::new(planned.work.into()),
        taken: AtomicUsize::new(0),
        exhausted: exhausted.clone(),
    });
    let progress = Arc::clone(&queue);
    Ok(ScanExec::new(dtype, exhausted, move || {
        let worker = Worker {
            queue: Arc::clone(&queue),
            limit: limit.clone(),
            run: Run::new(),
            child: None,
        };
        let arrays = stream::unfold(Some(worker), |worker| async move {
            let mut worker = worker?;
            match worker.next_array().await {
                Ok(Some(array)) => Some((Ok(array), Some(worker))),
                Ok(None) => None,
                Err(err) => Some((Err(err), None)),
            }
        })
        .boxed();
        limited(arrays, limit.clone())
    })
    .with_progress(move || progress.claims()))
}

/// One child's prepared scan, shared by the parts of it that streams run.
struct PreparedChild {
    rows: Range<u64>,
    io: Arc<dyn IoService>,
}

/// A part of a child: the planner its root spawned for some of its rows.
struct Part {
    child: Arc<PreparedChild>,
    scope: WorkScope,
    planner: Box<dyn Planner>,
}

/// Prepares the scan of `rows` of the unit's child once and composes its root.
fn prepare(
    state: &ScanState,
    unit: &Unit,
    rows: Range<u64>,
) -> VortexResult<(Arc<PreparedChild>, Box<dyn Planner>)> {
    let child = state.child(unit.child);
    let splits = child.splits(&state.projection, state.filter.as_ref())?;
    let scan = Arc::new(state.builder(child, rows).prepare()?);
    let splits = splits.unwrap_or_else(|| chunk_bounds(&scan));
    let dynamic = state
        .filter
        .as_ref()
        .is_some_and(|filter| DynamicExprUpdates::new(filter).is_some());
    let root = part_root(&scan, &splits, dynamic);
    let prepared = PreparedChild {
        rows: scan.rows().clone(),
        io: scan.io(),
    };
    Ok((Arc::new(prepared), root))
}

/// The boundaries of `scan`'s chunks, which cut the work of a child without splits of its own.
fn chunk_bounds(scan: &PreparedScan) -> Vec<u64> {
    let rows = scan.rows().clone();
    let chunks = scan_chunks(rows.clone(), scan.chunk_rows);
    iter::once(rows.start)
        .chain(chunks.iter().map(|chunk| chunk.end))
        .collect()
}

/// Runs a child's root on a run of its own, awaiting only its reads, and returns the work it
/// produces, in row order: the parts it spawns, and any batch it returns itself.
async fn run_root(child: Arc<PreparedChild>, root: Box<dyn Planner>) -> VortexResult<Vec<Work>> {
    let mut run = Run::new();
    let scope = WorkScope {
        file_ordinal: 0,
        rows: child.rows.clone(),
    };
    run.admit(root, scope, child.io.session());
    let mut work = Vec::new();
    loop {
        match run.advance()? {
            Progress::Spawn(scope, planner) => work.push(Work::Part(Part {
                child: Arc::clone(&child),
                scope,
                planner,
            })),
            Progress::Batch(batch) => work.push(Work::Batch(batch.array)),
            Progress::RootDone(_) if run.live_roots() > 0 => {}
            Progress::RootDone(_) | Progress::Idle => break,
            Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
        }
    }
    Ok(work)
}

/// A unit of work in the queue of an unordered scan.
enum Work {
    Part(Part),
    /// A batch a root returned itself.
    Batch(ArrayRef),
}

/// The queue of an unordered scan, which every stream of it takes from.
struct Queue {
    /// The work no stream has taken yet, in unit and row order.
    work: Mutex<VecDeque<Work>>,
    /// The work the queue started with.
    total: usize,
    /// The work streams have taken.
    taken: AtomicUsize,
    /// Set once the last work is taken, or the scan's limit is spent.
    exhausted: ExhaustedFlag,
}

impl Queue {
    /// Takes the next work, or `None` once every one has been taken, and marks the scan exhausted
    /// when it takes the last.
    fn take(&self) -> Option<Work> {
        let mut work = self.work.lock();
        let next = work.pop_front()?;
        if work.is_empty() {
            self.exhausted.set();
        }
        drop(work);
        self.taken.fetch_add(1, Ordering::Relaxed);
        Some(next)
    }

    /// How much of the queue's work streams have taken, out of how much in all.
    fn claims(&self) -> Claims {
        Claims {
            claimed: self.taken.load(Ordering::Relaxed).min(self.total),
            total: self.total,
        }
    }
}

/// One stream's worker: a run of its own, fed one part at a time from the scan's queue, and driven
/// on the task that polls the stream.
struct Worker {
    queue: Arc<Queue>,
    limit: Option<RowLimit>,
    run: Run,
    /// The child of the part live on the run, whose I/O the planners that part spawns read through.
    child: Option<Arc<PreparedChild>>,
}

impl Worker {
    /// The worker's next array, or `None` once the scan has no more work for it.
    async fn next_array(&mut self) -> VortexResult<Option<ArrayRef>> {
        loop {
            if self.run.live_roots() == 0 {
                if self.limit.as_ref().is_some_and(RowLimit::is_exhausted) {
                    self.queue.exhausted.set();
                    return Ok(None);
                }
                match self.queue.take() {
                    None => return Ok(None),
                    Some(Work::Batch(array)) => return Ok(Some(array)),
                    Some(Work::Part(part)) => {
                        self.run
                            .admit(part.planner, part.scope, part.child.io.session());
                        self.child = Some(part.child);
                    }
                }
            }
            match self.run.advance()? {
                Progress::Batch(batch) => return Ok(Some(batch.array)),
                Progress::Spawn(scope, planner) => {
                    let child = self
                        .child
                        .as_ref()
                        .vortex_expect("a planner is spawned by a live part");
                    self.run.admit(planner, scope, child.io.session());
                }
                Progress::RootDone(_) | Progress::Idle => {}
                Progress::Waiting => {
                    future::poll_fn(|cx| self.run.poll_completion(cx)).await?;
                }
            }
        }
    }
}

/// The arrays of an ordered scan: each unit's rows in order, one unit after another.
fn in_order(state: Arc<ScanState>, units: Vec<Unit>) -> BoxStream<'static, VortexResult<ArrayRef>> {
    stream::iter(units)
        .flat_map(move |unit| {
            let rows = match state.unit_rows(&unit) {
                Ok(rows) => rows,
                Err(err) => return stream::once(future::ready(Err(err))).boxed(),
            };
            if rows.is_empty() {
                return stream::empty().boxed();
            }
            ScanDriver::new(state.session.handle(), 1, true)
                .run(state.builder(state.child(unit.child), rows))
        })
        .boxed()
}

/// `arrays`, trimmed to the scan's limit.
fn limited(
    arrays: BoxStream<'static, VortexResult<ArrayRef>>,
    limit: Option<RowLimit>,
) -> BoxStream<'static, VortexResult<ArrayRef>> {
    match limit {
        Some(limit) => limit_rows(arrays, limit),
        None => arrays,
    }
}
