// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The stream of an opened [`PlanDataSource`](super::PlanDataSource) scan: V2's planners and
//! [`Run`] driver behind [`ScanStream`].
//!
//! An unordered scan prepares each child once and runs its [`part_root`], which cuts the child
//! into parts: the rows between the child's splits for the fields the scan reads, such as a
//! file's layout splits, or the scan's chunks for a child without splits. The parts go into one
//! queue that every clone of the stream takes from. Each clone owns a [`Run`], driven on the task
//! that polls the clone, and runs one part on it at a time: the part's chunks run side by side,
//! so one chunk's reads overlap another's compute, batches are returned as they are produced, and
//! reads are awaited with [`Run::poll_completion`]. Nothing is spawned, so a host that polls a
//! clone with `block_on` runs the scan's compute on its own thread.
//!
//! With a dynamic filter, such as the bound a top-N pushes down, a child's root prunes the child as
//! a whole before it cuts parts, which reads zone maps. Opening reads nothing, so such a root is
//! queued work too: the first clone to take it runs it and queues its parts.
//!
//! An ordered scan returns its rows in order through the first clone to be polled, running one
//! child after another on [`ScanDriver`], which keeps a scan's chunks in order. Later clones
//! return nothing. A host that reads an ordered scan in parallel
//! [splits](crate::DataSourceScan::split) it first.
//!
//! A limit is one [`RowLimit`] for the whole stream: clones stop taking parts once it is spent,
//! and each clone's output is trimmed to it.

use std::future;
use std::iter;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_channel::Receiver;
use async_channel::Sender;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
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
use crate::stream::ScanStream;

/// The parts of an unordered scan, cut before it is opened.
pub(super) struct Planned {
    work: Vec<Work>,
    total_rows: u64,
}

/// Prepares each of the scan's `units` and runs its root, awaiting any root that reads, and
/// returns the work the roots produce.
pub(super) async fn plan(state: &ScanState, units: &[Unit]) -> VortexResult<Planned> {
    let mut work = Vec::new();
    let mut total_rows = 0;
    for unit in units {
        let rows = state.unit_rows(unit)?;
        if rows.is_empty() {
            continue;
        }
        total_rows += rows.end - rows.start;
        let (child, root, _) = prepare(state, unit, rows)?;
        work.extend(run_root(child, root).await?);
    }
    Ok(Planned { work, total_rows })
}

/// Opens the stream of the scan's `units`, whose parts are `planned` when an unordered scan cut
/// them already.
pub(super) fn open(
    state: Arc<ScanState>,
    units: Vec<Unit>,
    planned: Option<Planned>,
) -> VortexResult<ScanStream> {
    let dtype = state.dtype.clone();
    let limit = state.limit.map(RowLimit::new);
    if state.ordered {
        let arrays = Arc::new(Mutex::new(Some(in_order(state, units))));
        return Ok(ScanStream::new(dtype, move || {
            let arrays = arrays
                .lock()
                .take()
                .unwrap_or_else(|| stream::empty().boxed());
            limited(arrays, limit.clone())
        }));
    }

    let (sender, receiver) = async_channel::unbounded();
    let mut total_rows = 0;
    match planned {
        Some(planned) => {
            total_rows = planned.total_rows;
            for work in planned.work {
                send(&sender, work);
            }
        }
        None => {
            for unit in &units {
                let rows = state.unit_rows(unit)?;
                if rows.is_empty() {
                    continue;
                }
                total_rows += rows.end - rows.start;
                let (child, root, dynamic) = prepare(&state, unit, rows)?;
                let parts = run_root(child, root);
                if dynamic {
                    send(&sender, Work::Root(parts, sender.clone()));
                    continue;
                }
                let work = parts
                    .now_or_never()
                    .vortex_expect("a root that only cuts parts reads nothing")?;
                for work in work {
                    send(&sender, work);
                }
            }
        }
    }
    // Only queued roots hold senders now, so the queue closes once it is empty and no root is
    // left that could add to it.
    drop(sender);

    let queue = Arc::new(Queue {
        receiver,
        total_rows,
        rows_taken: AtomicU64::new(0),
    });
    let progress = Arc::clone(&queue);
    Ok(ScanStream::new(dtype, move || {
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
    .with_progress(move || progress.progress()))
}

/// One child's prepared scan, shared by the parts of it that clones run.
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

/// Prepares the scan of `rows` of the unit's child once and composes its root, returning whether
/// the root reads before it spawns parts.
fn prepare(
    state: &ScanState,
    unit: &Unit,
    rows: Range<u64>,
) -> VortexResult<(Arc<PreparedChild>, Box<dyn Planner>, bool)> {
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
    Ok((Arc::new(prepared), root, dynamic && scan.filter().is_some()))
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
fn run_root(
    child: Arc<PreparedChild>,
    root: Box<dyn Planner>,
) -> BoxFuture<'static, VortexResult<Vec<Work>>> {
    async move {
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
    .boxed()
}

/// A unit of work in the queue of an unordered scan.
enum Work {
    Part(Part),
    /// A batch a root returned itself.
    Batch(ArrayRef),
    /// A root that reads before it spawns parts, and the queue to send its parts to. The sender
    /// keeps the queue open until they are in it.
    Root(BoxFuture<'static, VortexResult<Vec<Work>>>, Sender<Work>),
}

fn send(sender: &Sender<Work>, work: Work) {
    // The queue is unbounded and closes only once every clone is dropped, when the work is moot.
    drop(sender.try_send(work));
}

/// The queue of an unordered scan, which every clone of its stream takes from.
struct Queue {
    receiver: Receiver<Work>,
    /// The rows of the scan's units.
    total_rows: u64,
    /// The rows of the parts clones have taken.
    rows_taken: AtomicU64,
}

impl Queue {
    /// The fraction of the scan's rows whose parts clones have taken. Pruning can leave rows that
    /// no part covers, so a queue with nothing left to take has made all of its progress.
    fn progress(&self) -> f64 {
        if self.total_rows == 0 || self.is_drained() {
            return 1.0;
        }
        (self.rows_taken.load(Ordering::Relaxed) as f64 / self.total_rows as f64).min(1.0)
    }

    /// Whether the queue is empty and no root is left that could add to it.
    fn is_drained(&self) -> bool {
        self.receiver.is_empty() && self.receiver.sender_count() == 0
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        // Queued roots hold senders of the queue itself; drain them so it can be freed.
        self.receiver.close();
        while self.receiver.try_recv().is_ok() {}
    }
}

/// One clone's worker: a run of its own, fed one part at a time from the scan's queue, and driven
/// on the task that polls the clone.
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
                    return Ok(None);
                }
                match self.take().await? {
                    None => return Ok(None),
                    Some(Taken::Batch(array)) => return Ok(Some(array)),
                    Some(Taken::Part(part)) => {
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

    /// Takes the next part or batch from the scan's queue, running the roots it finds there
    /// first, or `None` once the queue is closed and empty.
    async fn take(&mut self) -> VortexResult<Option<Taken>> {
        loop {
            let Ok(work) = self.queue.receiver.recv().await else {
                return Ok(None);
            };
            match work {
                Work::Part(part) => {
                    self.queue.rows_taken.fetch_add(
                        part.scope.rows.end - part.scope.rows.start,
                        Ordering::Relaxed,
                    );
                    return Ok(Some(Taken::Part(part)));
                }
                Work::Batch(array) => return Ok(Some(Taken::Batch(array))),
                Work::Root(work, sender) => {
                    for work in work.await? {
                        send(&sender, work);
                    }
                }
            }
        }
    }
}

/// What a worker takes from the queue.
enum Taken {
    Part(Part),
    Batch(ArrayRef),
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
