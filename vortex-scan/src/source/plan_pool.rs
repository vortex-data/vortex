// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The pool of an opened [`PlanDataSource`](super::PlanDataSource) scan: V2's planners and
//! [`Run`] driver behind [`ScanPool`], in the shape of the V2 DuckDB scan
//! (`vortex-duckdb/src/scan_v2.rs`).
//!
//! An unordered scan prepares each child once and runs its [`part_root`], which cuts the child
//! into parts: the rows between the child's splits for the fields the scan reads, such as a
//! file's layout splits, or the scan's chunks for a child without splits. The parts go into one
//! queue that the pool's workers share. Each worker owns a [`Run`], driven on the task that polls
//! the worker, and runs one part on it at a time: the part's chunks run side by side, so one
//! chunk's reads overlap another's compute, batches are returned as they are produced, and reads
//! are awaited with [`Run::poll_completion`]. Nothing is spawned.
//!
//! A deferred child is queued work as well: the worker that takes it expands it and queues the
//! work of its children, pruned by the scan's filter, so files open in parallel and only once a
//! worker is free to read them. The queue stays open while any queued root or expansion may still
//! add work to it.
//!
//! With a dynamic filter, such as the bound a top-N pushes down, a child's root prunes the child as
//! a whole before it cuts parts, which reads zone maps. Opening reads nothing, so such a root is
//! queued work too: the first worker to take it runs it and queues its parts.
//!
//! An ordered scan returns its rows in order through the pool's first worker, running one child
//! after another on [`ScanDriver`], which keeps a scan's chunks in order. It expands a deferred
//! child once the children before it are read. A host that reads an
//! ordered scan in parallel [splits](crate::DataSourceScan::split) it first.
//!
//! A limit is one [`RowLimit`] for the whole pool: workers stop taking parts once it is spent, and
//! each worker's output is trimmed to it.

use std::future;
use std::iter;
use std::ops::Range;
use std::sync::Arc;

use async_channel::Receiver;
use async_channel::Sender;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::request::IoService;
use vortex_io::session::RuntimeSessionExt;

use super::plan::DeferredPlanChild;
use super::plan::PlanChild;
use super::plan::PlanSourceChild;
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
use crate::pool::ScanPool;
use crate::pool::ScanPoolRef;
use crate::prepare::PreparedScan;
use crate::prepare::part_root;

/// Opens a pool over the scan's `units`.
pub(super) fn open(state: Arc<ScanState>, units: Vec<Unit>) -> VortexResult<ScanPoolRef> {
    let dtype = state.dtype.clone();
    let limit = state.limit.map(RowLimit::new);
    if state.ordered {
        return Ok(Arc::new(OrderedPool {
            dtype,
            limit,
            arrays: Mutex::new(Some(in_order(state, units))),
        }));
    }

    let (sender, receiver) = async_channel::unbounded();
    for unit in units {
        match state.child(unit.child) {
            PlanSourceChild::Ready(child) => queue_child(&state, &sender, child, unit.rows)?,
            PlanSourceChild::Deferred { child, .. } => {
                send(&sender, Work::Expand(Arc::clone(child), sender.clone()));
            }
        }
    }
    Ok(Arc::new(UnorderedPool {
        dtype,
        state,
        limit,
        receiver,
    }))
}

/// Prepares this scan's `rows` of `child` and queues its work: its parts, or its root when the
/// root reads before it cuts parts.
fn queue_child(
    state: &ScanState,
    sender: &Sender<Work>,
    child: &PlanChild,
    rows: Option<Range<u64>>,
) -> VortexResult<()> {
    let rows = state.rows_of(child, rows)?;
    if rows.is_empty() {
        return Ok(());
    }
    let (prepared, root, dynamic) = prepare(state, child, rows)?;
    let parts = run_root(prepared, root);
    if dynamic {
        send(sender, Work::Root(parts, sender.clone()));
        return Ok(());
    }
    let parts = parts
        .now_or_never()
        .vortex_expect("a root that only cuts parts reads nothing")?;
    for part in parts {
        send(sender, Work::Part(part));
    }
    Ok(())
}

/// One child's prepared scan, shared by the parts of it that workers run.
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

/// Prepares the scan of `rows` of `child` once and composes its root, returning whether the root
/// reads before it spawns parts.
fn prepare(
    state: &ScanState,
    child: &PlanChild,
    rows: Range<u64>,
) -> VortexResult<(Arc<PreparedChild>, Box<dyn Planner>, bool)> {
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

/// Runs a child's root on a run of its own, awaiting only its reads, and returns the parts it
/// spawns, in row order.
fn run_root(
    child: Arc<PreparedChild>,
    root: Box<dyn Planner>,
) -> BoxFuture<'static, VortexResult<Vec<Part>>> {
    async move {
        let mut run = Run::new();
        let scope = WorkScope {
            file_ordinal: 0,
            rows: child.rows.clone(),
        };
        run.admit(root, scope, child.io.session());
        let mut parts = Vec::new();
        loop {
            match run.advance()? {
                Progress::Spawn(scope, planner) => parts.push(Part {
                    child: Arc::clone(&child),
                    scope,
                    planner,
                }),
                Progress::Batch(_) => vortex_bail!("a scan's root produced a batch"),
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => break,
                Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
        Ok(parts)
    }
    .boxed()
}

/// A unit of work in an unordered pool's queue.
enum Work {
    Part(Part),
    /// A root that reads before it spawns parts, and the queue to send its parts to. The sender
    /// keeps the queue open until they are in it.
    Root(BoxFuture<'static, VortexResult<Vec<Part>>>, Sender<Work>),
    /// A deferred child, and the queue to send the work of its children to.
    Expand(Arc<dyn DeferredPlanChild>, Sender<Work>),
}

fn send(sender: &Sender<Work>, work: Work) {
    // The queue is unbounded and closes only once the pool is dropped, when the work is moot.
    drop(sender.try_send(work));
}

/// A pool whose workers take parts from one queue.
struct UnorderedPool {
    dtype: DType,
    state: Arc<ScanState>,
    limit: Option<RowLimit>,
    receiver: Receiver<Work>,
}

impl ScanPool for UnorderedPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let dtype = self.dtype.clone();
        let limit = self.limit.clone();
        let worker = Worker {
            receiver: self.receiver.clone(),
            limit: self.limit.clone(),
            run: Run::new(),
            child: None,
            pool: self,
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
        output(dtype, arrays, limit)
    }
}

impl Drop for UnorderedPool {
    fn drop(&mut self) {
        // Queued roots and expansions hold senders of the queue itself; drain them so it can be freed.
        self.receiver.close();
        while self.receiver.try_recv().is_ok() {}
    }
}

/// One worker of an unordered pool: a run of its own, fed one part at a time from the pool's
/// queue, and driven on the task that polls the worker.
struct Worker {
    receiver: Receiver<Work>,
    limit: Option<RowLimit>,
    run: Run,
    /// The child of the part live on the run, whose I/O the planners that part spawns read through.
    child: Option<Arc<PreparedChild>>,
    /// The pool, dropped only after its last worker.
    pool: Arc<UnorderedPool>,
}

impl Worker {
    /// The worker's next array, or `None` once the pool has no more work for it.
    async fn next_array(&mut self) -> VortexResult<Option<ArrayRef>> {
        loop {
            if self.run.live_roots() == 0 {
                if self.limit.as_ref().is_some_and(RowLimit::is_exhausted) {
                    return Ok(None);
                }
                let Some(part) = self.take().await? else {
                    return Ok(None);
                };
                self.run
                    .admit(part.planner, part.scope, part.child.io.session());
                self.child = Some(part.child);
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

    /// Takes the next part from the pool's queue, running the roots and expanding the children it
    /// finds there first, or
    /// `None` once the queue is closed and empty.
    async fn take(&mut self) -> VortexResult<Option<Part>> {
        loop {
            let Ok(work) = self.receiver.recv().await else {
                return Ok(None);
            };
            match work {
                Work::Part(part) => return Ok(Some(part)),
                Work::Root(parts, sender) => {
                    for part in parts.await? {
                        send(&sender, Work::Part(part));
                    }
                }
                Work::Expand(child, sender) => {
                    for child in child.expand().await? {
                        match child {
                            PlanSourceChild::Ready(child) => {
                                if let Some(child) = self.pool.state.admit(child)? {
                                    queue_child(&self.pool.state, &sender, &child, None)?;
                                }
                            }
                            PlanSourceChild::Deferred { child, .. } => {
                                send(&sender, Work::Expand(child, sender.clone()));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The arrays of an ordered scan: each unit's rows in order, one child after another, with a
/// deferred child's children in its place.
fn in_order(state: Arc<ScanState>, units: Vec<Unit>) -> BoxStream<'static, VortexResult<ArrayRef>> {
    let children = units
        .into_iter()
        .map(|unit| (state.child(unit.child).clone(), unit.rows))
        .collect();
    let children = state.expand_in_order(children);
    children
        .flat_map(move |child| {
            let rows = child.and_then(|(child, rows)| Ok((state.rows_of(&child, rows)?, child)));
            let (rows, child) = match rows {
                Ok(rows) => rows,
                Err(err) => return stream::once(future::ready(Err(err))).boxed(),
            };
            if rows.is_empty() {
                return stream::empty().boxed();
            }
            ScanDriver::new(state.session.handle(), 1, true).run(state.builder(&child, rows))
        })
        .boxed()
}

/// A pool that gives the whole scan, in order, to its first worker. Later workers return nothing.
struct OrderedPool {
    dtype: DType,
    limit: Option<RowLimit>,
    arrays: Mutex<Option<BoxStream<'static, VortexResult<ArrayRef>>>>,
}

impl ScanPool for OrderedPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let arrays = self
            .arrays
            .lock()
            .take()
            .unwrap_or_else(|| stream::empty().boxed());
        output(self.dtype.clone(), arrays, self.limit.clone())
    }
}

/// A worker's output: `arrays`, trimmed to the pool's limit.
fn output(
    dtype: DType,
    arrays: BoxStream<'static, VortexResult<ArrayRef>>,
    limit: Option<RowLimit>,
) -> SendableArrayStream {
    let arrays = match limit {
        Some(limit) => limit_rows(arrays, limit),
        None => arrays,
    };
    ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays))
}
