// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The work that the workers of an opened [`PlanDataSource`](super::PlanDataSource) scan share.
//!
//! An opened scan runs V2's own driver, the planners, morsels and [`Run`] that develop's V2 DuckDB
//! and DataFusion scans run, behind [`ScanPool`]. Nothing here is a general work pool.
//!
//! - An unordered pool runs as develop's V2 DuckDB scan (`vortex-duckdb/src/scan_v2.rs`) does.
//!   Opening it prepares each ready child once, which plans without reading, and runs its root up
//!   front, which spawns its parts: a part is the rows between two of the child's splits for the
//!   fields the scan reads, such as a file's layout splits, else one of the scan's chunks. A root
//!   whose filter changes while the scan runs, as a top-N's bound does, prunes the child as a
//!   whole before it spawns parts; it reads zone maps to do so, so the worker that takes it runs
//!   it instead. Each worker owns a [`Run`], driven on the task that polls the worker, as a DuckDB
//!   thread drives its own run: it takes parts from a queue every worker shares, runs a part's
//!   chunks on its run beside each other, so one chunk's reads overlap another's compute, yields
//!   batches as they are produced, and awaits reads with [`Run::poll_completion`]. Nothing is
//!   spawned per part. Expanding a deferred child, whether a file to open, a manifest entry or a
//!   table partition, is queued work too: the worker that takes it expands it and queues the work
//!   of its children, which may be deferred in turn.
//! - An ordered pool gives every child, in row order, to its first worker, which expands each
//!   deferred child in its place and runs the parts in row order on a run of its own, a part's
//!   chunks beside each other, returning each part's batches in row order. Later workers get
//!   nothing. Parallel ordered reads cut the scan with `split` first.
//!
//! A limit is a [`RowLimit`] the workers take rows from as they yield them; workers stop taking
//! work once it is spent.

use std::future;
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

use super::plan::DeferredPlanChild;
use super::plan::PlanChild;
use super::plan::PlanSourceChild;
use super::plan::ScanState;
use super::plan::Unit;
use crate::execution::planning::driver::Progress;
use crate::execution::planning::driver::Run;
use crate::execution::planning::planner::Planner;
use crate::execution::planning::planner::WorkScope;
use crate::pool::RowLimit;
use crate::pool::ScanPool;
use crate::pool::ScanPoolRef;
use crate::pool::trim_to_limit;
use crate::prepare::PreparedScan;
use crate::prepare::part_root;

/// The parts a worker runs on its run at once when the request's
/// [`split_concurrency`](crate::ScanRequest::split_concurrency) does not say: one, as a DuckDB
/// thread runs one chunk of develop's V2 scan at a time. A part's chunks still run beside each
/// other on the run.
const PARTS_PER_WORKER: usize = 1;

/// One child's prepared scan, shared by the workers running its parts.
struct PreparedChild {
    scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
}

/// A part of a child: the planner its root spawned for some of its rows, which a worker runs on
/// its run.
struct Part {
    child: Arc<PreparedChild>,
    scope: WorkScope,
    planner: Box<dyn Planner>,
    /// The part's share of the pool's work, for its progress.
    share: f64,
}

/// The share of a pool's work that workers have taken, from 0.0 to 1.0.
///
/// Each top-level child of the pool has an equal share. A deferred child divides its share evenly
/// among the children it expands into, and a prepared child divides its share among its parts by
/// their rows. The share of rows that pruning skips counts as taken once they are skipped.
#[derive(Default)]
struct Taken(Mutex<f64>);

impl Taken {
    fn add(&self, share: f64) {
        *self.0.lock() += share;
    }

    fn get(&self) -> f64 {
        self.0.lock().clamp(0.0, 1.0)
    }
}

/// The parts a child's root spawns, in row order.
type PartsFuture = BoxFuture<'static, VortexResult<Vec<Part>>>;

/// A child of a pool, in row order.
enum PoolChild {
    /// A child whose root has run, with the parts it spawned.
    Parts(Vec<Part>),
    /// A child whose root reads before it spawns parts: it prunes the child as a whole first.
    Root(PartsFuture),
    /// A deferred child, and its share of the pool's work.
    Deferred(Arc<dyn DeferredPlanChild>, f64),
}

impl PoolChild {
    /// The child's parts once its root has run.
    fn into_parts(self) -> PartsFuture {
        match self {
            Self::Parts(parts) => future::ready(Ok(parts)).boxed(),
            Self::Root(parts) => parts,
            Self::Deferred(..) => unreachable!("a deferred child has no root"),
        }
    }
}

/// Opens a pool over the scan's `units`.
pub(super) fn open(state: Arc<ScanState>, units: Vec<Unit>) -> VortexResult<ScanPoolRef> {
    let taken = Arc::new(Taken::default());
    let share = 1.0 / units.len().max(1) as f64;
    let dtype = state.dtype.clone();
    let shared = Arc::new(Shared {
        limit: state.limit.map(RowLimit::new),
        parts_per_worker: state
            .split_concurrency
            .map_or(PARTS_PER_WORKER, |parts| parts.get()),
        state,
        taken,
    });
    let mut children = Vec::with_capacity(units.len());
    for unit in units {
        children.push(match shared.state.child(unit.child) {
            PlanSourceChild::Ready(child) => shared.prepare(child, unit.rows, share)?,
            PlanSourceChild::Deferred { child, .. } => {
                PoolChild::Deferred(Arc::clone(child), share)
            }
        });
    }
    if shared.state.ordered {
        return Ok(Arc::new(OrderedPool {
            dtype,
            shared,
            children: Mutex::new(Some(children)),
        }));
    }

    let (sender, receiver) = async_channel::unbounded();
    for child in children {
        queue(&sender, child);
    }
    Ok(Arc::new(UnorderedPool {
        dtype,
        shared,
        receiver,
    }))
}

/// Runs a child's root on a planning run of its own, awaiting only its reads, and returns the
/// parts it spawns, in row order, dividing the child's `share` among them by their rows. The share
/// of the rows they leave out counts as taken.
fn run_root(
    child: Arc<PreparedChild>,
    root: Box<dyn Planner>,
    share: f64,
    taken: Arc<Taken>,
) -> PartsFuture {
    async move {
        let rows = child.scan.rows();
        let per_row = share / (rows.end - rows.start).max(1) as f64;
        let mut unassigned = rows.end - rows.start;
        let mut run = Run::new();
        run.admit(
            root,
            WorkScope {
                file_ordinal: 0,
                rows: child.scan.rows().clone(),
            },
            child.io.session(),
        );
        let mut parts = Vec::new();
        loop {
            match run.advance()? {
                Progress::Spawn(scope, planner) => {
                    let part_rows = scope.rows.end - scope.rows.start;
                    unassigned = unassigned.saturating_sub(part_rows);
                    let share = per_row * part_rows as f64;
                    parts.push(Part {
                        child: Arc::clone(&child),
                        scope,
                        planner,
                        share,
                    });
                }
                Progress::Batch(_) => vortex_bail!("a scan's root produced a batch"),
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => break,
                Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
        if unassigned > 0 {
            taken.add(per_row * unassigned as f64);
        }
        Ok(parts)
    }
    .boxed()
}

/// Runs a part alone on a planning run of its own, awaiting only its reads, and returns its
/// batches in row order.
async fn run_part(part: Part) -> VortexResult<Vec<ArrayRef>> {
    let Part {
        child,
        scope,
        planner,
        ..
    } = part;
    let mut run = Run::new();
    run.admit(planner, scope, child.io.session());
    let mut batches = Vec::new();
    loop {
        match run.advance()? {
            Progress::Batch(batch) => batches.push((batch.scope.rows.start, batch.array)),
            Progress::Spawn(scope, planner) => {
                run.admit(planner, scope, child.io.session());
            }
            Progress::RootDone(_) if run.live_roots() > 0 => {}
            Progress::RootDone(_) | Progress::Idle => break,
            Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
        }
    }
    // A part's chunks finish in whatever order their reads arrive.
    batches.sort_by_key(|(start, _)| *start);
    Ok(batches.into_iter().map(|(_, array)| array).collect())
}

/// What every worker of a pool shares.
struct Shared {
    state: Arc<ScanState>,
    limit: Option<RowLimit>,
    taken: Arc<Taken>,
    /// The parts a worker runs on its run at once.
    parts_per_worker: usize,
}

impl Shared {
    /// Prepares the scan of `rows` of `child`, or of the request's rows when `None`, once, and runs
    /// its root up front, like the splits of a V1 scan, unless the root reads before it spawns
    /// parts.
    ///
    /// Parts are cut on the child's splits for the fields the scan reads, when it has them, else
    /// on the scan's chunks. A filter whose value changes while the scan runs, as a top-N's bound
    /// does, prunes the child as a whole first, so a child it rules out takes no worker.
    fn prepare(
        &self,
        child: &PlanChild,
        rows: Option<Range<u64>>,
        share: f64,
    ) -> VortexResult<PoolChild> {
        let state = &self.state;
        let rows = match rows {
            Some(rows) => rows,
            None => state.child_rows(child)?,
        };
        if rows.is_empty() {
            self.taken.add(share);
            return Ok(PoolChild::Parts(Vec::new()));
        }
        let splits = child.splits(&state.projection, state.filter.as_ref())?;
        let scan = Arc::new(state.builder(child, rows).prepare()?);
        let dynamic = state
            .filter
            .as_ref()
            .is_some_and(|filter| DynamicExprUpdates::new(filter).is_some());
        let root = part_root(&scan, splits.as_deref(), dynamic);
        let io = scan.io();
        let child = Arc::new(PreparedChild { scan, io });
        let parts = run_root(child, root, share, Arc::clone(&self.taken));
        if dynamic {
            return Ok(PoolChild::Root(parts));
        }
        let parts = parts
            .now_or_never()
            .vortex_expect("a root that only cuts parts reads nothing")?;
        Ok(PoolChild::Parts(parts))
    }

    fn is_limit_spent(&self) -> bool {
        self.limit.as_ref().is_some_and(RowLimit::is_exhausted)
    }

    /// The parts of the children that `children` expand into, in row order, expanding each
    /// deferred child only once the parts before it are taken.
    fn parts_in_order(
        self: &Arc<Self>,
        children: Vec<(PlanSourceChild, f64)>,
    ) -> BoxStream<'static, VortexResult<Vec<Part>>> {
        let shared = Arc::clone(self);
        let taken = Arc::clone(&self.taken);
        self.state
            .expand_in_order(children, Arc::new(move |share| taken.add(share)))
            .then(move |child| {
                match child.and_then(|(child, share)| shared.prepare(&child, None, share)) {
                    Ok(child) => child.into_parts(),
                    Err(err) => future::ready(Err(err)).boxed(),
                }
            })
            .boxed()
    }

    /// Expands a deferred child and queues the work of the children it expands into.
    async fn expand(
        &self,
        child: Arc<dyn DeferredPlanChild>,
        share: f64,
        sender: &Sender<Work>,
    ) -> VortexResult<()> {
        let children = child.expand().await?;
        if children.is_empty() {
            self.taken.add(share);
        }
        let share = share / children.len().max(1) as f64;
        for child in children {
            match child {
                PlanSourceChild::Ready(child) => match self.state.admit(child)? {
                    Some(child) => queue(sender, self.prepare(&child, None, share)?),
                    None => self.taken.add(share),
                },
                PlanSourceChild::Deferred { child, .. } => {
                    send(sender, Work::Expand(child, share, sender.clone()));
                }
            }
        }
        Ok(())
    }

    /// A worker's output: `arrays`, trimmed to the limit.
    fn output(&self, arrays: BoxStream<'static, VortexResult<ArrayRef>>) -> SendableArrayStream {
        let arrays = match &self.limit {
            Some(limit) => trim_to_limit(arrays, limit.clone()),
            None => arrays,
        };
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(self.state.dtype.clone(), arrays))
    }
}

/// A unit of an unordered pool.
enum Work {
    Part(Part),
    /// A child's root that reads before it spawns parts, and where to send the parts. The sender
    /// keeps the channel open until they are in it.
    Root(PartsFuture, Sender<Work>),
    /// A deferred child, its share of the pool's work, and where to send the units of its children
    /// once it is expanded.
    Expand(Arc<dyn DeferredPlanChild>, f64, Sender<Work>),
}

fn send(sender: &Sender<Work>, work: Work) {
    // The channel is unbounded and closes only once the pool is dropped, when the work is moot.
    drop(sender.try_send(work));
}

/// Queues the units of `child`.
fn queue(sender: &Sender<Work>, child: PoolChild) {
    match child {
        PoolChild::Parts(parts) => {
            for part in parts {
                send(sender, Work::Part(part));
            }
        }
        PoolChild::Root(parts) => send(sender, Work::Root(parts, sender.clone())),
        PoolChild::Deferred(child, share) => {
            send(sender, Work::Expand(child, share, sender.clone()));
        }
    }
}

/// A pool whose workers share its units through a channel.
struct UnorderedPool {
    dtype: DType,
    shared: Arc<Shared>,
    receiver: Receiver<Work>,
}

impl ScanPool for UnorderedPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let shared = Arc::clone(&self.shared);
        let worker = Worker {
            receiver: self.receiver.clone(),
            shared: Arc::clone(&shared),
            run: Run::new(),
            child: None,
            held: None,
            _pool: self,
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
        shared.output(arrays)
    }

    fn progress(&self) -> f64 {
        if self.is_exhausted() {
            return 1.0;
        }
        self.shared.taken.get()
    }

    fn is_exhausted(&self) -> bool {
        // Roots and expansions hold senders, so the channel closes only once every unit is in it.
        self.shared.is_limit_spent() || (self.receiver.is_closed() && self.receiver.is_empty())
    }
}

impl Drop for UnorderedPool {
    fn drop(&mut self) {
        // Pending roots and expansions hold senders of the channel itself; drain them so it can be
        // freed.
        self.receiver.close();
        while self.receiver.try_recv().is_ok() {}
    }
}

/// One worker of an unordered pool: a run of its own, fed with parts from the pool's queue, and
/// driven on the task that polls the worker.
struct Worker {
    receiver: Receiver<Work>,
    shared: Arc<Shared>,
    run: Run,
    /// The child whose parts are live on the run. A run reads each root through the child's IO,
    /// and a planner a part spawns does not say which part it came from, so a worker runs one
    /// child's parts at a time.
    child: Option<Arc<PreparedChild>>,
    /// A part of another child, taken while the run still held parts of `child`.
    held: Option<Part>,
    /// Holds the pool, so the pool is dropped only after its last worker.
    _pool: Arc<UnorderedPool>,
}

impl Worker {
    /// The worker's next array, or `None` once the pool has no more work for it. Keeps the run fed
    /// with parts, and awaits reads when every live part waits for them.
    async fn next_array(&mut self) -> VortexResult<Option<ArrayRef>> {
        loop {
            while self.run.live_roots() < self.shared.parts_per_worker
                && !self.shared.is_limit_spent()
            {
                let part = match self.held.take() {
                    Some(part) => part,
                    // A worker with nothing live waits for work; one with parts live only takes
                    // work that is already queued.
                    None => match self.take(self.run.live_roots() == 0).await? {
                        Some(part) => part,
                        None => break,
                    },
                };
                if self.run.live_roots() > 0
                    && !self
                        .child
                        .as_ref()
                        .is_some_and(|child| Arc::ptr_eq(child, &part.child))
                {
                    self.held = Some(part);
                    break;
                }
                self.admit(part);
            }
            if self.run.live_roots() == 0 {
                return Ok(None);
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

    /// Admits `part` to the run.
    fn admit(&mut self, part: Part) {
        let Part {
            child,
            scope,
            planner,
            share,
        } = part;
        self.shared.taken.add(share);
        self.run.admit(planner, scope, child.io.session());
        self.child = Some(child);
    }

    /// Takes the next part from the pool's queue, running the roots and expansions it finds there
    /// first, which queue parts of their own. With `wait`, waits for work until the queue closes;
    /// otherwise takes only what is queued already.
    async fn take(&mut self, wait: bool) -> VortexResult<Option<Part>> {
        loop {
            let work = if wait {
                match self.receiver.recv().await {
                    Ok(work) => work,
                    Err(_) => return Ok(None),
                }
            } else {
                match self.receiver.try_recv() {
                    Ok(work) => work,
                    Err(_) => return Ok(None),
                }
            };
            match work {
                Work::Part(part) => return Ok(Some(part)),
                Work::Root(parts, sender) => {
                    for part in parts.await? {
                        send(&sender, Work::Part(part));
                    }
                }
                Work::Expand(child, share, sender) => {
                    self.shared.expand(child, share, &sender).await?;
                }
            }
        }
    }
}

/// A pool that gives all of its children, in row order, to its first worker.
struct OrderedPool {
    dtype: DType,
    shared: Arc<Shared>,
    /// The children, until the first worker takes them.
    children: Mutex<Option<Vec<PoolChild>>>,
}

impl ScanPool for OrderedPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let Some(children) = self.children.lock().take() else {
            return ArrayStreamExt::boxed(ArrayStreamAdapter::new(
                self.dtype.clone(),
                stream::empty(),
            ));
        };
        let expander = Arc::clone(&self.shared);
        let shared = Arc::clone(&self.shared);
        let runner = Arc::clone(&self.shared);
        let arrays = stream::iter(children)
            .flat_map(move |child| match child {
                PoolChild::Deferred(child, share) => expander.parts_in_order(vec![(
                    PlanSourceChild::Deferred {
                        child,
                        byte_size: None,
                    },
                    share,
                )]),
                child => stream::once(child.into_parts()).boxed(),
            })
            .flat_map(|parts| match parts {
                Ok(parts) => stream::iter(parts.into_iter().map(Ok)).left_stream(),
                Err(err) => stream::once(future::ready(Err(err))).right_stream(),
            })
            .take_while(move |_| future::ready(!shared.is_limit_spent()))
            .then(move |part| {
                let runner = Arc::clone(&runner);
                async move {
                    let part = part?;
                    runner.taken.add(part.share);
                    run_part(part).await
                }
            })
            .flat_map(|arrays| match arrays {
                Ok(arrays) => stream::iter(arrays.into_iter().map(Ok)).left_stream(),
                Err(err) => stream::once(future::ready(Err(err))).right_stream(),
            })
            .boxed();
        self.shared.output(arrays)
    }

    fn progress(&self) -> f64 {
        self.shared.taken.get()
    }

    fn is_exhausted(&self) -> bool {
        // The first worker takes every child.
        self.shared.is_limit_spent() || self.children.lock().is_none()
    }
}
