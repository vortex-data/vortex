// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The work that the workers of an opened [`PlanDataSource`](super::PlanDataSource) scan share.
//!
//! This is the V1 layout pool's pattern (`vortex-layout/src/scan/pool.rs`) over the V2 executor,
//! with plan work for units. A pool's units are parts, rows of a prepared child's plan, and
//! deferred children not yet expanded. Opening the scan prepares each ready child once, which
//! plans without reading, and runs its root up front, which spawns its parts, as develop's V2
//! DuckDB scan does: a part is the rows between two of the child's splits for the fields the scan
//! reads, such as a file's layout splits, else one of the scan's chunks. A root whose filter
//! changes while the scan runs, as a top-N's bound does, prunes the child as a whole before it
//! spawns parts; it reads zone maps to do so, so it runs as a unit of work instead. A worker takes parts and keeps up to `split_concurrency` of them
//! in flight, each spawned onto the session's runtime when the worker takes it, as the V1 pool's
//! splits are, and driven there on a planning run of its own that awaits only its reads: with a
//! filter, the part's pruning plans first, continuing into the chunks of the rows pruning kept;
//! without, the chunks directly.
//!
//! Work creates work: expanding a deferred child, whether a file to open, a manifest entry or a
//! table partition, is a unit like any other, and queues the work of the children it expands
//! into, which may be deferred in turn.
//!
//! - An unordered pool hands units out through a channel. The worker that expands a deferred
//!   child sends its children's units back into the channel, where any worker can take them. The
//!   channel closes once every expansion has finished, which ends the workers.
//! - An ordered pool gives all of its units, in row order, to its first worker, which expands
//!   each deferred child in its place; later workers get nothing. Parallel ordered reads cut the
//!   scan with `split` first.
//!
//! A limit is a [`RowLimit`] the workers take rows from as they yield them; workers stop taking
//! units once it is spent.

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
use vortex_io::session::RuntimeSessionExt;
use vortex_utils::parallelism::get_available_parallelism;

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
use crate::pool::SPLITS_IN_FLIGHT_PER_WORKER;
use crate::pool::ScanPool;
use crate::pool::ScanPoolRef;
use crate::pool::trim_to_limit;
use crate::prepare::PreparedScan;
use crate::prepare::part_root;

/// One child's prepared scan, shared by the workers running its parts.
struct PreparedChild {
    scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
}

/// A part of a child: the planner its root spawned for some of its rows, which one worker runs on
/// a planning run of its own.
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

/// A part's batches, in row order.
type PartFuture = BoxFuture<'static, VortexResult<Vec<ArrayRef>>>;

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

/// Opens a pool over the scan's `units`, preparing each of its ready children.
pub(super) fn open(state: Arc<ScanState>, units: Vec<Unit>) -> VortexResult<ScanPoolRef> {
    let taken = Arc::new(Taken::default());
    let share = 1.0 / units.len().max(1) as f64;
    let mut children = Vec::with_capacity(units.len());
    for unit in units {
        children.push(match state.child(unit.child) {
            PlanSourceChild::Ready(child) => prepare(&state, child, unit.rows, share, &taken)?,
            PlanSourceChild::Deferred { child, .. } => {
                PoolChild::Deferred(Arc::clone(child), share)
            }
        });
    }
    let dtype = state.dtype.clone();
    // A worker takes the place of a partition that one host thread drains, so, like the layout
    // pool's, it keeps as many parts in flight as a partition's stream does by default: four per
    // core, at most the request's `split_concurrency`.
    let scaled =
        SPLITS_IN_FLIGHT_PER_WORKER.saturating_mul(get_available_parallelism().unwrap_or(1));
    let parts_in_flight = state
        .split_concurrency
        .map_or(scaled, |max| scaled.min(max.get()))
        .max(1);
    let shared = Arc::new(Shared {
        limit: state.limit.map(RowLimit::new),
        state,
        taken,
    });
    if shared.state.ordered {
        return Ok(Arc::new(OrderedPool {
            dtype,
            shared,
            children: Mutex::new(Some(children)),
            parts_in_flight,
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
        parts_in_flight,
    }))
}

/// Prepares the scan of `rows` of `child`, or of the request's rows when `None`, once, and runs its
/// root up front, like the splits of a V1 scan, unless the root reads before it spawns parts.
///
/// Parts are cut on the child's splits for the fields the scan reads, when it has them, else on
/// the scan's chunks. A filter whose value changes while the scan runs, as a top-N's bound does,
/// prunes the child as a whole first, so a child it rules out takes no worker.
fn prepare(
    state: &ScanState,
    child: &PlanChild,
    rows: Option<Range<u64>>,
    share: f64,
    taken: &Arc<Taken>,
) -> VortexResult<PoolChild> {
    let rows = match rows {
        Some(rows) => rows,
        None => state.child_rows(child)?,
    };
    if rows.is_empty() {
        taken.add(share);
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
    let parts = run_root(child, root, share, Arc::clone(taken));
    if dynamic {
        return Ok(PoolChild::Root(parts));
    }
    let parts = parts
        .now_or_never()
        .vortex_expect("a root that only cuts parts reads nothing")?;
    Ok(PoolChild::Parts(parts))
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

/// Runs a part on a planning run of its own, awaiting only its reads, and returns its batches in
/// row order.
fn run_part(part: Part) -> PartFuture {
    async move {
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
    .boxed()
}

/// What every worker of a pool shares.
struct Shared {
    state: Arc<ScanState>,
    limit: Option<RowLimit>,
    taken: Arc<Taken>,
}

impl Shared {
    /// Spawns a part onto the session's runtime: another runtime thread may run it while this
    /// worker drives its other parts, and its reads start right away.
    fn spawn_part(&self, part: Part) -> PartFuture {
        self.taken.add(part.share);
        self.state.session.handle().spawn(run_part(part)).boxed()
    }

    fn prepare(&self, child: &PlanChild, share: f64) -> VortexResult<PoolChild> {
        prepare(&self.state, child, None, share, &self.taken)
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
                match child.and_then(|(child, share)| shared.prepare(&child, share)) {
                    Ok(child) => child.into_parts(),
                    Err(err) => future::ready(Err(err)).boxed(),
                }
            })
            .boxed()
    }

    /// A worker's output: the batches of its parts, trimmed to the limit, ending after the first
    /// error.
    fn output(
        &self,
        outputs: BoxStream<'static, VortexResult<Vec<ArrayRef>>>,
    ) -> SendableArrayStream {
        let arrays = outputs
            .scan(false, |failed, output| {
                if *failed {
                    return future::ready(None);
                }
                *failed = output.is_err();
                future::ready(Some(output))
            })
            .flat_map(|arrays| match arrays {
                Ok(arrays) => stream::iter(arrays.into_iter().map(Ok)).left_stream(),
                Err(err) => stream::once(future::ready(Err(err))).right_stream(),
            })
            .boxed();
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
    parts_in_flight: usize,
}

impl UnorderedPool {
    fn run(shared: &Arc<Shared>, work: Work) -> PartFuture {
        match work {
            Work::Part(part) => shared.spawn_part(part),
            Work::Root(parts, sender) => async move {
                for part in parts.await? {
                    send(&sender, Work::Part(part));
                }
                Ok(Vec::new())
            }
            .boxed(),
            Work::Expand(child, share, sender) => {
                let shared = Arc::clone(shared);
                async move {
                    let children = child.expand().await?;
                    if children.is_empty() {
                        shared.taken.add(share);
                    }
                    let share = share / children.len().max(1) as f64;
                    for child in children {
                        match child {
                            PlanSourceChild::Ready(child) => match shared.state.admit(child)? {
                                Some(child) => queue(&sender, shared.prepare(&child, share)?),
                                None => shared.taken.add(share),
                            },
                            PlanSourceChild::Deferred { child, .. } => {
                                send(&sender, Work::Expand(child, share, sender.clone()));
                            }
                        }
                    }
                    Ok(Vec::new())
                }
                .boxed()
            }
        }
    }
}

impl ScanPool for UnorderedPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let shared = Arc::clone(&self.shared);
        let runner = Arc::clone(&shared);
        let receiver = self.receiver.clone();
        let parts_in_flight = self.parts_in_flight;
        // The worker holds the pool, so the pool is dropped only after its last worker.
        let outputs = receiver
            .take_while(move |_| future::ready(!self.shared.is_limit_spent()))
            .map(move |work| UnorderedPool::run(&runner, work))
            .buffer_unordered(parts_in_flight)
            .boxed();
        shared.output(outputs)
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
        // Pending opens hold senders of the channel itself; drain them so it can be freed.
        self.receiver.close();
        while self.receiver.try_recv().is_ok() {}
    }
}

/// A pool that gives all of its units, in row order, to its first worker.
struct OrderedPool {
    dtype: DType,
    shared: Arc<Shared>,
    /// The children, until the first worker takes them.
    children: Mutex<Option<Vec<PoolChild>>>,
    parts_in_flight: usize,
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
        let spawner = Arc::clone(&self.shared);
        let shared = Arc::clone(&self.shared);
        let outputs = stream::iter(children)
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
            .map(move |part| match part {
                Ok(part) => spawner.spawn_part(part),
                Err(err) => future::ready(Err(err)).boxed(),
            })
            .buffered(self.parts_in_flight)
            .boxed();
        self.shared.output(outputs)
    }

    fn progress(&self) -> f64 {
        self.shared.taken.get()
    }

    fn is_exhausted(&self) -> bool {
        // The first worker takes every child.
        self.shared.is_limit_spent() || self.children.lock().is_none()
    }
}
