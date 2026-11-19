// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The work that the workers of an opened [`PlanDataSource`](super::PlanDataSource) scan share.
//!
//! This is the V1 layout pool's pattern (`vortex-layout/src/scan/pool.rs`) over the V2 executor,
//! with plan work for units. A pool's units are parts, rows of a prepared child's plan, and
//! deferred children not yet expanded. Opening the scan prepares each ready child once, which
//! plans without reading, and cuts its rows into parts: the child's splits when it has them,
//! else the scan's chunks. A worker takes parts and keeps up to `split_concurrency` of them
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
use std::num::NonZeroUsize;
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
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_io::request::IoService;
use vortex_io::session::RuntimeSessionExt;

use super::plan::DeferredPlanChild;
use super::plan::PlanChild;
use super::plan::PlanSourceChild;
use super::plan::ScanState;
use super::plan::Unit;
use crate::execution::driver::compose;
use crate::execution::planning::driver::Progress;
use crate::execution::planning::driver::Run;
use crate::execution::planning::planner::WorkScope;
use crate::execution::scan_chunks;
use crate::pool::RowLimit;
use crate::pool::SPLITS_IN_FLIGHT_PER_WORKER;
use crate::pool::ScanPool;
use crate::pool::ScanPoolRef;
use crate::pool::trim_to_limit;
use crate::prepare::PreparedScan;

/// One child's prepared scan, shared by the workers running its parts.
struct PreparedChild {
    scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
}

/// A part of a child: rows of its plan that one worker runs on one planning run.
type Part = (Arc<PreparedChild>, Range<u64>);

/// A part's batches, in row order.
type PartFuture = BoxFuture<'static, VortexResult<Vec<ArrayRef>>>;

/// A child of a pool, in row order.
enum PoolChild {
    Prepared(Vec<Part>),
    Deferred(Arc<dyn DeferredPlanChild>),
}

/// Opens a pool over the scan's `units`, preparing each of its ready children.
pub(super) fn open(state: Arc<ScanState>, units: Vec<Unit>) -> VortexResult<ScanPoolRef> {
    let mut children = Vec::with_capacity(units.len());
    for unit in units {
        children.push(match state.child(unit.child) {
            PlanSourceChild::Ready(child) => {
                PoolChild::Prepared(prepare(&state, child, unit.rows)?)
            }
            PlanSourceChild::Deferred { child, .. } => PoolChild::Deferred(Arc::clone(child)),
        });
    }
    let dtype = state.dtype.clone();
    let parts_in_flight = state
        .split_concurrency
        .map_or(SPLITS_IN_FLIGHT_PER_WORKER, NonZeroUsize::get);
    let shared = Arc::new(Shared {
        limit: state.limit.map(RowLimit::new),
        state,
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
        match child {
            PoolChild::Prepared(parts) => {
                for part in parts {
                    send(&sender, Work::Part(part));
                }
            }
            PoolChild::Deferred(child) => send(&sender, Work::Expand(child, sender.clone())),
        }
    }
    Ok(Arc::new(UnorderedPool {
        dtype,
        shared,
        receiver,
        parts_in_flight,
    }))
}

/// Prepares the scan of `rows` of `child`, or of the request's rows when `None`, and cuts them
/// into parts, leaving out parts the selection selects nothing of.
fn prepare(
    state: &ScanState,
    child: &PlanChild,
    rows: Option<Range<u64>>,
) -> VortexResult<Vec<Part>> {
    let rows = match rows {
        Some(rows) => rows,
        None => state.child_rows(child)?,
    };
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let scan = Arc::new(state.builder(child, rows.clone()).prepare()?);
    let ranges = match child.splits() {
        Some(splits) => part_ranges(rows, splits),
        None => scan_chunks(rows, scan.chunk_rows),
    };
    let io = scan.io();
    let prepared = Arc::new(PreparedChild { scan, io });
    Ok(ranges
        .into_iter()
        .filter(|part| state.selection.row_count_in_range(part) > 0)
        .map(|part| (Arc::clone(&prepared), part))
        .collect())
}

/// The ranges between adjacent `splits`, clipped to `rows`, leaving out empty ones.
fn part_ranges(rows: Range<u64>, splits: &[u64]) -> Vec<Range<u64>> {
    splits
        .windows(2)
        .map(|w| w[0].max(rows.start)..w[1].min(rows.end))
        .filter(|part| part.start < part.end)
        .collect()
}

/// Runs a part on a planning run of its own, awaiting only its reads, and returns its batches in
/// row order.
fn run_part((file, rows): Part) -> PartFuture {
    async move {
        let mut run = Run::new();
        run.admit(
            compose(&file.scan, rows.clone()),
            WorkScope {
                file_ordinal: 0,
                rows,
            },
            file.io.session(),
        );
        let mut batches = Vec::new();
        loop {
            match run.advance()? {
                Progress::Batch(batch) => batches.push((batch.scope.rows.start, batch.array)),
                Progress::Spawn(scope, child) => {
                    run.admit(child, scope, file.io.session());
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
}

impl Shared {
    /// Spawns a part onto the session's runtime: another runtime thread may run it while this
    /// worker drives its other parts, and its reads start right away.
    fn spawn_part(&self, part: Part) -> PartFuture {
        self.state.session.handle().spawn(run_part(part)).boxed()
    }

    fn is_exhausted(&self) -> bool {
        self.limit.as_ref().is_some_and(RowLimit::is_exhausted)
    }

    /// The parts of the children that `children` expand into, in row order, expanding each
    /// deferred child only once the parts before it are taken.
    fn parts_in_order(
        self: &Arc<Self>,
        children: Vec<PlanSourceChild>,
    ) -> BoxStream<'static, VortexResult<Vec<Part>>> {
        let shared = Arc::clone(self);
        self.state
            .expand_in_order(children)
            .map(move |child| child.and_then(|child| prepare(&shared.state, &child, None)))
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
    /// A deferred child, and where to send the units of its children once it is expanded. The
    /// sender keeps the channel open until they are in it.
    Expand(Arc<dyn DeferredPlanChild>, Sender<Work>),
}

fn send(sender: &Sender<Work>, work: Work) {
    // The channel is unbounded and closes only once the pool is dropped, when the work is moot.
    drop(sender.try_send(work));
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
            Work::Expand(child, sender) => {
                let shared = Arc::clone(shared);
                async move {
                    for child in child.expand().await? {
                        match child {
                            PlanSourceChild::Ready(child) => {
                                let Some(child) = shared.state.admit(child)? else {
                                    continue;
                                };
                                for part in prepare(&shared.state, &child, None)? {
                                    send(&sender, Work::Part(part));
                                }
                            }
                            PlanSourceChild::Deferred { child, .. } => {
                                send(&sender, Work::Expand(child, sender.clone()));
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
            .take_while(move |_| future::ready(!self.shared.is_exhausted()))
            .map(move |work| UnorderedPool::run(&runner, work))
            .buffer_unordered(parts_in_flight)
            .boxed();
        shared.output(outputs)
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
                PoolChild::Prepared(parts) => stream::iter([Ok(parts)]).boxed(),
                PoolChild::Deferred(child) => {
                    expander.parts_in_order(vec![PlanSourceChild::Deferred {
                        child,
                        byte_size: None,
                    }])
                }
            })
            .flat_map(|parts| match parts {
                Ok(parts) => stream::iter(parts.into_iter().map(Ok)).left_stream(),
                Err(err) => stream::once(future::ready(Err(err))).right_stream(),
            })
            .take_while(move |_| future::ready(!shared.is_exhausted()))
            .map(move |part| match part {
                Ok(part) => spawner.spawn_part(part),
                Err(err) => future::ready(Err(err)).boxed(),
            })
            .buffered(self.parts_in_flight)
            .boxed();
        self.shared.output(outputs)
    }
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
