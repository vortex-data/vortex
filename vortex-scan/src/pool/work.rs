// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The one pool that every scan engine's work runs through.
//!
//! A source opens a pool over [`WorkUnit`]s, in row order. A unit either runs, returning arrays,
//! or expands into further units in its place, which may expand in turn: work creates work. A
//! deferred file to open, a table format's manifest entry, and a scan root that prunes before it
//! cuts parts are all expansions. The source decides what a unit does and where it runs, spawning
//! it onto the session's runtime if it likes; the pool decides who takes which unit, how many run
//! at once, and when the scan's row limit stops the workers.
//!
//! - An unordered pool hands units out through a channel. The worker that expands a unit sends
//!   the units it yields back into the channel, where any worker can take them. At most
//!   `expand_concurrency` expansions are queued or running at once; each holds a slot that it
//!   passes on to the next waiting expansion when it is done, or when its worker drops it. The
//!   channel closes once every expansion has finished, which ends the workers.
//! - An ordered pool gives all of its units, in row order, to its first worker, which expands each
//!   unit in its place, `expand_concurrency` top-level expansions ahead; later workers get
//!   nothing. Parallel ordered reads cut the scan with `split` first.
//!
//! Each worker keeps up to its in-flight bound of units running at once: four per core, at most
//! the request's `split_concurrency`, the in-flight total of a partition's stream.

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::task::Poll;

use async_channel::Receiver;
use async_channel::Sender;
use futures::FutureExt;
use futures::StreamExt;
use futures::future;
use futures::future::BoxFuture;
use futures::ready;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_utils::parallelism::get_available_parallelism;

use super::RowLimit;
use super::SPLITS_IN_FLIGHT_PER_WORKER;
use super::ScanPool;
use super::ScanPoolRef;

/// The arrays a unit returns, in row order.
pub type UnitOutput = BoxFuture<'static, VortexResult<Vec<ArrayRef>>>;

/// The units an expansion yields, in row order, in its place.
pub type Expansion = BoxFuture<'static, VortexResult<Vec<WorkUnit>>>;

/// Maps each array a worker returns, as a future that the worker runs `in flight` at a time.
pub type MapOutput = Arc<dyn Fn(ArrayRef) -> MappedArray + Send + Sync>;

/// One mapped array.
pub type MappedArray = BoxFuture<'static, VortexResult<ArrayRef>>;

/// A unit of a pool's work, and its share of the pool's work for progress.
///
/// Shares are the source's to give: the shares of a pool's units, and of the units an expansion
/// yields, sum to the share of what they stand for. A unit's share counts as taken when a worker
/// takes it. A part of the work that the source learns has nothing to read is a
/// [`skipped`](Self::skipped) unit, so its share is still taken.
pub struct WorkUnit {
    kind: UnitKind,
    share: f64,
}

enum UnitKind {
    /// Called once a worker takes the unit.
    Run(Box<dyn FnOnce() -> UnitOutput + Send>),
    /// Called once the unit is expanded.
    Expand(Box<dyn FnOnce() -> Expansion + Send>),
}

impl WorkUnit {
    /// A unit that `run` starts once a worker takes it.
    pub fn run(share: f64, run: impl FnOnce() -> UnitOutput + Send + 'static) -> Self {
        Self {
            kind: UnitKind::Run(Box::new(run)),
            share,
        }
    }

    /// A unit that `expand` turns into further units, in its place, once it is reached.
    pub fn expand(share: f64, expand: impl FnOnce() -> Expansion + Send + 'static) -> Self {
        Self {
            kind: UnitKind::Expand(Box::new(expand)),
            share,
        }
    }

    /// A unit with nothing to read, whose share is taken once a worker reaches it.
    pub fn skipped(share: f64) -> Self {
        Self::run(share, || future::ready(Ok(Vec::new())).boxed())
    }

    fn is_expand(&self) -> bool {
        matches!(self.kind, UnitKind::Expand(_))
    }
}

/// How a pool applies the scan's row limit.
#[derive(Clone)]
pub enum PoolLimit {
    /// The scan has no limit.
    Unlimited,
    /// The units reserve their rows from the budget before they read them, so their output needs
    /// no trimming. Workers keep taking units once it is spent, which then return nothing, but
    /// expansions not yet started are skipped.
    Reserve(RowLimit),
    /// A unit's output rows are unknown until it has run, so workers take rows from the budget as
    /// they return them, discard the excess, and end once it is spent.
    Trim(RowLimit),
}

impl PoolLimit {
    /// Whether the budget is spent, so no expansion that has not started has rows to return.
    fn is_spent(&self) -> bool {
        match self {
            PoolLimit::Unlimited => false,
            PoolLimit::Reserve(limit) | PoolLimit::Trim(limit) => limit.is_exhausted(),
        }
    }

    /// Whether a trimmed budget is spent, so workers end without returning any more rows.
    fn is_trimmed_out(&self) -> bool {
        matches!(self, PoolLimit::Trim(limit) if limit.is_exhausted())
    }
}

/// How a pool shares its units among its workers.
pub struct WorkPoolOptions {
    /// The dtype of every array the workers return.
    pub dtype: DType,
    /// Whether the scan returns rows in order, so the first worker takes every unit.
    pub ordered: bool,
    /// The scan's row limit, and how it is applied.
    pub limit: PoolLimit,
    /// The request's bound on the units each worker keeps in flight.
    pub split_concurrency: Option<NonZeroUsize>,
    /// The most expansions that are queued or running at once.
    pub expand_concurrency: NonZeroUsize,
    /// Maps every array the workers return under a [trimmed](PoolLimit::Trim) limit, such as to
    /// optimize the array that the limit sliced.
    pub map_trimmed: Option<MapOutput>,
}

/// Opens a pool over `units`, in row order.
pub fn open_work_pool(options: WorkPoolOptions, units: Vec<WorkUnit>) -> ScanPoolRef {
    let scaled =
        SPLITS_IN_FLIGHT_PER_WORKER.saturating_mul(get_available_parallelism().unwrap_or(1));
    let in_flight = options
        .split_concurrency
        .map_or(scaled, |max| scaled.min(max.get()))
        .max(1);
    let expand_concurrency = options.expand_concurrency.get();
    let shared = Arc::new(Shared {
        limit: options.limit,
        map_trimmed: options.map_trimmed,
        in_flight,
        taken: Mutex::new(0.0),
        slots: Mutex::new(Slots {
            free: expand_concurrency,
            waiting: VecDeque::new(),
        }),
    });
    if options.ordered {
        return Arc::new(OrderedPool {
            dtype: options.dtype,
            shared,
            units: Mutex::new(Some(units)),
            expand_concurrency,
        });
    }

    let (sender, receiver) = async_channel::unbounded();
    shared.queue(&sender, units);
    Arc::new(UnorderedPool {
        dtype: options.dtype,
        shared,
        receiver,
    })
}

/// What every worker of a pool shares.
struct Shared {
    limit: PoolLimit,
    map_trimmed: Option<MapOutput>,
    /// The units each worker keeps in flight.
    in_flight: usize,
    /// The share of the pool's work that workers have taken.
    taken: Mutex<f64>,
    /// An unordered pool's expansion slots.
    slots: Mutex<Slots>,
}

/// An unordered pool's expansion slots: how many are free, and the expansions that wait for one,
/// in row order.
struct Slots {
    free: usize,
    waiting: VecDeque<WorkUnit>,
}

impl Shared {
    fn take(&self, share: f64) {
        *self.taken.lock() += share;
    }

    /// Takes a unit that runs, and starts it.
    fn run(&self, unit: WorkUnit) -> UnitOutput {
        self.take(unit.share);
        match unit.kind {
            UnitKind::Run(run) => run(),
            UnitKind::Expand(_) => unreachable!("expansions are not run"),
        }
    }

    /// Expands a unit, or skips it once the limit is spent.
    fn expand(&self, unit: WorkUnit) -> Expansion {
        if self.limit.is_spent() {
            self.take(unit.share);
            return future::ready(Ok(Vec::new())).boxed();
        }
        match unit.kind {
            UnitKind::Expand(expand) => expand(),
            UnitKind::Run(_) => unreachable!("only expansions expand"),
        }
    }

    /// Queues `units` into an unordered pool's channel, in order. An expansion takes a free slot,
    /// or waits for one behind the expansions already waiting.
    fn queue(self: &Arc<Self>, sender: &Sender<Work>, units: Vec<WorkUnit>) {
        for unit in units {
            if !unit.is_expand() {
                send(sender, Work::Unit(unit));
                continue;
            }
            let mut slots = self.slots.lock();
            if slots.free > 0 {
                slots.free -= 1;
                drop(slots);
                send(sender, Work::Expand(unit, Slot::new(self, sender)));
            } else {
                slots.waiting.push_back(unit);
            }
        }
    }

    /// The expansion that a freed slot passes to, if any waits and the limit is not spent. Once it
    /// is spent, the expansions left count as taken. Without one, the slot is free again.
    fn pass_slot(&self) -> Option<WorkUnit> {
        let mut slots = self.slots.lock();
        if self.limit.is_spent() && !slots.waiting.is_empty() {
            let share: f64 = slots.waiting.iter().map(|unit| unit.share).sum();
            self.take(share);
            slots.waiting.clear();
        }
        let next = slots.waiting.pop_front();
        if next.is_none() {
            slots.free += 1;
        }
        next
    }

    fn progress(&self) -> f64 {
        if self.limit.is_trimmed_out() {
            return 1.0;
        }
        self.taken.lock().clamp(0.0, 1.0)
    }

    /// A worker's output: the arrays of its units, trimmed to a trimmed limit and mapped, ending
    /// after the first error or once a trimmed limit is spent.
    fn output(
        self: Arc<Self>,
        dtype: DType,
        outputs: BoxStream<'static, VortexResult<Vec<ArrayRef>>>,
        ordered: bool,
    ) -> SendableArrayStream {
        let arrays = stop_on_error(outputs)
            .flat_map(|arrays| match arrays {
                Ok(arrays) => stream::iter(arrays.into_iter().map(Ok)).left_stream(),
                Err(err) => stream::once(future::ready(Err(err))).right_stream(),
            })
            .boxed();
        let arrays = match &self.limit {
            PoolLimit::Trim(limit) => trim(arrays, limit.clone()),
            PoolLimit::Unlimited | PoolLimit::Reserve(_) => arrays,
        };
        let arrays = match (&self.limit, &self.map_trimmed) {
            (PoolLimit::Trim(_), Some(map)) => {
                let map = Arc::clone(map);
                let mapped = arrays.map(move |array| match array {
                    Ok(array) => map(array),
                    Err(err) => future::ready(Err(err)).boxed(),
                });
                let mapped = if ordered {
                    mapped.buffered(self.in_flight).boxed()
                } else {
                    mapped.buffer_unordered(self.in_flight).boxed()
                };
                stop_on_error(mapped)
            }
            _ => arrays,
        };
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays))
    }
}

/// Ends `arrays` once `limit` is spent, taking each array's rows from it as the array is yielded
/// and slicing the array that spends it. Leaves out empty arrays, and ends after the first error.
fn trim(
    arrays: BoxStream<'static, VortexResult<ArrayRef>>,
    limit: RowLimit,
) -> BoxStream<'static, VortexResult<ArrayRef>> {
    super::trim_to_limit(arrays, limit)
}

/// Ends `stream` after its first error.
fn stop_on_error<T: Send + 'static>(
    stream: BoxStream<'static, VortexResult<T>>,
) -> BoxStream<'static, VortexResult<T>> {
    let mut stream = Some(stream);
    stream::poll_fn(move |cx| {
        let Some(input) = stream.as_mut() else {
            return Poll::Ready(None);
        };
        let item = ready!(input.poll_next_unpin(cx));
        if !matches!(item, Some(Ok(_))) {
            stream = None;
        }
        Poll::Ready(item)
    })
    .boxed()
}

/// A unit of an unordered pool's channel.
enum Work {
    Unit(WorkUnit),
    /// An expansion, and the slot it holds until its units are in the channel.
    Expand(WorkUnit, Slot),
}

/// One of an unordered pool's `expand_concurrency` expansion slots. Its sender keeps the channel
/// open while the expansion that holds it is queued or running.
struct Slot {
    shared: Arc<Shared>,
    sender: Sender<Work>,
}

impl Slot {
    fn new(shared: &Arc<Shared>, sender: &Sender<Work>) -> Self {
        Self {
            shared: Arc::clone(shared),
            sender: sender.clone(),
        }
    }
}

impl Drop for Slot {
    /// Passes the slot to the next waiting expansion, queued behind the units of the expansion
    /// that held it, so expansions keep pace with the workers. The slot passes on whether the
    /// expansion finished or a dropped worker gave it up, so that worker strands only the
    /// expansion it held.
    fn drop(&mut self) {
        if self.sender.is_closed() {
            return;
        }
        if let Some(next) = self.shared.pass_slot() {
            let slot = Slot::new(&self.shared, &self.sender);
            send(&self.sender, Work::Expand(next, slot));
        }
    }
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
}

impl UnorderedPool {
    fn run(shared: &Arc<Shared>, work: Work) -> UnitOutput {
        match work {
            Work::Unit(unit) => shared.run(unit),
            Work::Expand(unit, slot) => {
                let expansion = shared.expand(unit);
                async move {
                    let expanded = expansion.await.map(|units| {
                        slot.shared.queue(&slot.sender, units);
                        Vec::new()
                    });
                    drop(slot);
                    expanded
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
        let dtype = self.dtype.clone();
        let shared = Arc::clone(&self.shared);
        let runner = Arc::clone(&shared);
        let receiver = self.receiver.clone();
        let in_flight = shared.in_flight;
        // The worker holds the pool, so the pool is dropped only after its last worker.
        let outputs = receiver
            .take_while(move |_| future::ready(!self.shared.limit.is_trimmed_out()))
            .map(move |work| UnorderedPool::run(&runner, work))
            .buffer_unordered(in_flight)
            .boxed();
        shared.output(dtype, outputs, false)
    }

    fn progress(&self) -> f64 {
        if self.is_exhausted() {
            return 1.0;
        }
        self.shared.progress()
    }

    fn is_exhausted(&self) -> bool {
        // Slots hold senders, so the channel closes only once every unit is in it.
        self.shared.limit.is_trimmed_out()
            || (self.receiver.is_closed() && self.receiver.is_empty())
    }
}

impl Drop for UnorderedPool {
    fn drop(&mut self) {
        // Queued expansions hold senders of the channel itself; drain them so it can be freed.
        self.receiver.close();
        while self.receiver.try_recv().is_ok() {}
    }
}

/// A pool that gives all of its units, in row order, to its first worker.
struct OrderedPool {
    dtype: DType,
    shared: Arc<Shared>,
    /// The units, until the first worker takes them.
    units: Mutex<Option<Vec<WorkUnit>>>,
    expand_concurrency: usize,
}

impl OrderedPool {
    /// The units that `units` stand for, in row order, with each expansion's units in its place.
    /// Expansions run up to `expand_concurrency` ahead of the units being taken, counting each
    /// run of consecutive units that need no expansion as one step, as a file's splits are.
    fn in_order(
        shared: Arc<Shared>,
        units: Vec<WorkUnit>,
        expand_concurrency: usize,
    ) -> BoxStream<'static, VortexResult<WorkUnit>> {
        let mut segments: Vec<Segment> = Vec::new();
        for unit in units {
            match (unit.is_expand(), segments.last_mut()) {
                (true, _) => segments.push(Segment::Expand(unit)),
                (false, Some(Segment::Units(run))) => run.push(unit),
                (false, _) => segments.push(Segment::Units(vec![unit])),
            }
        }
        let expander = Arc::clone(&shared);
        stream::iter(segments)
            .map(move |segment| match segment {
                Segment::Units(units) => future::ready(Ok((units, false))).boxed(),
                Segment::Expand(unit) => expander
                    .expand(unit)
                    .map(|units| units.map(|units| (units, true)))
                    .boxed(),
            })
            .buffered(expand_concurrency)
            .flat_map(move |segment| match segment {
                Ok((units, false)) => stream::iter(units.into_iter().map(Ok)).boxed(),
                Ok((units, true)) => {
                    OrderedPool::in_order(Arc::clone(&shared), units, expand_concurrency)
                }
                Err(err) => stream::once(future::ready(Err(err))).boxed(),
            })
            .boxed()
    }
}

/// A step of an ordered pool's units: consecutive units that need no expansion, or one expansion.
enum Segment {
    Units(Vec<WorkUnit>),
    Expand(WorkUnit),
}

impl ScanPool for OrderedPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let dtype = self.dtype.clone();
        let Some(units) = self.units.lock().take() else {
            return ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, stream::empty()));
        };
        let shared = Arc::clone(&self.shared);
        let runner = Arc::clone(&shared);
        let gate = Arc::clone(&shared);
        let outputs = OrderedPool::in_order(Arc::clone(&shared), units, self.expand_concurrency)
            .take_while(move |_| future::ready(!gate.limit.is_trimmed_out()))
            .map(move |unit| match unit {
                Ok(unit) => runner.run(unit),
                Err(err) => future::ready(Err(err)).boxed(),
            })
            .buffered(shared.in_flight)
            .boxed();
        shared.output(dtype, outputs, true)
    }

    fn progress(&self) -> f64 {
        self.shared.progress()
    }

    fn is_exhausted(&self) -> bool {
        // The first worker takes every unit.
        self.shared.limit.is_trimmed_out() || self.units.lock().is_none()
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::ops::Range;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;
    use futures::TryStreamExt;
    use futures::executor::block_on;
    use futures::future::try_join_all;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::PoolLimit;
    use super::WorkPoolOptions;
    use super::WorkUnit;
    use super::open_work_pool;
    use crate::pool::RowLimit;
    use crate::pool::ScanPoolRef;
    use crate::test::SESSION;

    fn dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    /// A unit returning the values of `rows`.
    fn rows(rows: Range<i32>, share: f64) -> WorkUnit {
        WorkUnit::run(share, move || {
            futures::future::ready(Ok(vec![Buffer::from_iter(rows).into_array()])).boxed()
        })
    }

    /// An expansion of `rows` into `parts` units, or into nested expansions `depth` levels deep,
    /// counting how many expansions run at once in `running` and the most in `peak`.
    fn expansion(
        rows_: Range<i32>,
        parts: i32,
        depth: usize,
        share: f64,
        running: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    ) -> WorkUnit {
        WorkUnit::expand(share, move || {
            async move {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                yield_once().await;
                let step = (rows_.end - rows_.start) / parts;
                let share = share / f64::from(parts);
                let units = (0..parts)
                    .map(|i| {
                        let start = rows_.start + i * step;
                        let part = start..start + step;
                        if depth == 0 {
                            rows(part, share)
                        } else {
                            let (running, peak) = (Arc::clone(&running), Arc::clone(&peak));
                            expansion(part, parts, depth - 1, share, running, peak)
                        }
                    })
                    .collect();
                running.fetch_sub(1, Ordering::SeqCst);
                Ok(units)
            }
            .boxed()
        })
    }

    /// Pends once, waking itself, so that other work runs in between.
    async fn yield_once() {
        let mut yielded = false;
        futures::future::poll_fn(|cx| {
            if yielded {
                return std::task::Poll::Ready(());
            }
            yielded = true;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        })
        .await;
    }

    fn options(ordered: bool, limit: PoolLimit, expand_concurrency: usize) -> WorkPoolOptions {
        WorkPoolOptions {
            dtype: dtype(),
            ordered,
            limit,
            split_concurrency: NonZeroUsize::new(2),
            expand_concurrency: NonZeroUsize::new(expand_concurrency).unwrap_or(NonZeroUsize::MIN),
            map_trimmed: None,
        }
    }

    fn values(arrays: Vec<ArrayRef>) -> VortexResult<Vec<i32>> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = Vec::new();
        for array in arrays {
            values.extend_from_slice(array.execute::<PrimitiveArray>(&mut ctx)?.as_slice::<i32>());
        }
        Ok(values)
    }

    /// Reads `pool` with `workers` workers, returning each worker's values.
    fn read(pool: &ScanPoolRef, workers: usize) -> VortexResult<Vec<Vec<i32>>> {
        let workers = (0..workers).map(|_| Arc::clone(pool).worker().try_collect::<Vec<_>>());
        block_on(try_join_all(workers))?
            .into_iter()
            .map(values)
            .collect()
    }

    /// Three top-level units: rows, a two-level expansion, and rows again.
    fn units(running: &Arc<AtomicUsize>, peak: &Arc<AtomicUsize>) -> Vec<WorkUnit> {
        let share = 1.0 / 3.0;
        vec![
            rows(0..10, share),
            expansion(10..90, 2, 2, share, Arc::clone(running), Arc::clone(peak)),
            rows(90..100, share),
        ]
    }

    /// An ordered pool's first worker reads every unit in row order, with each expansion's units
    /// in its place however deep it nests; an unordered pool's workers read each unit once.
    #[rstest]
    #[case::ordered(true)]
    #[case::unordered(false)]
    fn pool_reads_expansions_in_place(#[case] ordered: bool) -> VortexResult<()> {
        let (running, peak) = Default::default();
        let pool = open_work_pool(
            options(ordered, PoolLimit::Unlimited, 4),
            units(&running, &peak),
        );
        assert_eq!(pool.progress(), 0.0);
        let workers = read(&pool, 3)?;
        if ordered {
            assert_eq!(workers[0], (0..100).collect::<Vec<_>>());
            assert!(workers[1..].iter().all(Vec::is_empty));
        } else {
            let mut all: Vec<i32> = workers.into_iter().flatten().collect();
            all.sort_unstable();
            assert_eq!(all, (0..100).collect::<Vec<_>>());
        }
        assert!(pool.is_exhausted());
        assert!((pool.progress() - 1.0).abs() < 1e-9, "{}", pool.progress());
        Ok(())
    }

    /// An unordered pool runs at most `expand_concurrency` expansions at once.
    #[test]
    fn unordered_pool_bounds_expansions() -> VortexResult<()> {
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let units = (0..8)
            .map(|i| {
                let (running, peak) = (Arc::clone(&running), Arc::clone(&peak));
                expansion(i * 20..i * 20 + 20, 2, 1, 1.0 / 8.0, running, peak)
            })
            .collect();
        let pool = open_work_pool(options(false, PoolLimit::Unlimited, 1), units);
        let mut all: Vec<i32> = read(&pool, 4)?.into_iter().flatten().collect();
        all.sort_unstable();
        assert_eq!(all, (0..160).collect::<Vec<_>>());
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        Ok(())
    }

    /// A trimmed limit ends the workers once spent, slicing the array that spends it, and leaves
    /// the pool exhausted.
    #[rstest]
    #[case::ordered(true)]
    #[case::unordered(false)]
    fn trimmed_limit_ends_the_workers(#[case] ordered: bool) -> VortexResult<()> {
        let (running, peak) = Default::default();
        let limit = PoolLimit::Trim(RowLimit::new(25));
        let pool = open_work_pool(options(ordered, limit, 4), units(&running, &peak));
        let workers = read(&pool, 2)?;
        let rows: usize = workers.iter().map(Vec::len).sum();
        assert_eq!(rows, 25);
        if ordered {
            assert_eq!(workers[0], (0..25).collect::<Vec<_>>());
        }
        assert!(pool.is_exhausted());
        assert_eq!(pool.progress(), 1.0);
        Ok(())
    }
}
