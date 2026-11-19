// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The split work that the workers of an opened layout scan share.
//!
//! A [`SplitPool`] holds the remaining splits of one or more files, in file and row order, and
//! the files not yet opened. Each [`ScanPool::worker`] takes splits from it and keeps several of
//! them in flight on its own thread: it builds a split's task only when it admits the split, and
//! never spawns. Opening a deferred file is work like any other: the worker that takes it polls
//! the open and returns the file's splits to the pool, where any worker can take them.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesOrdered;
use futures::stream::FuturesUnordered;
use parking_lot::Mutex;
use tracing::Instrument;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_scan::ScanPool;
use vortex_scan::row_mask::RowMask;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::limit::RowLimit;
use crate::scan::multi::LayoutReaderFactory;
use crate::scan::repeated_scan::RepeatedScan;
use crate::scan::repeated_scan::can_prune_file;
use crate::scan::tasks::TaskContext;

/// A scan prepared once and shared by every piece and execution of it.
pub(crate) type PreparedScan = RepeatedScan<ArrayRef>;

/// A prepared scan and the row ranges of its splits, in row order.
pub(crate) type PlannedSplits = (Arc<PreparedScan>, Vec<Range<u64>>);

/// Splits in flight for each core: enough I/O read-ahead that a worker's thread rarely waits.
const SPLITS_IN_FLIGHT_PER_CORE: usize = 4;

/// The default for the most splits in flight across all workers of one pool.
///
/// A single worker may use all of it, so a scan read by one consumer keeps as many splits' I/O in
/// flight as one scan stream did when it spawned its split tasks. Several workers share it, so
/// adding workers does not multiply the memory held by decoding splits.
pub(crate) fn default_splits_in_flight() -> NonZeroUsize {
    NonZeroUsize::new(
        SPLITS_IN_FLIGHT_PER_CORE.saturating_mul(get_available_parallelism().unwrap_or(1)),
    )
    .unwrap_or(NonZeroUsize::MIN)
}

/// One execution of one file's prepared scan.
pub(crate) struct FileScan {
    scan: Arc<PreparedScan>,
    /// Records this execution's events, so file events count once for each execution.
    ctx: Arc<TaskContext>,
}

/// One file's splits for one execution, ready to be handed to workers.
pub(crate) struct FileWork {
    scan: Option<Arc<FileScan>>,
    ranges: Vec<Range<u64>>,
}

impl FileWork {
    /// A file with nothing to read, for example one that statistics pruned.
    pub(crate) fn empty() -> Self {
        Self {
            scan: None,
            ranges: Vec::new(),
        }
    }

    /// Starts an execution of `scan` over its split `ranges`.
    ///
    /// Counts a file scan attempt when the ranges select any row, and returns no work when
    /// whole-file statistics reject the filter.
    pub(crate) fn start(scan: Arc<PreparedScan>, ranges: Vec<Range<u64>>) -> VortexResult<Self> {
        if !ranges.iter().any(|range| scan.selects_rows(range)) {
            return Ok(Self::empty());
        }
        let ctx = scan.task_context();
        if can_prune_file(&ctx)? {
            return Ok(Self::empty());
        }
        Ok(Self {
            scan: Some(Arc::new(FileScan { scan, ctx })),
            ranges,
        })
    }
}

/// Opens a deferred file's reader into its work for this execution.
pub(crate) type PrepareFile = Arc<dyn Fn(LayoutReaderRef) -> VortexResult<FileWork> + Send + Sync>;

/// How a pool's row limit is applied.
enum PoolLimit {
    Unlimited,
    /// A split's selected rows are its output rows, so each split reserves them from the budget
    /// when a worker admits it, and rows past the limit are never decoded. Admission order is
    /// row order for a single worker, so an ordered worker returns exactly the first rows.
    Reserve(RowLimit),
    /// A split's output rows are unknown until its filter has run, so the worker takes rows from
    /// the budget as it yields them and discards the excess: in row order for an ordered worker,
    /// in completion order otherwise.
    Trim(RowLimit),
}

impl PoolLimit {
    fn is_exhausted(&self) -> bool {
        match self {
            PoolLimit::Unlimited => false,
            PoolLimit::Reserve(limit) | PoolLimit::Trim(limit) => limit.is_exhausted(),
        }
    }
}

/// How a pool reads its splits.
pub(crate) struct PoolOptions {
    pub(crate) dtype: DType,
    pub(crate) ordered: bool,
    pub(crate) limit: Option<u64>,
    /// Whether the scan filters rows, so a split's output row count is unknown up front.
    pub(crate) has_filter: bool,
    /// The most splits in flight across all workers; each worker may always have one.
    pub(crate) max_splits_in_flight: NonZeroUsize,
}

/// A file whose splits workers may take.
struct ReadyFile {
    scan: Arc<FileScan>,
    ranges: VecDeque<Range<u64>>,
    /// The progress that handing out one of the file's rows adds.
    progress_per_row: f64,
}

struct PoolState {
    /// Opened files with splits left, by file index.
    ready: BTreeMap<usize, ReadyFile>,
    /// Files not yet opened, by file index.
    deferred: VecDeque<(usize, Arc<dyn LayoutReaderFactory>)>,
    /// Files a worker is opening.
    opening: BTreeSet<usize>,
    /// Workers waiting for an opening file's splits.
    waiters: Vec<Waker>,
    /// The fraction of the scan handed out so far.
    progress: f64,
    file_count: usize,
}

impl PoolState {
    /// Makes an opened file's splits available.
    fn add_file(&mut self, index: usize, work: FileWork) {
        let file_share = 1.0 / self.file_count.max(1) as f64;
        let rows: u64 = work
            .ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum();
        match work.scan {
            Some(scan) if rows > 0 => {
                self.ready.insert(
                    index,
                    ReadyFile {
                        scan,
                        ranges: work.ranges.into(),
                        progress_per_row: file_share / rows as f64,
                    },
                );
            }
            _ => self.progress += file_share,
        }
    }

    /// The lowest file index whose splits are not known yet.
    fn first_unopened(&self) -> Option<usize> {
        let deferred = self.deferred.front().map(|(index, _)| *index);
        let opening = self.opening.first().copied();
        deferred.into_iter().chain(opening).min()
    }

    fn wake_all(&mut self) {
        for waker in self.waiters.drain(..) {
            waker.wake();
        }
    }
}

/// What a worker should do next.
enum Next {
    Split(Arc<FileScan>, Range<u64>),
    Open(usize, Arc<dyn LayoutReaderFactory>),
    /// Nothing to take now, but a file being opened may add splits. The worker is woken then.
    Wait,
    /// Nothing is left to take.
    Done,
}

/// The shared work of an opened layout scan; see the module documentation.
pub(crate) struct SplitPool {
    dtype: DType,
    ordered: bool,
    limit: PoolLimit,
    max_splits_in_flight: usize,
    splits_in_flight: AtomicUsize,
    /// How deferred files become work, and how many may open at once.
    prepare_file: Option<(PrepareFile, usize)>,
    state: Mutex<PoolState>,
}

impl SplitPool {
    /// Creates a pool over opened files and deferred files, each given with its file index.
    ///
    /// File indices order the files. `file_count` counts every file of the scan, including any
    /// the caller already dropped, and weights progress.
    pub(crate) fn new(
        options: PoolOptions,
        opened: Vec<(usize, FileWork)>,
        deferred: Vec<(usize, Arc<dyn LayoutReaderFactory>)>,
        file_count: usize,
        prepare_file: Option<(PrepareFile, usize)>,
    ) -> Self {
        let limit = match options.limit {
            None => PoolLimit::Unlimited,
            Some(limit) if options.has_filter => PoolLimit::Trim(RowLimit::new(limit)),
            Some(limit) => PoolLimit::Reserve(RowLimit::new(limit)),
        };
        let mut state = PoolState {
            ready: BTreeMap::new(),
            deferred: deferred.into(),
            opening: BTreeSet::new(),
            waiters: Vec::new(),
            progress: 0.0,
            file_count,
        };
        state
            .deferred
            .make_contiguous()
            .sort_by_key(|(index, _)| *index);
        for (index, work) in opened {
            state.add_file(index, work);
        }
        Self {
            dtype: options.dtype,
            ordered: options.ordered,
            limit,
            max_splits_in_flight: options.max_splits_in_flight.get(),
            splits_in_flight: AtomicUsize::new(0),
            prepare_file: prepare_file.filter(|_| !state.deferred.is_empty()),
            state: Mutex::new(state),
        }
    }

    fn next(&self, cx: &Context<'_>) -> Next {
        if self.limit.is_exhausted() {
            return Next::Done;
        }
        let mut state = self.state.lock();
        let first_unopened = state.first_unopened();
        if let Some(mut entry) = state.ready.first_entry() {
            // An ordered worker never takes a split past a file whose splits are not known yet.
            let blocked = self.ordered && first_unopened.is_some_and(|index| index < *entry.key());
            if !blocked {
                let file = entry.get_mut();
                if let Some(range) = file.ranges.pop_front() {
                    let progress = file.progress_per_row * (range.end - range.start) as f64;
                    let scan = Arc::clone(&file.scan);
                    if file.ranges.is_empty() {
                        entry.remove();
                    }
                    state.progress += progress;
                    return Next::Split(scan, range);
                }
            }
        }
        if let Some((_, max_opens)) = &self.prepare_file
            && state.opening.len() < *max_opens
            && let Some((index, factory)) = state.deferred.pop_front()
        {
            state.opening.insert(index);
            return Next::Open(index, factory);
        }
        if state.opening.is_empty() {
            return Next::Done;
        }
        if !state
            .waiters
            .iter()
            .any(|waker| waker.will_wake(cx.waker()))
        {
            state.waiters.push(cx.waker().clone());
        }
        Next::Wait
    }

    /// Adds an opened file's splits, or nothing when it failed or was skipped.
    fn finish_open(&self, index: usize, work: FileWork) {
        let mut state = self.state.lock();
        state.opening.remove(&index);
        state.add_file(index, work);
        state.wake_all();
    }

    /// Hands back a file that a dropped worker was opening, so another worker opens it.
    fn return_open(&self, index: usize, factory: Arc<dyn LayoutReaderFactory>) {
        let mut state = self.state.lock();
        state.opening.remove(&index);
        let position = state.deferred.partition_point(|(other, _)| *other < index);
        state.deferred.insert(position, (index, factory));
        state.wake_all();
    }
}

impl ScanPool for SplitPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let dtype = self.dtype.clone();
        let splits = if self.ordered {
            InFlight::Ordered(FuturesOrdered::new())
        } else {
            InFlight::Unordered(FuturesUnordered::new())
        };
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype,
            Worker {
                pool: self,
                splits,
                opens: FuturesUnordered::new(),
                held_opens: Vec::new(),
                is_finished: false,
            },
        ))
    }

    fn is_exhausted(&self) -> bool {
        if self.limit.is_exhausted() {
            return true;
        }
        let state = self.state.lock();
        state.ready.is_empty() && state.deferred.is_empty() && state.opening.is_empty()
    }

    fn progress(&self) -> f64 {
        if self.limit.is_exhausted() {
            return 1.0;
        }
        self.state.lock().progress.clamp(0.0, 1.0)
    }
}

type SplitFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;
type OpenFuture = BoxFuture<'static, (usize, VortexResult<Option<LayoutReaderRef>>)>;

/// A worker's splits in flight: yielded in admission order for an ordered scan.
enum InFlight {
    Ordered(FuturesOrdered<SplitFuture>),
    Unordered(FuturesUnordered<SplitFuture>),
}

impl InFlight {
    fn len(&self) -> usize {
        match self {
            InFlight::Ordered(futures) => futures.len(),
            InFlight::Unordered(futures) => futures.len(),
        }
    }

    fn push(&mut self, future: SplitFuture) {
        match self {
            InFlight::Ordered(futures) => futures.push_back(future),
            InFlight::Unordered(futures) => futures.push(future),
        }
    }

    fn clear(&mut self) {
        match self {
            InFlight::Ordered(futures) => *futures = FuturesOrdered::new(),
            InFlight::Unordered(futures) => futures.clear(),
        }
    }

    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<Option<VortexResult<Option<ArrayRef>>>> {
        match self {
            InFlight::Ordered(futures) => futures.poll_next_unpin(cx),
            InFlight::Unordered(futures) => futures.poll_next_unpin(cx),
        }
    }
}

/// Why a worker stopped taking work.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fill {
    /// The worker has as many splits in flight as it may.
    Full,
    /// The pool has nothing now but may later; the worker is registered to be woken.
    Waiting,
    /// The pool has nothing left.
    Done,
}

/// One worker of a [`SplitPool`]: a stream polled inline by its caller.
struct Worker {
    pool: Arc<SplitPool>,
    splits: InFlight,
    opens: FuturesUnordered<OpenFuture>,
    /// The files this worker is opening, handed back to the pool if it is dropped.
    held_opens: Vec<(usize, Arc<dyn LayoutReaderFactory>)>,
    is_finished: bool,
}

impl Worker {
    fn may_admit(&self) -> bool {
        self.splits.len() == 0
            || self.pool.splits_in_flight.load(Ordering::Relaxed) < self.pool.max_splits_in_flight
    }

    /// Takes work from the pool until the worker is full or the pool has nothing to give.
    fn fill(&mut self, cx: &Context<'_>) -> VortexResult<Fill> {
        loop {
            if !self.may_admit() {
                return Ok(Fill::Full);
            }
            match self.pool.next(cx) {
                Next::Split(file, range) => self.admit(&file, &range)?,
                Next::Open(index, factory) => {
                    self.held_opens.push((index, Arc::clone(&factory)));
                    self.opens.push(
                        async move {
                            let reader = factory
                                .open()
                                .instrument(tracing::info_span!("LayoutReaderFactory::open"))
                                .await;
                            (index, reader)
                        }
                        .boxed(),
                    );
                }
                Next::Wait => return Ok(Fill::Waiting),
                Next::Done => return Ok(Fill::Done),
            }
        }
    }

    /// Builds the task of a split taken from the pool and puts it in flight.
    fn admit(&mut self, file: &FileScan, range: &Range<u64>) -> VortexResult<()> {
        let Some(mask) = file.scan.split_mask(range) else {
            return Ok(());
        };
        let mask = match &self.pool.limit {
            PoolLimit::Reserve(limit) => {
                let reserved = limit.limit(mask.mask().clone());
                if reserved.all_false() {
                    return Ok(());
                }
                RowMask::new(range.start, reserved)
            }
            PoolLimit::Unlimited | PoolLimit::Trim(_) => mask,
        };
        let task = file.scan.split_task(&file.ctx, mask)?;
        self.pool.splits_in_flight.fetch_add(1, Ordering::Relaxed);
        self.splits.push(task);
        Ok(())
    }

    /// Drops the splits in flight and releases their share of the pool's budget.
    fn release_splits(&mut self) {
        self.pool
            .splits_in_flight
            .fetch_sub(self.splits.len(), Ordering::Relaxed);
        self.splits.clear();
    }

    /// Hands every file this worker is opening back to the pool.
    fn release_opens(&mut self) {
        self.opens = FuturesUnordered::new();
        for (index, factory) in self.held_opens.drain(..) {
            self.pool.return_open(index, factory);
        }
    }

    fn fail<T>(&mut self, err: VortexError) -> Poll<Option<VortexResult<T>>> {
        self.is_finished = true;
        self.release_splits();
        self.release_opens();
        Poll::Ready(Some(Err(err)))
    }

    /// Finishes an open: hands the file's splits to the pool, or reports its error.
    fn finish_open(
        &mut self,
        index: usize,
        reader: VortexResult<Option<LayoutReaderRef>>,
    ) -> VortexResult<()> {
        self.held_opens.retain(|(held, _)| *held != index);
        let work = match reader {
            Ok(Some(reader)) => match &self.pool.prepare_file {
                Some((prepare, _)) => prepare(reader),
                None => Ok(FileWork::empty()),
            },
            Ok(None) => Ok(FileWork::empty()),
            Err(err) => Err(err),
        };
        match work {
            Ok(work) => {
                self.pool.finish_open(index, work);
                Ok(())
            }
            Err(err) => {
                self.pool.finish_open(index, FileWork::empty());
                Err(err)
            }
        }
    }
}

impl Stream for Worker {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            if this.is_finished {
                return Poll::Ready(None);
            }
            let fill = match this.fill(cx) {
                Ok(fill) => fill,
                Err(err) => return this.fail(err),
            };

            if let Poll::Ready(Some((index, reader))) = this.opens.poll_next_unpin(cx) {
                if let Err(err) = this.finish_open(index, reader) {
                    return this.fail(err);
                }
                // The opened file's splits may be ours to take.
                continue;
            }

            match this.splits.poll_next(cx) {
                Poll::Ready(Some(result)) => {
                    this.pool.splits_in_flight.fetch_sub(1, Ordering::Relaxed);
                    let array = match result {
                        Ok(Some(array)) => array,
                        Ok(None) => continue,
                        Err(err) => return this.fail(err),
                    };
                    let array = match &this.pool.limit {
                        PoolLimit::Trim(limit) => {
                            let granted = limit.take(array.len());
                            if limit.is_exhausted() {
                                this.release_splits();
                            }
                            if granted == 0 {
                                continue;
                            }
                            if granted < array.len() {
                                match array.slice(0..granted) {
                                    Ok(array) => array,
                                    Err(err) => return this.fail(err),
                                }
                            } else {
                                array
                            }
                        }
                        PoolLimit::Unlimited | PoolLimit::Reserve(_) => array,
                    };
                    return Poll::Ready(Some(Ok(array)));
                }
                Poll::Ready(None) => {
                    if fill == Fill::Done
                        && (this.opens.is_empty() || this.pool.limit.is_exhausted())
                    {
                        this.release_opens();
                        this.is_finished = true;
                        return Poll::Ready(None);
                    }
                    // Woken by an open in progress, or by the pool when another worker's open
                    // adds splits.
                    return Poll::Pending;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.release_splits();
        self.release_opens();
    }
}
