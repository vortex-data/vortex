// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The work that the workers of an opened layout scan share.
//!
//! A pool's units are the splits of one or more files, and the files not yet opened. Its splits
//! run as tasks on the session's runtime, at most `splits_in_flight` at once across the whole
//! pool; it never starts threads of its own. At most `open_concurrency` deferred files open at
//! once.
//!
//! Splits run as [`RepeatedScan::execute_stream`] runs them:
//!
//! - A scan with a filter and a limit builds each split on a CPU thread only once it is taken
//!   to run ([`spawn_lazy_split`]), so no new split starts once the limit is spent, and optimizes the
//!   rows the limit keeps on a second CPU task ([`spawn_map_trimmed`]).
//! - Every other scan builds all of a file's split tasks together, when the first of them is
//!   taken, so the file's reads are registered at once and can coalesce. A limit is reserved
//!   then, in split order.
//!
//! The pools differ in how workers share the work:
//!
//! - An unordered pool starts `open_concurrency` openers and its lanes on the runtime once its
//!   first worker starts. Each opener opens deferred files one after another and queues each to
//!   be planned. The lanes start one for each run of splits or plan that is ready, more as they
//!   queue, and at most enough to keep `splits_in_flight` splits running. Each lane takes the
//!   next unit from a channel of units, runs it, and sends its rows into one bounded output
//!   channel. An opened file is one unit, its turn, which a lane queues again after taking the
//!   file's next run of adjacent splits, so the lanes read the opened files in turn, a run at a
//!   time, and poll each run's splits together so that their reads coalesce. A lane that plans a
//!   file queues its turn. Workers only take rows from the output channel, so whichever worker
//!   asks next gets the next rows, however many workers there are, and the number of workers is
//!   only the host's parallelism. The output channel closes once every opener and lane has
//!   finished, which ends the workers.
//! - An ordered pool gives all of its units, in row order, to its first worker, which keeps
//!   `splits_in_flight` of them running at once; later workers get nothing. Parallel ordered reads
//!   cut the scan with `split` first.

use std::any::Any;
use std::collections::VecDeque;
use std::iter;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use async_channel::Receiver;
use async_channel::Sender;
use async_lock::OnceCell;
use futures::FutureExt;
use futures::StreamExt;
use futures::future;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use futures::stream::FuturesUnordered;
use parking_lot::Mutex;
use tracing::Instrument;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Precision;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::pool::ScanPool;
use vortex_scan::pool::ScanPoolRef;
use vortex_scan::row_mask::RowMask;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::scan::limit::RowLimit;
use crate::scan::multi::LayoutReaderFactory;
use crate::scan::repeated_scan::DEFAULT_SPLITS_PER_THREAD;
use crate::scan::repeated_scan::MapFn;
use crate::scan::repeated_scan::RepeatedScan;
use crate::scan::repeated_scan::SplitConcurrency;
use crate::scan::repeated_scan::can_prune_file;
use crate::scan::repeated_scan::recheck_prune_file;
use crate::scan::repeated_scan::spawn_lazy_split;
use crate::scan::repeated_scan::spawn_map_trimmed;
use crate::scan::repeated_scan::stop_on_error;
use crate::scan::tasks::SplitFuture;
use crate::scan::tasks::TaskContext;
use crate::scan::tasks::split_exec;

/// A scan prepared once and shared by every piece and execution of it.
pub(crate) type PreparedScan = RepeatedScan<ArrayRef>;

/// A prepared scan and the row ranges of its splits, in row order.
pub(crate) type PlannedSplits = (Arc<PreparedScan>, Vec<Range<u64>>);

/// One execution of one file's prepared scan over some of its splits.
pub(crate) struct FileScan {
    scan: Arc<PreparedScan>,
    /// Records this execution's events, so file events count once for each execution.
    ctx: Arc<TaskContext>,
    /// The row ranges of the splits, in row order.
    ranges: Vec<Range<u64>>,
    /// The task of each split, built together when a worker first takes one of them. `None`
    /// where the split has no rows to read, or a worker has taken it.
    tasks: OnceCell<Vec<Mutex<Option<SplitFuture>>>>,
    /// Whether whole-file statistics reject the filter when the file's first split is taken.
    pruned: OnceCell<bool>,
}

impl FileScan {
    /// Whether whole-file statistics reject the filter, checked again when the first split is
    /// taken, as [`RepeatedScan::execute_stream`] checks when it starts: a filter with dynamic
    /// parts may reject the file by then though it did not when the pool was opened.
    async fn is_pruned(&self) -> VortexResult<bool> {
        let pruned = self
            .pruned
            .get_or_try_init(|| future::ready(recheck_prune_file(&self.ctx)))
            .await?;
        Ok(*pruned)
    }

    /// The tasks of the file's splits, built on first use. A file whose tasks are built once a
    /// reserved limit is spent, or that statistics now reject, builds none.
    async fn tasks(&self, limit: &PoolLimit) -> VortexResult<&[Mutex<Option<SplitFuture>>]> {
        let tasks = self
            .tasks
            .get_or_try_init(|| async {
                if self.is_pruned().await? {
                    return Ok(Vec::new());
                }
                self.build_tasks(limit)
            })
            .await?;
        Ok(tasks)
    }

    /// Builds every split's task, which registers its reads, reserving a reserved limit in split
    /// order.
    fn build_tasks(&self, limit: &PoolLimit) -> VortexResult<Vec<Mutex<Option<SplitFuture>>>> {
        self.ranges
            .iter()
            .map(|range| {
                let mask = self.scan.split_mask(range).and_then(|mask| match limit {
                    PoolLimit::Reserve(limit) => {
                        let reserved = limit.limit(mask.mask().clone());
                        (!reserved.all_false()).then(|| RowMask::new(range.start, reserved))
                    }
                    PoolLimit::Unlimited | PoolLimit::Trim(_) => Some(mask),
                });
                let task = mask
                    .map(|mask| self.scan.split_task(&self.ctx, mask))
                    .transpose()?;
                Ok(Mutex::new(task))
            })
            .collect()
    }
}

/// One file's splits for one execution, ready to be handed to workers.
pub(crate) struct FileWork {
    file: Option<Arc<FileScan>>,
}

impl FileWork {
    /// A file with nothing to read, for example one that statistics pruned.
    pub(crate) fn empty() -> Self {
        Self { file: None }
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
            file: Some(Arc::new(FileScan {
                scan,
                ctx,
                ranges,
                tasks: OnceCell::new(),
                pruned: OnceCell::new(),
            })),
        })
    }
}

/// Opens a deferred file's reader into its work for this execution.
pub(crate) type PrepareFile = Arc<dyn Fn(LayoutReaderRef) -> VortexResult<FileWork> + Send + Sync>;

/// A file of a pool, in file order.
pub(crate) enum PoolFile {
    Opened(FileWork),
    Deferred(Arc<dyn LayoutReaderFactory>),
}

/// How a pool reads its splits.
pub(crate) struct PoolOptions {
    pub(crate) session: VortexSession,
    pub(crate) dtype: DType,
    pub(crate) ordered: bool,
    pub(crate) limit: Option<u64>,
    /// Whether the scan filters rows, so a split's output row count is unknown up front.
    pub(crate) has_filter: bool,
    /// The request's bound on the splits each worker keeps in flight.
    pub(crate) split_concurrency: Option<NonZeroUsize>,
    /// The most deferred files that are queued or being opened at once.
    pub(crate) open_concurrency: usize,
}

/// Opens a pool over `files`, in file order. `prepare` turns an opened deferred file into work.
pub(crate) fn open_pool(
    options: PoolOptions,
    files: Vec<PoolFile>,
    prepare: Option<PrepareFile>,
) -> ScanPoolRef {
    let limit = match options.limit {
        None => PoolLimit::Unlimited,
        Some(limit) if options.has_filter => PoolLimit::Trim(RowLimit::new(limit)),
        Some(limit) => PoolLimit::Reserve(RowLimit::new(limit)),
    };
    // The pool keeps as many splits in flight as one scan stream does by default: four per core,
    // at most the request's `split_concurrency`, however many workers read it. The runtime's
    // threads decode them, so more in flight would only read further ahead, and the file's
    // registered reads already let I/O coalesce ahead of the splits being decoded.
    let splits_in_flight = SplitConcurrency {
        per_worker: DEFAULT_SPLITS_PER_THREAD,
        max: options.split_concurrency,
    }
    .splits_in_flight()
    .max(1);
    let open_concurrency = options.open_concurrency.max(1);
    let shared = Arc::new(Shared {
        limit,
        prepare,
        session: options.session,
        file_share: 1.0 / files.len().max(1) as f64,
        progress: Mutex::new(0.0),
        deferred: Mutex::new(VecDeque::new()),
        failed: AtomicBool::new(false),
        pending: AtomicUsize::new(0),
    });
    if options.ordered {
        return Arc::new(OrderedPool {
            dtype: options.dtype,
            shared,
            files: Mutex::new(Some(files)),
            splits_in_flight,
            open_concurrency,
        });
    }

    let (sender, receiver) = async_channel::unbounded();
    for file in files {
        match file {
            PoolFile::Opened(work) => shared.queue_file(&sender, work),
            PoolFile::Deferred(factory) => shared.deferred.lock().push_back(factory),
        }
    }
    // Each open file gets an equal share of the splits in flight, and a lane takes a file's share
    // as one run of adjacent splits that it polls together. The run's reads then reach the
    // file's read driver together and coalesce, as they do when one scan stream polls its
    // file's splits; a run of one would hand the driver one read at a time. The rotation moves
    // between files after each run, and the lanes hold `splits_in_flight` splits in all.
    let run = (splits_in_flight / open_concurrency).max(1);
    let (output, outputs) = async_channel::bounded(splits_in_flight);
    let lanes = Arc::new(Lanes {
        shared: Arc::clone(&shared),
        units: receiver.clone(),
        run,
        max: (splits_in_flight / run).max(1),
        waiting: AtomicUsize::new(0),
        started: AtomicUsize::new(0),
        tasks: Mutex::new(Some(Vec::new())),
        open_concurrency,
        units_sender: Mutex::new(Some(sender)),
        openers: Mutex::new(Some(Vec::new())),
    });
    Arc::new(UnorderedPool {
        dtype: options.dtype,
        shared,
        units: receiver,
        outputs,
        lanes,
        output: Mutex::new(Some(output)),
    })
}

/// How a pool's row limit is applied.
enum PoolLimit {
    Unlimited,
    /// A split's selected rows are its output rows, so a file reserves its splits' rows from the
    /// budget in split order when it builds their tasks, and rows past the limit are never
    /// decoded. Files build in row order in an ordered pool, so it returns exactly the first rows.
    Reserve(RowLimit),
    /// A split's output rows are unknown until its filter has run, so workers take rows from the
    /// budget as they yield them and discard the excess.
    Trim(RowLimit),
}

impl PoolLimit {
    /// Whether the budget is spent, so no file that has not started has rows to return.
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

/// A split's output: `None` when it has no rows to return.
type SplitOutput = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// A split of a file by its index, with the progress that taking it adds.
type FileSplit = (Arc<FileScan>, usize, f64);

/// What every worker of a pool shares.
struct Shared {
    limit: PoolLimit,
    prepare: Option<PrepareFile>,
    /// The session whose runtime runs the pool's splits and opens.
    session: VortexSession,
    /// The progress that one file adds.
    file_share: f64,
    progress: Mutex<f64>,
    /// An unordered pool's deferred files that no opener has taken yet, in file order.
    deferred: Mutex<VecDeque<Arc<dyn LayoutReaderFactory>>>,
    /// Whether a unit of an unordered pool failed, which stops its lanes.
    failed: AtomicBool,
    /// The splits and opens of an unordered pool that are queued and not yet taken by a lane.
    pending: AtomicUsize,
}

impl Shared {
    /// The splits of an opened file, in row order.
    fn file_splits(&self, work: FileWork) -> Vec<FileSplit> {
        let Some(file) = work.file else {
            *self.progress.lock() += self.file_share;
            return Vec::new();
        };
        let rows: u64 = file
            .ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum();
        if rows == 0 {
            *self.progress.lock() += self.file_share;
            return Vec::new();
        }
        let per_row = self.file_share / rows as f64;
        (0..file.ranges.len())
            .map(|index| {
                let range = &file.ranges[index];
                let progress = per_row * (range.end - range.start) as f64;
                (Arc::clone(&file), index, progress)
            })
            .collect()
    }

    /// Opens a deferred file into its work, on the task that polls it. A file that starts after
    /// the limit is spent is not opened.
    fn open_file(
        self: &Arc<Self>,
        factory: Arc<dyn LayoutReaderFactory>,
    ) -> BoxFuture<'static, VortexResult<FileWork>> {
        if self.limit.is_spent() {
            return future::ready(Ok(FileWork::empty())).boxed();
        }
        let shared = Arc::clone(self);
        async move {
            let reader = factory
                .open()
                .instrument(tracing::info_span!("LayoutReaderFactory::open"))
                .await?;
            match (reader, &shared.prepare) {
                (Some(reader), Some(prepare)) => prepare(reader),
                _ => Ok(FileWork::empty()),
            }
        }
        .boxed()
    }

    /// Opens a deferred file into its work as a task on the session's runtime.
    fn spawn_open_file(
        self: &Arc<Self>,
        factory: Arc<dyn LayoutReaderFactory>,
    ) -> BoxFuture<'static, VortexResult<FileWork>> {
        if self.limit.is_spent() {
            return future::ready(Ok(FileWork::empty())).boxed();
        }
        self.session.handle().spawn(self.open_file(factory)).boxed()
    }

    /// The next deferred file of an unordered pool to open, if any is left and the limit is not
    /// spent. Once it is spent, the files left count as read.
    fn next_deferred(&self) -> Option<Arc<dyn LayoutReaderFactory>> {
        let mut deferred = self.deferred.lock();
        if self.limit.is_spent() && !deferred.is_empty() {
            *self.progress.lock() += self.file_share * deferred.len() as f64;
            deferred.clear();
        }
        deferred.pop_front()
    }

    /// Takes a split and runs it as a task on the session's runtime.
    fn admit(self: &Arc<Self>, split: FileSplit) -> SplitOutput {
        self.take_split(split, true)
    }

    /// Takes a split. A split of a scan with a filter and a limit is built and run as
    /// [`spawn_lazy_split`] runs it; any other split runs as a task on the session's runtime when
    /// `spawn` is set, or else on the task that polls it, such as a lane, which is already one.
    fn take_split(
        self: &Arc<Self>,
        (file, index, progress): FileSplit,
        spawn: bool,
    ) -> SplitOutput {
        *self.progress.lock() += progress;
        let handle = self.session.handle();
        if let PoolLimit::Trim(_) = &self.limit {
            let Some(mask) = file.scan.split_mask(&file.ranges[index]) else {
                return future::ready(Ok(None)).boxed();
            };
            return async move {
                if file.is_pruned().await? {
                    return Ok(None);
                }
                spawn_lazy_split(&handle, move || split_exec(&file.ctx, mask).map(Some)).await
            }
            .boxed();
        }
        let shared = Arc::clone(self);
        async move {
            let task = file
                .tasks(&shared.limit)
                .await?
                .get(index)
                .and_then(|task| task.lock().take());
            match task {
                Some(task) if spawn => handle.spawn(task).await,
                Some(task) => task.await,
                None => Ok(None),
            }
        }
        .boxed()
    }

    /// Queues a unit of an unordered pool, counting the splits or open it holds as pending.
    fn queue(&self, sender: &Sender<Work>, work: Work) {
        let units = match &work {
            Work::File(splits, _) => splits.len(),
            Work::Plan(..) => 1,
        };
        self.pending.fetch_add(units, Ordering::AcqRel);
        send(sender, work);
    }

    /// Queues an opened file of an unordered pool as one turn among the files.
    fn queue_file(&self, sender: &Sender<Work>, work: FileWork) {
        let splits: VecDeque<FileSplit> = self.file_splits(work).into();
        if !splits.is_empty() {
            self.queue(sender, Work::File(splits, sender.clone()));
        }
    }

    fn progress(&self) -> f64 {
        if self.limit.is_trimmed_out() {
            return 1.0;
        }
        self.progress.lock().clamp(0.0, 1.0)
    }

    /// An ordered worker's output: skips splits with no rows, trims to a trimmed limit, and ends
    /// after the first error or once a trimmed limit is spent. A trimmed array is optimized again,
    /// on a CPU task, `splits_in_flight` at a time.
    fn output(
        self: Arc<Self>,
        outputs: BoxStream<'static, VortexResult<Option<ArrayRef>>>,
        splits_in_flight: usize,
    ) -> BoxStream<'static, VortexResult<ArrayRef>> {
        let trimmed = matches!(self.limit, PoolLimit::Trim(_));
        let handle = self.session.handle();
        let session = self.session.clone();
        let arrays = limit_output(outputs, self);
        if !trimmed {
            return arrays;
        }
        let identity: MapFn<ArrayRef> = Arc::new(Ok);
        let mapped = arrays.map(move |array| {
            spawn_map_trimmed(&handle, session.clone(), Arc::clone(&identity), array)
        });
        stop_on_error(mapped.buffered(splits_in_flight).boxed())
    }
}

/// Skips splits with no rows, trims to a trimmed limit, and ends after the first error or once
/// a trimmed limit is spent.
fn limit_output(
    outputs: BoxStream<'static, VortexResult<Option<ArrayRef>>>,
    shared: Arc<Shared>,
) -> BoxStream<'static, VortexResult<ArrayRef>> {
    stream::unfold(Some(outputs), move |outputs| {
        let shared = Arc::clone(&shared);
        async move {
            let mut outputs = outputs?;
            loop {
                if shared.limit.is_trimmed_out() {
                    return None;
                }
                let array = match outputs.next().await? {
                    Ok(Some(array)) => array,
                    Ok(None) => continue,
                    Err(err) => return Some((Err(err), None)),
                };
                let PoolLimit::Trim(limit) = &shared.limit else {
                    return Some((Ok(array), Some(outputs)));
                };
                let granted = limit.take(array.len());
                if granted == 0 {
                    return None;
                }
                if granted == array.len() {
                    return Some((Ok(array), Some(outputs)));
                }
                return Some((array.slice(0..granted), None));
            }
        }
    })
    .boxed()
}

/// A unit of an unordered pool's channel.
enum Work {
    /// An opened file's turn: the splits not yet taken, in row order. The lane that takes the
    /// turn takes the file's next run of splits and queues the turn again behind the other
    /// files' turns, so the lanes take runs from every opened file in turn rather than from one
    /// file after another. Each file then has only a few splits in flight, which keeps its first rows, and
    /// any dynamic filter they tighten, ahead of its later splits.
    File(VecDeque<FileSplit>, Sender<Work>),
    /// A deferred file that an opener has opened, to plan into its turn.
    Plan(LayoutReaderRef, Sender<Work>),
}

/// What a lane takes from a unit.
enum Taken {
    /// A run of a file's adjacent splits, in row order.
    Splits(Vec<FileSplit>),
    Plan(LayoutReaderRef, Sender<Work>),
}

impl Taken {
    /// The pending splits or open that this takes.
    fn units(&self) -> usize {
        match self {
            Taken::Splits(splits) => splits.len(),
            Taken::Plan(..) => 1,
        }
    }
}

impl Work {
    /// Takes the unit's next `run` splits, or its plan, queueing a file's turn again while it
    /// has splits left.
    fn take(self, run: usize) -> Option<Taken> {
        match self {
            Work::File(mut splits, sender) => {
                let taken: Vec<_> = splits.drain(..run.min(splits.len())).collect();
                if !splits.is_empty() {
                    let requeue = sender.clone();
                    send(&requeue, Work::File(splits, sender));
                }
                (!taken.is_empty()).then_some(Taken::Splits(taken))
            }
            Work::Plan(reader, sender) => Some(Taken::Plan(reader, sender)),
        }
    }
}

fn send(sender: &Sender<Work>, work: Work) {
    // The channel is unbounded and closes only once the pool is dropped, when the work is moot.
    drop(sender.try_send(work));
}

/// The lanes of an unordered pool. They start when its first worker does, as many as there are
/// splits and opens ready, and grow while those wait for a lane, up to `splits_in_flight`, so a
/// scan with few units spawns and wakes few tasks.
struct Lanes {
    shared: Arc<Shared>,
    units: Receiver<Work>,
    /// The most adjacent splits of a file that a lane takes and polls together.
    run: usize,
    /// The most lanes that run at once.
    max: usize,
    /// Lanes waiting for a unit.
    waiting: AtomicUsize,
    /// Lanes started, so a lane can see without a lock that no more may start.
    started: AtomicUsize,
    /// The lanes started, until the pool is dropped, which aborts them. `None` once it is.
    tasks: Mutex<Option<Vec<Task<()>>>>,
    /// The most deferred files that are opened at once, each by an opener of its own.
    open_concurrency: usize,
    /// The sender of the channel of units until the openers start, which hold it from then on.
    units_sender: Mutex<Option<Sender<Work>>>,
    /// The openers started, until the pool is dropped, which aborts them. `None` once it is.
    openers: Mutex<Option<Vec<Task<()>>>>,
}

impl Lanes {
    /// Starts lanes for the units that no waiting lane will take, up to the most that run at
    /// once. Each lane holds a sender of the output channel.
    fn grow(self: &Arc<Self>, output: &Sender<VortexResult<ArrayRef>>) {
        // Every lane calls this for every unit it takes, so it locks only to start lanes.
        if self.started.load(Ordering::Acquire) >= self.max || self.unserved() == 0 {
            return;
        }
        let mut tasks = self.tasks.lock();
        let Some(tasks) = tasks.as_mut() else {
            return;
        };
        let new = self
            .unserved()
            .div_ceil(self.run)
            .min(self.max.saturating_sub(tasks.len()));
        if new == 0 {
            return;
        }
        let handle = self.shared.session.handle();
        tasks.extend((0..new).map(|_| handle.spawn(lane(Arc::clone(self), output.clone()))));
        self.started.store(tasks.len(), Ordering::Release);
    }

    /// The pending splits and opens that the lanes waiting for a unit will not take, each of
    /// which takes up to a run.
    fn unserved(&self) -> usize {
        self.shared
            .pending
            .load(Ordering::Acquire)
            .saturating_sub(self.waiting.load(Ordering::Acquire) * self.run)
    }
}

/// A pool whose lanes run its units on the runtime, and whose workers share their output.
struct UnorderedPool {
    dtype: DType,
    shared: Arc<Shared>,
    units: Receiver<Work>,
    outputs: Receiver<VortexResult<ArrayRef>>,
    lanes: Arc<Lanes>,
    /// The sender of the output channel until the first worker starts the lanes, which hold it
    /// from then on.
    output: Mutex<Option<Sender<VortexResult<ArrayRef>>>>,
}

impl UnorderedPool {
    /// Runs a unit on the lane that took it. The lane is already a task on the runtime, so
    /// spawning the unit as another task would only add a hop.
    fn run(shared: &Arc<Shared>, taken: Taken) -> FuturesUnordered<SplitOutput> {
        match taken {
            Taken::Splits(splits) => splits
                .into_iter()
                .map(|split| shared.take_split(split, false))
                .collect(),
            Taken::Plan(reader, sender) => {
                let shared = Arc::clone(shared);
                iter::once(
                    async move {
                        let work = match &shared.prepare {
                            Some(prepare) if !shared.limit.is_spent() => prepare(reader)?,
                            _ => FileWork::empty(),
                        };
                        shared.queue_file(&sender, work);
                        Ok(None)
                    }
                    .boxed(),
                )
                .collect()
            }
        }
    }

    /// Starts the openers and the lanes, once. With no units and no deferred files, it starts
    /// none, and the output channel closes.
    fn start_lanes(&self) {
        // Dropping the pool's own senders leaves the openers and lanes holding the channels open.
        let Some(output) = self.output.lock().take() else {
            return;
        };
        if let Some(units) = self.lanes.units_sender.lock().take() {
            let count = self
                .lanes
                .open_concurrency
                .min(self.shared.deferred.lock().len());
            let handle = self.shared.session.handle();
            if let Some(openers) = self.lanes.openers.lock().as_mut() {
                openers.extend((0..count).map(|_| {
                    handle.spawn(opener(
                        Arc::clone(&self.lanes),
                        units.clone(),
                        output.clone(),
                    ))
                }));
            }
        }
        self.lanes.grow(&output);
    }
}

/// An opener: opens the pool's deferred files one after another, in file order, and queues each
/// opened file to be planned by a lane, so that planning one file overlaps opening the next, as
/// when a partition stream opens its files ahead of the partitions that plan and read them. The
/// openers are the only tasks that open files, so at most `open_concurrency` open at once, and
/// opens never wait behind the units in the channel. Ends once no deferred file is left, the
/// limit is spent, or a unit fails.
async fn opener(lanes: Arc<Lanes>, units: Sender<Work>, output: Sender<VortexResult<ArrayRef>>) {
    let shared = Arc::clone(&lanes.shared);
    while !shared.limit.is_trimmed_out() && !shared.failed.load(Ordering::Acquire) {
        let Some(factory) = shared.next_deferred() else {
            return;
        };
        let opened = factory
            .open()
            .instrument(tracing::info_span!("LayoutReaderFactory::open"))
            .await;
        match opened {
            Ok(Some(reader)) => {
                shared.queue(&units, Work::Plan(reader, units.clone()));
                lanes.grow(&output);
            }
            Ok(None) => shared.queue_file(&units, FileWork::empty()),
            Err(err) => {
                shared.failed.store(true, Ordering::Release);
                drop(output.send(Err(err)).await);
                return;
            }
        }
    }
}

/// A lane: runs units one at a time and sends their rows to the workers, trimming them to a
/// trimmed limit. Ends once the units run out, a limit is spent, a unit fails, or every worker
/// is gone.
async fn lane(lanes: Arc<Lanes>, output: Sender<VortexResult<ArrayRef>>) {
    let shared = Arc::clone(&lanes.shared);
    let handle = shared.session.handle();
    let identity: MapFn<ArrayRef> = Arc::new(Ok);
    while !shared.limit.is_trimmed_out() && !shared.failed.load(Ordering::Acquire) {
        lanes.waiting.fetch_add(1, Ordering::AcqRel);
        let work = lanes.units.recv().await;
        lanes.waiting.fetch_sub(1, Ordering::AcqRel);
        let Ok(work) = work else {
            return;
        };
        let Some(taken) = work.take(lanes.run) else {
            continue;
        };
        shared.pending.fetch_sub(taken.units(), Ordering::AcqRel);
        // Units the waiting lanes will not take get lanes of their own, such as the splits an
        // open has just queued.
        lanes.grow(&output);
        let mut running = UnorderedPool::run(&shared, taken);
        while let Some(array) = running.next().await {
            let array = match array {
                Ok(Some(array)) => Ok(array),
                Ok(None) => continue,
                Err(err) => Err(err),
            };
            if !emit(&shared, &handle, &identity, &output, array).await {
                return;
            }
        }
    }
}

/// Sends a lane's rows to the workers, trimmed to a trimmed limit. Returns whether the lane
/// should go on: not once the limit is spent, a unit failed, or every worker is gone.
async fn emit(
    shared: &Shared,
    handle: &Handle,
    identity: &MapFn<ArrayRef>,
    output: &Sender<VortexResult<ArrayRef>>,
    array: VortexResult<ArrayRef>,
) -> bool {
    let array = match (&shared.limit, array) {
        (PoolLimit::Trim(limit), Ok(array)) => {
            let granted = limit.take(array.len());
            if granted == 0 {
                return false;
            }
            let array = if granted < array.len() {
                array.slice(0..granted)
            } else {
                Ok(array)
            };
            spawn_map_trimmed(handle, shared.session.clone(), Arc::clone(identity), array).await
        }
        (_, array) => array,
    };
    let failed = array.is_err();
    if failed {
        shared.failed.store(true, Ordering::Release);
    }
    output.send(array).await.is_ok() && !failed
}

impl ScanPool for UnorderedPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        self.start_lanes();
        let dtype = self.dtype.clone();
        // The worker holds the pool, so the lanes run until the pool's last worker is dropped.
        let arrays = stream::unfold(self, |pool| async move {
            let array = pool.outputs.recv().await.ok()?;
            Some((array, pool))
        })
        .boxed();
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, stop_on_error(arrays)))
    }

    fn progress(&self) -> f64 {
        if self.is_exhausted() {
            return 1.0;
        }
        self.shared.progress()
    }

    fn is_exhausted(&self) -> bool {
        // The openers and lanes hold the output channel's senders, so it closes once every one
        // of them has ended.
        self.outputs.is_closed() && self.outputs.is_empty()
    }
}

impl Drop for UnorderedPool {
    fn drop(&mut self) {
        // Queued turns and plans hold senders of the channel of units itself; drain them so it
        // can be freed. The openers and lanes are aborted when they are dropped.
        self.units.close();
        while self.units.try_recv().is_ok() {}
        self.outputs.close();
        drop(self.lanes.openers.lock().take());
        drop(self.lanes.tasks.lock().take());
        drop(self.lanes.units_sender.lock().take());
    }
}

/// A pool that gives all of its units, in row order, to its first worker.
struct OrderedPool {
    dtype: DType,
    shared: Arc<Shared>,
    /// The files, until the first worker takes them.
    files: Mutex<Option<Vec<PoolFile>>>,
    splits_in_flight: usize,
    open_concurrency: usize,
}

impl ScanPool for OrderedPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let dtype = self.dtype.clone();
        let Some(files) = self.files.lock().take() else {
            return ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, stream::empty()));
        };
        let shared = Arc::clone(&self.shared);
        let opener = Arc::clone(&shared);
        let splitter = Arc::clone(&shared);
        let admitter = Arc::clone(&shared);
        let gate = Arc::clone(&shared);
        // Deferred files open `open_concurrency` at a time, ahead of the splits being read.
        let outputs = stream::iter(files)
            .map(move |file| match file {
                PoolFile::Opened(work) => future::ready(Ok(work)).boxed(),
                PoolFile::Deferred(factory) => opener.spawn_open_file(factory),
            })
            .buffered(self.open_concurrency)
            .flat_map(move |work| match work {
                Ok(work) => {
                    stream::iter(splitter.file_splits(work).into_iter().map(Ok)).left_stream()
                }
                Err(err) => stream::once(future::ready(Err(err))).right_stream(),
            })
            .take_while(move |_| future::ready(!gate.limit.is_trimmed_out()))
            .map(move |split| match split {
                Ok(split) => admitter.admit(split),
                Err(err) => future::ready(Err(err)).boxed(),
            })
            .buffered(self.splits_in_flight)
            .boxed();
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype,
            shared.output(outputs, self.splits_in_flight),
        ))
    }

    fn progress(&self) -> f64 {
        self.shared.progress()
    }

    fn is_exhausted(&self) -> bool {
        // The first worker takes every file.
        self.shared.limit.is_trimmed_out() || self.files.lock().is_none()
    }
}

/// The single partition of a scan piece that is read through the partitions API: it opens the
/// piece and reads it with one worker.
pub(crate) struct PiecePartition {
    piece: DataSourceScanRef,
    index: usize,
}

impl PiecePartition {
    /// The partitions of `piece`: one, whose index is `index`.
    pub(crate) fn stream(piece: DataSourceScanRef, index: usize) -> PartitionStream {
        stream::iter([Ok(Box::new(PiecePartition { piece, index }) as PartitionRef)]).boxed()
    }
}

impl Partition for PiecePartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.index
    }

    fn row_count(&self) -> Precision<u64> {
        self.piece.row_count()
    }

    fn byte_size(&self) -> Precision<u64> {
        self.piece.byte_size()
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        Ok(self.piece.open()?.worker())
    }
}
