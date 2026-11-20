// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The work that the workers of an opened layout scan share.
//!
//! A pool's units are the splits of one or more files, and the files not yet opened. A worker
//! takes units and keeps several splits in flight, each spawned onto the session's runtime; it
//! never starts threads of its own. Opening a deferred file is a unit like any other, and at most
//! `open_concurrency` opens are queued or running at once.
//!
//! Splits run as [`RepeatedScan::execute_stream`] runs them:
//!
//! - A scan with a filter and a limit builds each split on a CPU thread only once a worker takes
//!   it ([`spawn_lazy_split`]), so no new split starts once the limit is spent, and optimizes the
//!   rows the limit keeps on a second CPU task ([`spawn_map_trimmed`]).
//! - Every other scan builds all of a file's split tasks together, when a worker first takes one
//!   of them, so the file's reads are registered at once and can coalesce. A limit is reserved
//!   then, in split order.
//!
//! The pools differ in how workers share units:
//!
//! - An unordered pool hands units out through a channel. The worker that opens a deferred file
//!   sends the file's splits back into the channel, where any worker can take them, followed by
//!   the next deferred file. The channel closes once every open has finished, which ends the
//!   workers.
//! - An ordered pool gives all of its units, in row order, to its first worker; later workers
//!   get nothing. Parallel ordered reads cut the scan with `split` first.

use std::any::Any;
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_channel::Receiver;
use async_channel::Sender;
use async_lock::OnceCell;
use futures::FutureExt;
use futures::StreamExt;
use futures::future;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use tracing::Instrument;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Precision;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
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
}

impl FileScan {
    /// The tasks of the file's splits, built on first use. A file whose tasks are built once a
    /// reserved limit is spent builds none.
    async fn tasks(&self, limit: &PoolLimit) -> VortexResult<&[Mutex<Option<SplitFuture>>]> {
        let tasks = self
            .tasks
            .get_or_try_init(|| future::ready(self.build_tasks(limit)))
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
    // A worker takes the place of a partition that one host thread drains, so it keeps as many
    // splits in flight as a partition's stream does by default: four per core, at most the
    // request's `split_concurrency`. A host with a worker per core therefore has the in-flight
    // total of the partitions path, where each of DuckDB's threads drained one file's partition
    // and each of DataFusion's `target_partitions` streams drained one partition, each with up to
    // `4 × cores` splits in flight.
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
    let mut opens = 0;
    for file in files {
        match file {
            PoolFile::Opened(work) => {
                for split in shared.file_splits(work) {
                    send(&sender, Work::Split(split));
                }
            }
            PoolFile::Deferred(factory) if opens < open_concurrency => {
                opens += 1;
                send(&sender, Work::Open(factory, sender.clone()));
            }
            PoolFile::Deferred(factory) => shared.deferred.lock().push_back(factory),
        }
    }
    Arc::new(UnorderedPool {
        dtype: options.dtype,
        shared,
        receiver,
        splits_in_flight,
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
    /// An unordered pool's deferred files that are not yet queued for opening, in file order.
    deferred: Mutex<VecDeque<Arc<dyn LayoutReaderFactory>>>,
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

    /// Opens a deferred file into its work on the session's runtime. A file that starts after
    /// the limit is spent is not opened.
    fn open_file(
        self: &Arc<Self>,
        factory: Arc<dyn LayoutReaderFactory>,
    ) -> BoxFuture<'static, VortexResult<FileWork>> {
        if self.limit.is_spent() {
            return future::ready(Ok(FileWork::empty())).boxed();
        }
        let shared = Arc::clone(self);
        self.session
            .handle()
            .spawn(async move {
                let reader = factory
                    .open()
                    .instrument(tracing::info_span!("LayoutReaderFactory::open"))
                    .await?;
                match (reader, &shared.prepare) {
                    (Some(reader), Some(prepare)) => prepare(reader),
                    _ => Ok(FileWork::empty()),
                }
            })
            .boxed()
    }

    /// The next deferred file of an unordered pool to open, if any is left and the limit is not
    /// spent.
    fn next_deferred(&self) -> Option<Arc<dyn LayoutReaderFactory>> {
        let mut deferred = self.deferred.lock();
        if self.limit.is_spent() {
            deferred.clear();
        }
        deferred.pop_front()
    }

    /// Takes a split and runs it on the session's runtime.
    fn admit(self: &Arc<Self>, (file, index, progress): FileSplit) -> SplitOutput {
        *self.progress.lock() += progress;
        let handle = self.session.handle();
        if let PoolLimit::Trim(_) = &self.limit {
            let Some(mask) = file.scan.split_mask(&file.ranges[index]) else {
                return future::ready(Ok(None)).boxed();
            };
            return spawn_lazy_split(&handle, move || split_exec(&file.ctx, mask).map(Some))
                .boxed();
        }
        let shared = Arc::clone(self);
        async move {
            let task = file.tasks(&shared.limit).await?[index].lock().take();
            match task {
                Some(task) => handle.spawn(task).await,
                None => Ok(None),
            }
        }
        .boxed()
    }

    fn progress(&self) -> f64 {
        if self.limit.is_trimmed_out() {
            return 1.0;
        }
        self.progress.lock().clamp(0.0, 1.0)
    }

    /// A worker's output: skips splits with no rows, trims to a trimmed limit, and ends after the
    /// first error or once a trimmed limit is spent. A trimmed array is optimized again, on a CPU
    /// task, `splits_in_flight` at a time.
    fn output(
        self: Arc<Self>,
        outputs: BoxStream<'static, VortexResult<Option<ArrayRef>>>,
        ordered: bool,
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
        let mapped = if ordered {
            mapped.buffered(splits_in_flight).boxed()
        } else {
            mapped.buffer_unordered(splits_in_flight).boxed()
        };
        stop_on_error(mapped)
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

/// A unit of an unordered pool.
enum Work {
    Split(FileSplit),
    /// A deferred file, and where to send its splits once it is open. The sender keeps the
    /// channel open until the file's splits, and the next deferred file, are in it.
    Open(Arc<dyn LayoutReaderFactory>, Sender<Work>),
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
    splits_in_flight: usize,
}

impl UnorderedPool {
    fn run(shared: &Arc<Shared>, work: Work) -> SplitOutput {
        match work {
            Work::Split(split) => shared.admit(split),
            Work::Open(factory, sender) => {
                let shared = Arc::clone(shared);
                async move {
                    let opened = shared.open_file(factory).await.map(|work| {
                        for split in shared.file_splits(work) {
                            send(&sender, Work::Split(split));
                        }
                        None
                    });
                    // Queue the next file behind this one's splits, so opens keep pace with the
                    // workers and at most `open_concurrency` are queued or running at once.
                    if let Some(next) = shared.next_deferred() {
                        send(&sender, Work::Open(next, sender.clone()));
                    }
                    opened
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
        let splits_in_flight = self.splits_in_flight;
        // The worker holds the pool, so the pool is dropped only after its last worker.
        let outputs = receiver
            .take_while(move |_| future::ready(!self.shared.limit.is_trimmed_out()))
            .map(move |work| UnorderedPool::run(&runner, work))
            .buffer_unordered(splits_in_flight)
            .boxed();
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype,
            shared.output(outputs, false, splits_in_flight),
        ))
    }

    fn progress(&self) -> f64 {
        self.shared.progress()
    }

    fn is_exhausted(&self) -> bool {
        // Pending opens hold senders, so the channel closes only once every unit is in it.
        self.shared.limit.is_trimmed_out()
            || (self.receiver.is_closed() && self.receiver.is_empty())
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
                PoolFile::Deferred(factory) => opener.open_file(factory),
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
            shared.output(outputs, true, self.splits_in_flight),
        ))
    }

    fn progress(&self) -> f64 {
        self.shared.progress()
    }

    fn is_exhausted(&self) -> bool {
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
