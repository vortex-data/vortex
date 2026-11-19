// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The work that the workers of an opened layout scan share.
//!
//! A pool's units are the splits of one or more files, and the files not yet opened. A worker
//! takes units and keeps up to `splits_in_flight` of them running on its own thread: it
//! builds a split's task only when it takes the split, and never spawns. Opening a deferred file
//! is a unit like any other.
//!
//! - An unordered pool hands units out through a channel. The worker that opens a deferred file
//!   sends the file's splits back into the channel, where any worker can take them. The channel
//!   closes once every open has finished, which ends the workers.
//! - An ordered pool gives all of its units, in row order, to its first worker; later workers
//!   get nothing. Parallel ordered reads cut the scan with `split` first.

use std::any::Any;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_channel::Receiver;
use async_channel::Sender;
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
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::pool::ScanPool;
use vortex_scan::pool::ScanPoolRef;
use vortex_scan::row_mask::RowMask;

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

/// The most splits each worker keeps in flight when the request does not say; stands in for the
/// scan engine's memory budget.
const SPLITS_IN_FLIGHT_PER_WORKER: usize = 4;

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

/// A file of a pool, in file order.
pub(crate) enum PoolFile {
    Opened(FileWork),
    Deferred(Arc<dyn LayoutReaderFactory>),
}

/// How a pool reads its splits.
pub(crate) struct PoolOptions {
    pub(crate) dtype: DType,
    pub(crate) ordered: bool,
    pub(crate) limit: Option<u64>,
    /// Whether the scan filters rows, so a split's output row count is unknown up front.
    pub(crate) has_filter: bool,
    /// The most splits each worker keeps in flight, or the default when `None`.
    pub(crate) splits_in_flight: Option<NonZeroUsize>,
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
    let splits_in_flight = options
        .splits_in_flight
        .map_or(SPLITS_IN_FLIGHT_PER_WORKER, NonZeroUsize::get);
    let shared = Arc::new(Shared {
        limit,
        prepare,
        file_share: 1.0 / files.len().max(1) as f64,
        progress: Mutex::new(0.0),
    });
    if options.ordered {
        return Arc::new(OrderedPool {
            dtype: options.dtype,
            shared,
            files: Mutex::new(Some(files)),
            splits_in_flight,
        });
    }

    let (sender, receiver) = async_channel::unbounded();
    for file in files {
        match file {
            PoolFile::Opened(work) => {
                for (file, range, progress) in shared.file_splits(work) {
                    send(&sender, Work::Split(file, range, progress));
                }
            }
            PoolFile::Deferred(factory) => send(&sender, Work::Open(factory, sender.clone())),
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
    /// A split's selected rows are its output rows, so each split reserves them from the budget
    /// when a worker takes it, and rows past the limit are never decoded. Admission order is
    /// row order for an ordered pool, so it returns exactly the first rows.
    Reserve(RowLimit),
    /// A split's output rows are unknown until its filter has run, so workers take rows from the
    /// budget as they yield them and discard the excess.
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

/// A split's output: `None` when it has no rows to return.
type SplitFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// A split of a file, with the progress that taking it adds.
type FileSplit = (Arc<FileScan>, Range<u64>, f64);

/// What every worker of a pool shares.
struct Shared {
    limit: PoolLimit,
    prepare: Option<PrepareFile>,
    /// The progress that one file adds.
    file_share: f64,
    progress: Mutex<f64>,
}

impl Shared {
    /// The splits of an opened file, in row order.
    fn file_splits(&self, work: FileWork) -> Vec<FileSplit> {
        let rows: u64 = work
            .ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum();
        match work.scan {
            Some(scan) if rows > 0 => {
                let per_row = self.file_share / rows as f64;
                work.ranges
                    .into_iter()
                    .map(|range| {
                        let progress = per_row * (range.end - range.start) as f64;
                        (Arc::clone(&scan), range, progress)
                    })
                    .collect()
            }
            _ => {
                *self.progress.lock() += self.file_share;
                Vec::new()
            }
        }
    }

    /// Opens a deferred file into its work.
    fn open_file(
        self: &Arc<Self>,
        factory: Arc<dyn LayoutReaderFactory>,
    ) -> BoxFuture<'static, VortexResult<FileWork>> {
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

    /// Takes a split: reserves its rows from a reserved limit and builds its task.
    fn admit(&self, (file, range, progress): FileSplit) -> SplitFuture {
        *self.progress.lock() += progress;
        let Some(mask) = file.scan.split_mask(&range) else {
            return future::ready(Ok(None)).boxed();
        };
        let mask = match &self.limit {
            PoolLimit::Reserve(limit) => {
                let reserved = limit.limit(mask.mask().clone());
                if reserved.all_false() {
                    return future::ready(Ok(None)).boxed();
                }
                RowMask::new(range.start, reserved)
            }
            PoolLimit::Unlimited | PoolLimit::Trim(_) => mask,
        };
        match file.scan.split_task(&file.ctx, mask) {
            Ok(task) => task,
            Err(err) => future::ready(Err(err)).boxed(),
        }
    }

    fn progress(&self) -> f64 {
        if self.limit.is_exhausted() {
            return 1.0;
        }
        self.progress.lock().clamp(0.0, 1.0)
    }
}

/// A worker's output: skips splits with no rows, trims to a trimmed limit, and ends after the
/// first error or once a trimmed limit is spent.
fn limit_output(
    outputs: BoxStream<'static, VortexResult<Option<ArrayRef>>>,
    shared: Arc<Shared>,
) -> BoxStream<'static, VortexResult<ArrayRef>> {
    stream::unfold(Some(outputs), move |outputs| {
        let shared = Arc::clone(&shared);
        async move {
            let mut outputs = outputs?;
            loop {
                if let PoolLimit::Trim(limit) = &shared.limit
                    && limit.is_exhausted()
                {
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
    Split(Arc<FileScan>, Range<u64>, f64),
    /// A deferred file, and where to send its splits once it is open. The sender keeps the
    /// channel open until the file's splits are in it.
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
    fn run(shared: &Arc<Shared>, work: Work) -> SplitFuture {
        match work {
            Work::Split(file, range, progress) => shared.admit((file, range, progress)),
            Work::Open(factory, sender) => {
                let shared = Arc::clone(shared);
                async move {
                    let work = shared.open_file(factory).await?;
                    for (file, range, progress) in shared.file_splits(work) {
                        send(&sender, Work::Split(file, range, progress));
                    }
                    Ok(None)
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
            .take_while(move |_| future::ready(!self.shared.limit.is_exhausted()))
            .map(move |work| UnorderedPool::run(&runner, work))
            .buffer_unordered(splits_in_flight)
            .boxed();
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype,
            limit_output(outputs, shared),
        ))
    }

    fn progress(&self) -> f64 {
        self.shared.progress()
    }

    fn is_exhausted(&self) -> bool {
        // Pending opens hold senders, so the channel closes only once every split is in it.
        self.shared.limit.is_exhausted() || (self.receiver.is_closed() && self.receiver.is_empty())
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
        let outputs = stream::iter(files)
            .then(move |file| match file {
                PoolFile::Opened(work) => future::ready(Ok(work)).boxed(),
                PoolFile::Deferred(factory) => opener.open_file(factory),
            })
            .flat_map(move |work| match work {
                Ok(work) => {
                    stream::iter(splitter.file_splits(work).into_iter().map(Ok)).left_stream()
                }
                Err(err) => stream::once(future::ready(Err(err))).right_stream(),
            })
            .take_while(move |_| future::ready(!shared.limit.is_exhausted()))
            .map(move |split| match split {
                Ok(split) => admitter.admit(split),
                Err(err) => future::ready(Err(err)).boxed(),
            })
            .buffered(self.splits_in_flight)
            .boxed();
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype,
            limit_output(outputs, Arc::clone(&self.shared)),
        ))
    }

    fn progress(&self) -> f64 {
        self.shared.progress()
    }

    fn is_exhausted(&self) -> bool {
        self.shared.limit.is_exhausted() || self.files.lock().is_none()
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
