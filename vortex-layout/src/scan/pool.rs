// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The work of an opened layout scan, for the scan pool.
//!
//! A pool's units are the splits of one or more files, and the files not yet opened. Opening a
//! deferred file is an expansion that yields the file's splits, and at most `open_concurrency`
//! opens are queued or running at once. Each split is spawned onto the session's runtime when a
//! worker takes it; the [pool](vortex_scan::pool::open_work_pool) shares the units among the
//! workers and never starts threads of its own. Each file has an equal share of the pool's
//! progress, divided among its splits by their rows.
//!
//! Splits run as [`RepeatedScan::execute_stream`] runs them:
//!
//! - A scan with a filter and a limit builds each split on a CPU thread only once a worker takes
//!   it ([`spawn_lazy_split`]), so no new split starts once the limit is spent, and optimizes the
//!   rows the limit keeps on a second CPU task ([`spawn_map_trimmed`]).
//! - Every other scan builds all of a file's split tasks together, when a worker first takes one
//!   of them, so the file's reads are registered at once and can coalesce. A limit is reserved
//!   then, in split order.

use std::any::Any;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_lock::OnceCell;
use futures::FutureExt;
use futures::StreamExt;
use futures::future;
use futures::future::BoxFuture;
use futures::stream;
use parking_lot::Mutex;
use tracing::Instrument;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Precision;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::pool::Expansion;
use vortex_scan::pool::MapOutput;
use vortex_scan::pool::PoolLimit;
use vortex_scan::pool::ScanPoolRef;
use vortex_scan::pool::WorkPoolOptions;
use vortex_scan::pool::WorkUnit;
use vortex_scan::pool::open_work_pool;
use vortex_scan::row_mask::RowMask;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::scan::limit::RowLimit;
use crate::scan::multi::LayoutReaderFactory;
use crate::scan::repeated_scan::MapFn;
use crate::scan::repeated_scan::RepeatedScan;
use crate::scan::repeated_scan::can_prune_file;
use crate::scan::repeated_scan::spawn_lazy_split;
use crate::scan::repeated_scan::spawn_map_trimmed;
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
    /// The request's bound on the splits the pool keeps in flight.
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
    let files_work = Arc::new(Files {
        limit: limit.clone(),
        prepare,
        session: options.session.clone(),
    });
    let file_share = 1.0 / files.len().max(1) as f64;
    let mut units = Vec::new();
    for file in files {
        match file {
            PoolFile::Opened(work) => units.extend(files_work.units(work, file_share)),
            PoolFile::Deferred(factory) => {
                let files_work = Arc::clone(&files_work);
                units.push(WorkUnit::expand(file_share, move || {
                    files_work.open(factory, file_share)
                }));
            }
        }
    }
    // A limit's trimmed arrays are optimized again, on a CPU task, as `execute_stream` does.
    let handle = options.session.handle();
    let map_handle = handle.clone();
    let session = options.session;
    let identity: MapFn<ArrayRef> = Arc::new(Ok);
    let map_trimmed: MapOutput = Arc::new(move |array| {
        spawn_map_trimmed(
            &map_handle,
            session.clone(),
            Arc::clone(&identity),
            Ok(array),
        )
        .boxed()
    });
    open_work_pool(
        WorkPoolOptions {
            dtype: options.dtype,
            ordered: options.ordered,
            limit,
            // The pool keeps as many splits in flight as one scan stream does: four per core, at
            // most the request's `split_concurrency`.
            split_concurrency: options.split_concurrency,
            handle,
            expand_concurrency: NonZeroUsize::new(options.open_concurrency)
                .unwrap_or(NonZeroUsize::MIN),
            map_trimmed: Some(map_trimmed),
        },
        units,
    )
}

/// What the units of a pool's files share.
struct Files {
    limit: PoolLimit,
    prepare: Option<PrepareFile>,
    /// The session whose runtime runs the pool's splits and opens.
    session: VortexSession,
}

impl Files {
    /// The units of an opened file's splits, in row order, dividing the file's `share` among them
    /// by their rows.
    fn units(self: &Arc<Self>, work: FileWork, share: f64) -> Vec<WorkUnit> {
        let Some(file) = work.file else {
            return vec![WorkUnit::skipped(share)];
        };
        let rows: u64 = file
            .ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum();
        if rows == 0 {
            return vec![WorkUnit::skipped(share)];
        }
        let per_row = share / rows as f64;
        (0..file.ranges.len())
            .map(|index| {
                let range = &file.ranges[index];
                let files = Arc::clone(self);
                let file = Arc::clone(&file);
                WorkUnit::run(per_row * (range.end - range.start) as f64, move || {
                    files
                        .admit(file, index)
                        .map(|array| array.map(|array| array.into_iter().collect()))
                        .boxed()
                })
            })
            .collect()
    }

    /// Opens a deferred file on the session's runtime into the units of its splits.
    fn open(self: &Arc<Self>, factory: Arc<dyn LayoutReaderFactory>, share: f64) -> Expansion {
        let files = Arc::clone(self);
        let opened = self.session.handle().spawn(async move {
            let reader = factory
                .open()
                .instrument(tracing::info_span!("LayoutReaderFactory::open"))
                .await?;
            match (reader, &files.prepare) {
                (Some(reader), Some(prepare)) => prepare(reader),
                _ => Ok(FileWork::empty()),
            }
        });
        let files = Arc::clone(self);
        async move { Ok(files.units(opened.await?, share)) }.boxed()
    }

    /// Runs a split on the session's runtime.
    fn admit(
        self: &Arc<Self>,
        file: Arc<FileScan>,
        index: usize,
    ) -> BoxFuture<'static, VortexResult<Option<ArrayRef>>> {
        let handle = self.session.handle();
        if let PoolLimit::Trim(_) = &self.limit {
            let Some(mask) = file.scan.split_mask(&file.ranges[index]) else {
                return future::ready(Ok(None)).boxed();
            };
            return spawn_lazy_split(&handle, move || split_exec(&file.ctx, mask).map(Some))
                .boxed();
        }
        let files = Arc::clone(self);
        async move {
            let task = file.tasks(&files.limit).await?[index].lock().take();
            match task {
                Some(task) => handle.spawn(task).await,
                None => Ok(None),
            }
        }
        .boxed()
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
