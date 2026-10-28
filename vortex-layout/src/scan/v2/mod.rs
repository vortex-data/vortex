// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The planner-driven scan executor.
//!
//! A V2 scan lowers the file's layout once into a physical plan and runs every split of it
//! through the planning protocol: the [`Run`](vortex_scan::planning::driver::Run) driver advances
//! the split's planners and morsels, and every segment read is a protocol request served from the
//! file's [`SegmentSource`]. The layout reader is never asked to evaluate anything.
//!
//! Import this module's [`ScanBuilder`] to select V2 explicitly. It accepts a [`ScanFile`]
//! directly and supports projection, predicates, row ranges, and selections. Unsupported layouts
//! return an error; row limits remain the caller's responsibility. Work is partitioned into fixed
//! chunks of up to 65,536 rows, independently of the stored layout's chunk boundaries.
//! Full-file projections use smaller 8,192-row multiples when serialized bytes per row are high.
//! A file planner emits selected chunk planners incrementally to tasks on the runtime, without
//! an upfront walk or allocation of all selected chunk planners.
//!
//! A [`ScanFile`] carrying a [`ScanCache`] shares its lowered plan, segment index, dictionary
//! values, and default segment source with other scans of that file, like a cached layout reader
//! does for V1 scans.

mod cache;
mod io;
mod split;
#[cfg(test)]
mod tests;

use std::ops::Range;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::ArrowError;
use arrow_schema::Field;
use arrow_schema::SchemaRef;
use futures::Stream;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::transform::find_between;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_io::request::IoService;
use vortex_io::request::ScopedIoCache;
use vortex_io::session::RuntimeSessionExt;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
use vortex_io::uring::UringWorkerPool;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

pub use self::cache::ScanCache;
use crate::LayoutRef;
use crate::plan::physical::lower;
use crate::plan::physical::optimize;
use crate::plan::physical::plan_row_idx_expression;
use crate::scan::arrow::to_record_batch;
use crate::scan::execution::FilePlanner;
use crate::scan::execution::FilterPlans;
use crate::scan::execution::ScanPlans;
pub use crate::scan::execution::SegmentLocation;
use crate::scan::v2::io::SegmentScanIo;
use crate::scan::v2::io::segment_ranges;
use crate::scan::v2::split::SplitPlan;
use crate::scan::filter::FilterExpr;
use crate::segments::SegmentSource;

const ROW_BLOCK_SIZE: usize = 8192;
const MAX_ROWS_PER_CHUNK: usize = 8 * ROW_BLOCK_SIZE;
const TARGET_BYTES_PER_CHUNK: u64 = 8 * 1024 * 1024;

/// Limit estimated bytes per task without changing the writer-aligned row granularity.
fn rows_per_chunk(row_count: u64, locations: &[SegmentLocation]) -> usize {
    if row_count <= MAX_ROWS_PER_CHUNK as u64 {
        return MAX_ROWS_PER_CHUNK;
    }
    let bytes: u64 = locations
        .iter()
        .map(|location| u64::from(location.length))
        .sum();
    let bytes_per_row = bytes.div_ceil(row_count).max(1);
    let blocks = TARGET_BYTES_PER_CHUNK / bytes_per_row / ROW_BLOCK_SIZE as u64;
    blocks.clamp(1, 8) as usize * ROW_BLOCK_SIZE
}

/// The parts of a file the planner-driven executor reads through.
#[derive(Clone)]
pub struct ScanFile {
    /// The root layout to scan.
    pub layout: LayoutRef,
    /// The location of every segment, indexed by segment id.
    pub locations: Arc<[SegmentLocation]>,
    /// Fetches segment bytes by id.
    pub segments: Arc<dyn SegmentSource>,
    /// State shared by every scan of this file, or `None` to rebuild it for each scan.
    pub cache: Option<ScanCache>,
}

/// Builds a projection scan on the planner-driven V2 executor.
///
/// This builder reads the file's layout and segments directly. Unsupported layouts return an
/// error when the stream is polled. Predicates run before projection over the selected rows.
pub struct ScanBuilder<A = ArrayRef> {
    session: VortexSession,
    file: ScanFile,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    ordered: bool,
    row_range: Option<Range<u64>>,
    selection: Selection,
    concurrency: usize,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    row_offset: u64,
    subsegment_reads: bool,
    io: Option<Arc<dyn IoService>>,
    io_cache_bytes: usize,
    chunk_rows: Option<usize>,
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    workers: Option<UringWorkerPool>,
}

impl ScanBuilder<ArrayRef> {
    /// Creates a V2 scan over the file's layout and segment source.
    pub fn new(session: VortexSession, file: ScanFile) -> Self {
        Self {
            projection: BoundExpression::new_root(file.layout.dtype().clone()),
            filter: None,
            session,
            file,
            ordered: true,
            row_range: None,
            selection: Selection::default(),
            concurrency: 4,
            map_fn: Arc::new(Ok),
            row_offset: 0,
            subsegment_reads: true,
            io: None,
            io_cache_bytes: 0,
            chunk_rows: None,
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            workers: None,
        }
    }

    /// Converts each output batch to Arrow on the split's task.
    pub fn into_record_batch_stream(
        self,
        schema: SchemaRef,
    ) -> VortexResult<impl Stream<Item = Result<RecordBatch, ArrowError>> + Send + 'static> {
        let struct_field = Field::new_struct("", schema.fields().clone(), false);
        let session = self.session.clone();
        Ok(self
            .map(move |chunk| {
                let mut ctx = session.create_execution_ctx();
                to_record_batch(chunk, &struct_field, &mut ctx)
            })
            .into_stream()?
            .map_err(|error| ArrowError::ExternalError(Box::new(error))))
    }
}

impl<A: 'static + Send> ScanBuilder<A> {
    /// Uses a positional I/O service instead of the file's segment source.
    /// The service must address the same immutable file as the layout and segment locations.
    /// Each split receives an independent registration scope.
    pub fn with_io_service(mut self, io: Arc<dyn IoService>) -> Self {
        self.io = Some(io);
        self
    }

    /// Runs each split on one scan worker, driving that worker's ring between compute steps.
    /// With multiple active jobs configured, other splits can run while one awaits I/O.
    /// With one active job, the split uses the service's native blocking completion wait.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub fn with_uring_workers(mut self, workers: UringWorkerPool) -> Self {
        self.workers = Some(workers);
        self
    }

    /// The output dtype after projection, before any output mapping.
    pub fn dtype(&self) -> &DType {
        self.projection.dtype()
    }

    /// Sets a projection bound against the file's dtype.
    pub fn with_projection(mut self, projection: BoundExpression) -> Self {
        self.projection = projection;
        self
    }

    /// Evaluates a Boolean predicate before requesting the projection's rows.
    pub fn with_filter(mut self, filter: BoundExpression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets or clears the predicate evaluated before projection.
    pub fn with_some_filter(mut self, filter: Option<BoundExpression>) -> Self {
        self.filter = filter;
        self
    }

    /// Configures whether batches are yielded in file order.
    pub fn with_ordered(mut self, ordered: bool) -> Self {
        self.ordered = ordered;
        self
    }

    /// Restricts scanning to a contiguous file-local row range.
    pub fn with_row_range(mut self, row_range: Range<u64>) -> Self {
        self.row_range = Some(row_range);
        self
    }

    /// Applies a row selection within the scan's row range.
    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// Selects rows by strictly sorted file-local indices.
    pub fn with_row_indices(self, row_indices: StrictSortedBuffer<u64>) -> Self {
        self.with_selection(Selection::IncludeByIndex(row_indices))
    }

    /// Sets the root row offset used by row-index expressions.
    pub fn with_row_offset(mut self, row_offset: u64) -> Self {
        self.row_offset = row_offset;
        self
    }

    /// Sets the number of active tasks per worker thread. Must be greater than zero.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
        self
    }

    /// Enables verified sparse reads within supported primitive segments.
    pub fn with_subsegment_reads(mut self, enabled: bool) -> Self {
        self.subsegment_reads = enabled;
        self
    }

    /// Retains completed reads for reuse by one row-work chunk's predicates and projection.
    /// The requested-byte budget applies independently to each active chunk; zero disables it.
    /// Retained bytes are released when that chunk completes, rather than at file or scan end.
    pub fn with_io_cache_bytes(mut self, budget: usize) -> Self {
        self.io_cache_bytes = budget;
        self
    }

    /// Overrides the number of file rows in each independently scheduled work chunk.
    /// Larger chunks can reuse storage segments across more rows, at the cost of larger
    /// working sets and fewer independent tasks. Must be greater than zero.
    pub fn with_chunk_rows(mut self, rows: usize) -> Self {
        assert!(rows > 0);
        self.chunk_rows = Some(rows);
        self
    }

    /// Maps each output batch on its split's task.
    pub fn map<B: 'static>(
        self,
        map_fn: impl Fn(A) -> VortexResult<B> + 'static + Send + Sync,
    ) -> ScanBuilder<B> {
        let old_map_fn = self.map_fn;
        ScanBuilder {
            session: self.session,
            file: self.file,
            projection: self.projection,
            filter: self.filter,
            ordered: self.ordered,
            row_range: self.row_range,
            selection: self.selection,
            concurrency: self.concurrency,
            row_offset: self.row_offset,
            subsegment_reads: self.subsegment_reads,
            io: self.io,
            io_cache_bytes: self.io_cache_bytes,
            chunk_rows: self.chunk_rows,
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            workers: self.workers,
            map_fn: Arc::new(move |array| old_map_fn(array).and_then(&map_fn)),
        }
    }

    /// Runs the scan on the session's runtime, preparing it on the first poll.
    /// Preparation errors are returned as stream items.
    pub fn into_stream(self) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        let handle = self.session.handle();
        Ok(stream::once(async move {
            handle
                .spawn_cpu(move || PreparedScan::try_new(self)?.execute_stream())
                .await
        })
        .try_flatten()
        .boxed())
    }
}

/// A scan planned over its file, ready to run its splits.
struct PreparedScan<A: 'static + Send> {
    session: VortexSession,
    plans: ScanPlans,
    /// Serves the splits' reads.
    io: Arc<dyn IoService>,
    rows: Range<u64>,
    chunk_rows: usize,
    selection: Selection,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    ordered: bool,
    concurrency: usize,
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    workers: Option<UringWorkerPool>,
}

impl<A: 'static + Send> PreparedScan<A> {
    /// Lowers the file's layout and plans the projection without enumerating work chunks.
    fn try_new(builder: ScanBuilder<A>) -> VortexResult<Self> {
        let file = builder.file;
        let cache = file.cache.unwrap_or_default();
        let root = match cache.root() {
            Some(root) => root,
            None => cache.insert_root(lower(&file.layout)?),
        };
        let row_count = root.row_count();
        let filter = builder.filter.map(|expression| -> VortexResult<_> {
            vortex_ensure!(expression.dtype().is_boolean(), "Scan filter must be Boolean");
            let expressions = FilterExpr::new(find_between(expression)?);
            let plans = expressions.conjuncts().iter().map(|expression| {
                if let Some(plan) = cache.projection(expression) {
                    return Ok(plan);
                }
                let plan = optimize(plan_row_idx_expression(expression.clone(), root.clone())?)?;
                Ok(cache.insert_projection(expression.clone(), plan))
            }).collect::<VortexResult<Vec<_>>>()?;
            Ok(Arc::new(FilterPlans { expressions, plans }))
        }).transpose()?;
        // An identity projection needs no rewrites. Keep the lowered children lazy, including
        // statistics that a projection-only scan never reads.
        let identity = builder.projection.is_root() && builder.projection.dtype() == root.dtype();
        let projection = if identity {
            root
        } else {
            match cache.projection(&builder.projection) {
                Some(plan) => plan,
                None => {
                    let plan =
                        optimize(plan_row_idx_expression(builder.projection.clone(), root)?)?;
                    cache.insert_projection(builder.projection, plan)
                }
            }
        };
        let chunk_rows = builder.chunk_rows.unwrap_or_else(|| if identity {
            rows_per_chunk(row_count, &file.locations)
        } else {
            MAX_ROWS_PER_CHUNK
        });
        let segments = if builder.io.is_none() {
            cache.insert_segments(file.segments)
        } else {
            file.segments
        };
        let subsegment_reads = builder.subsegment_reads
            && (builder.io.is_some() || segments.supports_ranges());
        let ranges = match cache.ranges() {
            Some(ranges) => ranges,
            None => cache.insert_ranges(segment_ranges(&file.locations)),
        };
        let io = builder.io.unwrap_or_else(|| Arc::new(SegmentScanIo::new(segments, ranges)));
        let io = if builder.io_cache_bytes == 0 { io } else {
            Arc::new(ScopedIoCache::new(io, builder.io_cache_bytes)) as Arc<dyn IoService>
        };
        Ok(Self {
            session: builder.session.clone(),
            plans: ScanPlans {
                session: builder.session,
                locations: file.locations,
                projection,
                filter,
                row_offset: builder.row_offset,
                values: cache.values(),
                subsegment_reads,
            },
            io,
            rows: builder.row_range.unwrap_or(0..row_count),
            chunk_rows,
            selection: builder.selection,
            map_fn: builder.map_fn,
            ordered: builder.ordered,
            concurrency: builder.concurrency,
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            workers: builder.workers,
        })
    }

    /// Runs one file planner and schedules its children independently as they are emitted.
    fn execute_stream(self) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        let worker_count = get_available_parallelism().unwrap_or(1);
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        let worker_count = self.workers.as_ref().map_or(worker_count, UringWorkerPool::workers);
        let concurrency = self.concurrency * worker_count;
        let handle = self.session.handle();
        let planner = FilePlanner::new(self.plans, self.rows, self.chunk_rows, self.selection);
        let io = self.io;
        let map_fn = self.map_fn;
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        let workers = self.workers;
        let tasks = planner
            .into_stream(concurrency)
            .map_ok(move |(scope, root)| {
                let io = Arc::clone(&io);
                let map_fn = Arc::clone(&map_fn);
                #[cfg(all(target_os = "linux", feature = "io-uring"))]
                if let Some(workers) = &workers {
                    if workers.max_active_jobs() > 1 {
                        return workers.spawn_async(move || async move {
                            SplitPlan { scope, root }.run(io).await?
                                .into_iter().map(|array| map_fn(array)).collect::<VortexResult<Vec<_>>>()
                        });
                    }
                    return workers.spawn(move || {
                        SplitPlan { scope, root }.run_blocking(io)?
                            .into_iter().map(|array| map_fn(array)).collect::<VortexResult<Vec<_>>>()
                    });
                }
                handle.spawn(async move {
                    SplitPlan { scope, root }
                        .run(io)
                        .await?
                        .into_iter()
                        .map(|array| map_fn(array))
                        .collect::<VortexResult<Vec<_>>>()
                }).boxed()
            });
        let batches = if self.ordered {
            tasks.try_buffered(concurrency).boxed()
        } else {
            tasks.try_buffer_unordered(concurrency).boxed()
        };
        Ok(batches
            .map_ok(|batches| stream::iter(batches.into_iter().map(Ok)))
            .try_flatten()
            .boxed())
    }
}
