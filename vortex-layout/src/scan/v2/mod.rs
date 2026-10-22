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
//! directly and supports projection, filters, row ranges, and selections. Unsupported layouts return an
//! error; limits are not part of this builder's API. Work is partitioned into fixed
//! chunks of up to 65,536 rows, or 1,048,576 rows for filtered scans, independently of the stored
//! layout's chunk boundaries.
//! Full-file projections use smaller 8,192-row multiples when serialized bytes per row are high.
//!
//! A [`ScanFile`] carrying a [`ScanCache`] shares its lowered plan, filter plans and proved
//! zones, segment index, and dictionary values with every other scan of that file, like a cached
//! layout reader does for V1 scans.

mod cache;
mod io;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::future;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::ArrowError;
use arrow_schema::Field;
use arrow_schema::SchemaRef;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoService;
use vortex_io::runtime::Handle;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::planning::driver::Batch;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

pub use self::cache::ScanCache;
pub use self::cache::WeakScanCache;
use crate::LayoutRef;
use crate::plan::physical::lower;
use crate::plan::physical::optimize;
use crate::plan::physical::plan_row_idx_expression;
use crate::scan::arrow::to_record_batch;
use crate::scan::execution::FILTERED_ROWS_PER_CHUNK;
use crate::scan::execution::FilterPlans;
use crate::scan::execution::MAX_ROWS_PER_CHUNK;
use crate::scan::execution::PreparedScan;
use crate::scan::execution::ROW_BLOCK_SIZE;
pub use crate::scan::execution::SegmentLocation;
use crate::scan::execution::rows_per_chunk;
use crate::scan::v2::io::SegmentScanIo;
use crate::scan::v2::io::segment_ranges;
use crate::segments::SegmentSource;

/// The rows a byte range of the file stands for when an engine splits the file by bytes before
/// opening it. Consecutive byte ranges give consecutive row ranges that tile the file, with
/// boundaries on the grid's 8,192-row blocks so the executor's chunks stay block aligned.
pub fn partition_row_range(bytes: Range<u64>, file_size: u64, row_count: u64) -> Range<u64> {
    let row = |byte: u64| -> u64 {
        if byte >= file_size {
            return row_count;
        }
        let row = u128::from(byte) * u128::from(row_count) / u128::from(file_size);
        let row = u64::try_from(row).vortex_expect("a row of the file fits u64");
        row / ROW_BLOCK_SIZE as u64 * ROW_BLOCK_SIZE as u64
    };
    let start = row(bytes.start);
    start..row(bytes.end).max(start)
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
/// error when the stream is polled. Row limits are not supported.
pub struct ScanBuilder<A = ArrayRef> {
    session: VortexSession,
    file: ScanFile,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    ordered: bool,
    row_range: Option<Range<u64>>,
    selection: Selection,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    row_offset: u64,
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
            map_fn: Arc::new(Ok),
            row_offset: 0,
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
    /// The output dtype after projection, before any output mapping.
    pub fn dtype(&self) -> &DType {
        self.projection.dtype()
    }

    /// Sets a projection bound against the file's dtype.
    pub fn with_projection(mut self, projection: BoundExpression) -> Self {
        self.projection = projection;
        self
    }

    /// Evaluates a bound predicate before projecting the surviving rows.
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

    /// Kept for callers of the default executor: a V2 scan runs on one task, its chunks
    /// overlapping their reads inside one planning run, so this sets nothing.
    pub fn with_concurrency(self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
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
            row_offset: self.row_offset,
            map_fn: Arc::new(move |array| old_map_fn(array).and_then(&map_fn)),
        }
    }

    /// Runs the scan on the session's runtime, preparing it on the first poll.
    /// Preparation errors are returned as stream items.
    pub fn into_stream(self) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        let handle = self.session.handle();
        Ok(stream::once(async move {
            handle
                .spawn_cpu(move || ScanRun::try_new(self)?.execute_stream())
                .await
        })
        .try_flatten()
        .boxed())
    }
}

/// A scan prepared over its file, ready to run.
struct ScanRun<A: 'static + Send> {
    session: VortexSession,
    scan: Arc<PreparedScan>,
    /// Serves the scan's reads.
    io: Arc<dyn IoService>,
    /// The rows of the root row domain the scan covers.
    rows: Range<u64>,
    selection: Selection,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    ordered: bool,
    /// Held until the scan ends, so scans of the file that start meanwhile share it.
    cache: ScanCache,
}

impl<A: 'static + Send> ScanRun<A> {
    /// Lowers the file's layout and plans the filter and the projection, from the file's cache
    /// where an earlier scan planned them already. Touches no segment.
    fn try_new(builder: ScanBuilder<A>) -> VortexResult<Self> {
        let file = builder.file;
        let cache = file.cache.unwrap_or_default();
        let root = match cache.root() {
            Some(root) => root,
            None => cache.insert_root(lower(&file.layout)?),
        };
        let row_count = root.row_count();
        let filter = builder
            .filter
            .map(|filter| -> VortexResult<_> {
                match cache.filter_plans(&filter) {
                    Some(plans) => Ok(plans),
                    None => {
                        let plans =
                            Arc::new(FilterPlans::new(filter.clone(), &root, &builder.session)?);
                        Ok(cache.insert_filter_plans(filter, plans))
                    }
                }
            })
            .transpose()?;
        // Optimizing drops the zone tables a projection never reads and keeps the chunks of a
        // chunked column lazy, so an identity projection is optimized like any other.
        let identity = builder.projection.is_root() && builder.projection.dtype() == root.dtype();
        let projection = match cache.projection(&builder.projection) {
            Some(plan) => plan,
            None => {
                let plan = optimize(
                    plan_row_idx_expression(builder.projection.clone(), root)?,
                    &builder.session,
                )?;
                cache.insert_projection(builder.projection, plan)
            }
        };
        let chunk_rows = if filter.is_some() {
            FILTERED_ROWS_PER_CHUNK
        } else if identity {
            rows_per_chunk(row_count, &file.locations)
        } else {
            MAX_ROWS_PER_CHUNK
        };
        let ranges = match cache.ranges() {
            Some(ranges) => ranges,
            None => cache.insert_ranges(segment_ranges(&file.locations)),
        };
        let io = Arc::new(SegmentScanIo::new(file.segments, ranges));
        Ok(Self {
            session: builder.session.clone(),
            scan: Arc::new(PreparedScan {
                session: builder.session,
                locations: file.locations,
                projection,
                filter,
                row_offset: builder.row_offset,
                values: cache.values(),
                chunk_rows,
            }),
            io,
            rows: builder.row_range.unwrap_or(0..row_count),
            selection: builder.selection,
            map_fn: builder.map_fn,
            ordered: builder.ordered,
            cache,
        })
    }

    /// Runs the scan on a planning run the returned stream owns and drives as it is polled.
    ///
    /// Every CPU step of the run, the planners' and morsels' compute, is a [`Handle::spawn_cpu`]
    /// task; every read is a `spawn_io` task of the segment source; the stream's own poll only
    /// awaits one or the other, so the engine thread polling it does no scan work itself and the
    /// scan advances only as fast as its consumer pulls. Batches of an unordered scan are yielded
    /// as morsels finish; an ordered scan yields them in row order once every morsel has.
    fn execute_stream(self) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        let mut run = Run::new();
        run.admit(
            self.scan.planner(self.rows.clone(), self.selection),
            WorkScope {
                file_ordinal: 0,
                rows: self.rows.clone(),
            },
            self.io.session(),
        );
        let driver = Driver {
            handle: self.session.handle(),
            run: Some(run),
            map_fn: self.map_fn,
            ordered: self.ordered,
            held: VecDeque::new(),
            done: false,
            cache: self.cache,
        };
        Ok(stream::try_unfold(driver, Driver::next).boxed())
    }
}

/// The state of a scan's stream: its planning run and what it holds back.
struct Driver<A> {
    handle: Handle,
    /// Taken while a CPU step runs on its task and put back with its result.
    run: Option<Run>,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    ordered: bool,
    /// Batches of an ordered scan, in row order once the run is done.
    held: VecDeque<Batch>,
    done: bool,
    /// Held until the scan ends, so scans of the file that start meanwhile share it.
    #[expect(dead_code, reason = "kept alive, never read")]
    cache: ScanCache,
}

impl<A: 'static + Send> Driver<A> {
    /// The next mapped batch, or `None` once the run has retired everything.
    async fn next(mut self) -> VortexResult<Option<(A, Self)>> {
        loop {
            if self.done {
                return match self.held.pop_front() {
                    Some(batch) => Ok(Some(((self.map_fn)(batch.array)?, self))),
                    None => Ok(None),
                };
            }
            let run = self
                .run
                .take()
                .ok_or_else(|| vortex_err!("the scan's run is missing"))?;
            let (run, progress) = self
                .handle
                .spawn_cpu(move || {
                    let mut run = run;
                    let progress = run.advance();
                    (run, progress)
                })
                .await;
            let run = self.run.insert(run);
            match progress? {
                Progress::Batch(batch) if self.ordered => self.held.push_back(batch),
                Progress::Batch(batch) => {
                    return Ok(Some(((self.map_fn)(batch.array)?, self)));
                }
                Progress::RootDone(_) | Progress::Idle => {
                    self.done = true;
                    // Morsels finish in whatever order their reads arrive.
                    self.held
                        .make_contiguous()
                        .sort_by_key(|batch| batch.scope.rows.start);
                }
                Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
    }
}
