// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The planner-driven scan executor.
//!
//! A V2 scan lowers the file's layout once into a physical plan and runs every split of it
//! through the planning protocol: the [`Run`] driver advances
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

use std::future;
use std::ops::Range;
use std::sync::Arc;
use std::task::Poll;

use arrow_array::RecordBatch;
use arrow_schema::ArrowError;
use arrow_schema::Field;
use arrow_schema::SchemaRef;
use futures::SinkExt;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
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
use vortex_io::request::IoSource;
use vortex_io::runtime::Handle;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

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
    concurrency: usize,
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
            concurrency: 4,
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

    /// Sets the number of chunks run at once per worker thread. Must be greater than zero.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
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
    concurrency: usize,
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
            concurrency: builder.concurrency,
            cache,
        })
    }

    /// Runs the scan on planning runs the returned stream owns.
    ///
    /// The scan's first planner runs on one run the stream drives as it is polled, each of its
    /// CPU steps a [`Handle::spawn_cpu`] task. Every chunk it spawns runs on a run and a session
    /// of its own, on one task of the runtime that drives the run to its end and awaits only its
    /// reads, up to `concurrency` chunks per worker thread at once; a chunk is started only when
    /// the consumer has room for its batches. Every read is a `spawn_io` task of the segment
    /// source. The engine thread polling the stream does no scan work itself. Batches of an
    /// unordered scan are yielded as morsels finish; an ordered scan yields them in row order
    /// once every morsel has.
    fn execute_stream(self) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        let handle = self.session.handle();
        let in_flight = self.concurrency * get_available_parallelism().unwrap_or(1);
        let root = self.scan.planner(self.rows.clone(), self.selection);
        let scope = WorkScope {
            file_ordinal: 0,
            rows: self.rows.clone(),
        };
        let io = self.io;
        let map_fn = self.map_fn;
        let (chunk_handle, chunk_io, chunk_map) =
            (handle.clone(), Arc::clone(&io), Arc::clone(&map_fn));
        let inline = in_flight <= 1;
        let batches = run_stream(handle, root, scope, io.session(), map_fn, inline)
            .map_ok(move |yielded| match yielded {
                Yielded::Batch(start, batch) => {
                    stream::once(future::ready(Ok((start, batch)))).boxed()
                }
                Yielded::Spawn(scope, chunk) => chunk_stream(
                    &chunk_handle,
                    chunk,
                    scope,
                    chunk_io.session(),
                    Arc::clone(&chunk_map),
                ),
            })
            .try_flatten_unordered(in_flight);
        let cache = self.cache;
        let batches = batches.chain(stream::poll_fn(move |_| {
            let _ = &cache;
            Poll::Ready(None)
        }));
        if !self.ordered {
            return Ok(batches.map_ok(|(_, batch)| batch).boxed());
        }
        Ok(stream::once(async move {
            let mut held: Vec<(u64, A)> = batches.try_collect().await?;
            // Morsels finish in whatever order their reads arrive.
            held.sort_by_key(|(start, _)| *start);
            VortexResult::Ok(stream::iter(held.into_iter().map(|(_, batch)| Ok(batch))))
        })
        .try_flatten()
        .boxed())
    }
}

/// What the root run's stream yields: a mapped batch with the row it starts at, or a chunk a
/// planner of the run spawned.
enum Yielded<A> {
    Batch(u64, A),
    Spawn(WorkScope, Box<dyn Planner>),
}

/// One CPU step of a run, taken on a `spawn_cpu` task with the run.
enum Step<A> {
    Batch(u64, A),
    Spawn(WorkScope, Box<dyn Planner>),
    RootDone,
    Idle,
    Waiting,
}

/// The state of the root run's stream.
struct RunState<A> {
    handle: Handle,
    /// Taken while a CPU step runs on its task and put back with its result.
    run: Option<Run>,
    io: Arc<dyn IoSource>,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// Whether a spawned chunk runs on this run, as another root, instead of being yielded.
    inline: bool,
}

/// Runs `root` over `scope` on a planning run of its own, reading through `io`, each CPU step
/// on a `spawn_cpu` task, and yields its mapped batches and, unless `inline`, the chunks it
/// spawns. Inline, a spawned chunk runs on this run beside the others: the run's steps already
/// interleave every chunk's work, and a run per chunk would only add a hand-off per chunk for
/// parallelism the engine provides.
fn run_stream<A: 'static + Send>(
    handle: Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoSource>,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    inline: bool,
) -> BoxStream<'static, VortexResult<Yielded<A>>> {
    let mut run = Run::new();
    run.admit(root, scope, Arc::clone(&io));
    let state = RunState {
        handle,
        run: Some(run),
        io,
        map_fn,
        inline,
    };
    stream::try_unfold(state, |mut state| async move {
        loop {
            let run = state
                .run
                .take()
                .ok_or_else(|| vortex_err!("the scan's run is missing"))?;
            let map_fn = Arc::clone(&state.map_fn);
            let (run, step) = state
                .handle
                .spawn_cpu(move || {
                    let mut run = run;
                    let step = run.advance().and_then(|progress| {
                        Ok(match progress {
                            Progress::Batch(batch) => {
                                Step::Batch(batch.scope.rows.start, map_fn(batch.array)?)
                            }
                            Progress::Spawn(scope, child) => Step::Spawn(scope, child),
                            Progress::RootDone(_) => Step::RootDone,
                            Progress::Idle => Step::Idle,
                            Progress::Waiting => Step::Waiting,
                        })
                    });
                    (run, step)
                })
                .await;
            let run = state.run.insert(run);
            match step? {
                Step::Batch(start, batch) => {
                    return Ok(Some((Yielded::Batch(start, batch), state)));
                }
                Step::Spawn(scope, child) if state.inline => {
                    run.admit(child, scope, Arc::clone(&state.io));
                }
                Step::Spawn(scope, child) => {
                    return Ok(Some((Yielded::Spawn(scope, child), state)));
                }
                Step::RootDone if run.live_roots() > 0 => {}
                Step::RootDone | Step::Idle => return Ok(None),
                Step::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
    })
    .boxed()
}

/// Runs a spawned chunk's `root` over `scope` on a planning run of its own, on one task that
/// drives the run to its end and awaits only its reads, and yields its mapped batches with the
/// row each starts at. A planner the chunk spawns in turn runs as another root of the same run.
///
/// One task per chunk, not one per step: a chunk's run waits for reads many times, and a task
/// hop at each wait is latency on the chunk's critical path that no other chunk's work fills.
fn chunk_stream<A: 'static + Send>(
    handle: &Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoSource>,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
) -> BoxStream<'static, VortexResult<(u64, A)>> {
    let (mut send, recv) = mpsc::channel(1);
    let task = handle.spawn(async move {
        let mut run = Run::new();
        run.admit(root, scope, Arc::clone(&io));
        loop {
            match run.advance()? {
                Progress::Batch(batch) => {
                    let start = batch.scope.rows.start;
                    let batch = map_fn(batch.array);
                    let failed = batch.is_err();
                    if send.send(batch.map(|batch| (start, batch))).await.is_err() || failed {
                        return Ok(());
                    }
                }
                Progress::Spawn(scope, child) => {
                    run.admit(child, scope, Arc::clone(&io));
                }
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => return Ok(()),
                Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
    });
    let errors = stream::once(task)
        .filter_map(|result: VortexResult<()>| future::ready(result.err().map(Err)));
    stream::select(recv, errors).boxed()
}
