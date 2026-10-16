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
//! directly and supports projection, row ranges, and selections. Unsupported layouts return an
//! error; filters and limits are not part of this builder's API. Work is partitioned into fixed
//! chunks independently of the stored layout's chunk boundaries. Dense scans use up to 65,536
//! rows per chunk, in 8,192-row multiples, targeting 8 MiB of serialized data. Sparse identity
//! projections can use larger aligned chunks within that estimate, targeting four tasks per
//! worker to reduce repeated segment reads.

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
use vortex_io::request::IoService;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutRef;
use crate::plan::physical::lower;
use crate::plan::physical::optimize;
use crate::plan::physical::plan_row_idx_expression;
use crate::scan::arrow::to_record_batch;
use crate::scan::execution::ScanPlans;
pub use crate::scan::execution::SegmentLocation;
use crate::scan::execution::plan_projection;
use crate::scan::v2::io::SegmentScanIo;
use crate::scan::v2::io::segment_ranges;
use crate::scan::v2::split::SplitPlan;
use crate::segments::SegmentSource;

const ROW_BLOCK_SIZE: usize = 8192;
const MAX_ROWS_PER_CHUNK: usize = 8 * ROW_BLOCK_SIZE;
const TARGET_BYTES_PER_CHUNK: u64 = 8 * 1024 * 1024;

/// Limit estimated bytes per task without changing the writer-aligned row granularity.
fn rows_per_chunk(
    row_count: u64,
    locations: &[SegmentLocation],
    sparse_target: Option<u64>,
) -> usize {
    if row_count <= MAX_ROWS_PER_CHUNK as u64 {
        return MAX_ROWS_PER_CHUNK;
    }
    let bytes: u64 = locations
        .iter()
        .map(|location| u64::from(location.length))
        .sum();
    let bytes_per_row = bytes.div_ceil(row_count).max(1);
    let blocks = TARGET_BYTES_PER_CHUNK / bytes_per_row / ROW_BLOCK_SIZE as u64;
    let max_blocks = sparse_target.map_or(8, |rows| {
        // Grow sparse tasks only within the byte target, keeping power-of-two boundaries
        // aligned with the original work chunks and enough tasks to occupy the workers.
        let blocks = (rows / ROW_BLOCK_SIZE as u64).min(blocks).max(8);
        1_u64 << blocks.ilog2()
    });
    usize::try_from(blocks.clamp(1, max_blocks))
        .vortex_expect("the byte target bounds the block count to at most 1024")
        * ROW_BLOCK_SIZE
}

/// Fixed boundaries in the file's row domain, clipped to the requested range.
fn row_chunks(rows: Range<u64>, rows_per_chunk: usize) -> Vec<Range<u64>> {
    if rows.is_empty() {
        return Vec::new();
    }
    let first = rows.start / rows_per_chunk as u64 * rows_per_chunk as u64;
    (first..rows.end)
        .step_by(rows_per_chunk)
        .map(|start| {
            start.max(rows.start)..start.saturating_add(rows_per_chunk as u64).min(rows.end)
        })
        .collect()
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
}

/// Builds a projection scan on the planner-driven V2 executor.
///
/// This builder reads the file's layout and segments directly. Layouts are checked as their
/// plans are lowered, so unsupported layouts outside the requested range can remain unread.
/// Unsupported plans return an error when the stream is polled. Filters and row limits are
/// not supported.
pub struct ScanBuilder<A = ArrayRef> {
    session: VortexSession,
    file: ScanFile,
    projection: BoundExpression,
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

    /// Sets the number of active splits per worker thread. Must be greater than zero.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
        self
    }

    /// Maps each output batch on its split's task.
    fn map<B: 'static>(
        self,
        map_fn: impl Fn(A) -> VortexResult<B> + 'static + Send + Sync,
    ) -> ScanBuilder<B> {
        let old_map_fn = self.map_fn;
        ScanBuilder {
            session: self.session,
            file: self.file,
            projection: self.projection,
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
    /// The scan's splits, over the root row domain, in row order.
    splits: Vec<Range<u64>>,
    selection: Selection,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    ordered: bool,
    concurrency: usize,
}

impl<A: 'static + Send> PreparedScan<A> {
    /// Lowers the file's layout, plans the projection, and partitions the selected range into
    /// fixed-size row chunks.
    fn try_new(builder: ScanBuilder<A>) -> VortexResult<Self> {
        let file = builder.file;
        let root = lower(&file.layout)?;
        let row_count = root.row_count();
        // An identity projection needs no rewrites. Keep the lowered children lazy, including
        // statistics that a projection-only scan never reads.
        let identity = builder.projection.is_root() && builder.projection.dtype() == root.dtype();
        let projection = if identity {
            root
        } else {
            optimize(plan_row_idx_expression(builder.projection, root)?)?
        };
        let chunk_rows = if identity {
            let rows = builder.row_range.clone().unwrap_or(0..row_count);
            let selected = builder.selection.row_count_in_range(&rows);
            let span = rows.end.saturating_sub(rows.start);
            let sparse_target = (selected <= span / 16).then(|| {
                // Tiny selections cannot occupy the full worker budget. Larger aligned tasks
                // let nearby selected rows share their stored segments.
                let tasks = (4 * get_available_parallelism().unwrap_or(1)) as u64;
                span / tasks.min(selected).max(1)
            });
            rows_per_chunk(row_count, &file.locations, sparse_target)
        } else {
            MAX_ROWS_PER_CHUNK
        };
        let splits = row_chunks(builder.row_range.unwrap_or(0..row_count), chunk_rows);
        let io = Arc::new(SegmentScanIo::new(
            file.segments,
            segment_ranges(Arc::clone(&file.locations)),
        ));
        Ok(Self {
            session: builder.session.clone(),
            plans: ScanPlans {
                session: builder.session,
                locations: file.locations,
                projection,
                row_offset: builder.row_offset,
                values: Default::default(),
            },
            io,
            splits,
            selection: builder.selection,
            map_fn: builder.map_fn,
            ordered: builder.ordered,
            concurrency: builder.concurrency,
        })
    }

    /// The splits with selected rows, each ready to be admitted to a planning run.
    fn split_plans(&self) -> Vec<SplitPlan> {
        let selection = Arc::new(self.selection.clone());
        self.splits
            .iter()
            .filter_map(|rows| {
                if self.selection.row_count_in_range(rows) == 0 {
                    return None;
                }
                let scope = WorkScope {
                    file_ordinal: 0,
                    rows: rows.clone(),
                };
                Some(SplitPlan {
                    root: plan_projection(
                        self.plans.clone(),
                        scope.clone(),
                        Arc::clone(&selection),
                    ),
                    scope,
                })
            })
            .collect()
    }

    /// Runs every split as a task on the session's runtime and yields their mapped batches.
    ///
    /// Like the default executor, up to `concurrency` splits per thread run at once, and their
    /// batches are yielded in row order when the scan is ordered.
    fn execute_stream(self) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        let concurrency = self.concurrency * get_available_parallelism().unwrap_or(1);
        let handle = self.session.handle();
        let split_plans = self.split_plans();
        let io = self.io;
        let map_fn = self.map_fn;
        // Spawned as the buffer pulls them, so at most `concurrency` splits read at once.
        let tasks = stream::iter(split_plans).map(move |split| {
            let io = Arc::clone(&io);
            let map_fn = Arc::clone(&map_fn);
            handle.spawn(async move {
                split
                    .run(io)
                    .await?
                    .into_iter()
                    .map(|array| map_fn(array))
                    .collect::<VortexResult<Vec<_>>>()
            })
        });
        let batches = if self.ordered {
            tasks.buffered(concurrency).boxed()
        } else {
            tasks.buffer_unordered(concurrency).boxed()
        };
        Ok(batches
            .map_ok(|batches| stream::iter(batches.into_iter().map(Ok)))
            .try_flatten()
            .boxed())
    }
}
