use std::sync::Arc;

use futures::stream::BoxStream;
use futures::{stream, Stream};
use itertools::Itertools;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter, ArrayStreamExt};
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_expr::{ExprRef, Identity};
pub(crate) mod filter;
mod split_by;
mod task;
pub mod unified;

use futures::StreamExt;
pub use split_by::*;
pub use task::*;
use vortex_array::{Array, ContextRef};
use vortex_dtype::{DType, Field, FieldMask, FieldPath};
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_mask::Mask;

use crate::scan::filter::FilterExpr;
use crate::scan::unified::UnifiedDriverStream;
use crate::segments::{AsyncSegmentReader, SegmentId};
use crate::{ExprEvaluator, Layout, LayoutReader, LayoutReaderExt, RowMask};

pub trait ScanDriver: 'static + Sized {
    type Options: Default;

    fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader>;

    /// Return a future that drives the I/O stream for the segment reader.
    /// The future should return when the stream is complete, and can return an error to
    /// terminate the scan early.
    ///
    /// It is recommended that I/O is spawned and processed on its own thread, with this driver
    /// serving only as a mechanism to signal completion or error. There is no guarantee around
    /// how frequently this future will be polled, so it should not be used to drive I/O.
    ///
    /// TODO(ngates): make this a future
    fn io_stream(self) -> impl Stream<Item = VortexResult<()>> + 'static;
}

/// A struct for building a scan operation.
pub struct ScanBuilder<D: ScanDriver> {
    driver: D,
    driver_options: Option<D::Options>,
    task_executor: Option<Arc<dyn TaskExecutor>>,
    layout: Layout,
    ctx: ContextRef, // TODO(ngates): store this on larger context on Layout
    projection: ExprRef,
    filter: Option<ExprRef>,
    row_indices: Option<Buffer<u64>>,
    split_by: SplitBy,
    canonicalize: bool,
    // The number of splits to make progress on concurrently.
    concurrency: usize,
}

impl<D: ScanDriver> ScanBuilder<D> {
    pub fn new(driver: D, layout: Layout, ctx: ContextRef) -> Self {
        Self {
            driver,
            driver_options: None,
            task_executor: None,
            layout,
            ctx,
            projection: Identity::new_expr(),
            filter: None,
            row_indices: None,
            split_by: SplitBy::Layout,
            canonicalize: false,
            concurrency: 32,
        }
    }

    pub fn with_filter(mut self, filter: ExprRef) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_some_filter(mut self, filter: Option<ExprRef>) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_projection(mut self, projection: ExprRef) -> Self {
        self.projection = projection;
        self
    }

    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.row_indices = Some(row_indices);
        self
    }

    pub fn with_some_row_indices(mut self, row_indices: Option<Buffer<u64>>) -> Self {
        self.row_indices = row_indices;
        self
    }

    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    /// Set whether the scan should canonicalize the output.
    pub fn with_canonicalize(mut self, canonicalize: bool) -> Self {
        self.canonicalize = canonicalize;
        self
    }

    /// The number of row splits to make progress on concurrently.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    pub fn with_task_executor(mut self, task_executor: Arc<dyn TaskExecutor>) -> Self {
        self.task_executor = Some(task_executor);
        self
    }

    pub fn with_options(mut self, options: D::Options) -> Self {
        self.driver_options = Some(options);
        self
    }

    pub fn build(self) -> VortexResult<Scan<D>> {
        let projection = simplify_typed(self.projection, self.layout.dtype())?;
        let filter = self
            .filter
            .map(|f| simplify_typed(f, self.layout.dtype()))
            .transpose()?;
        let field_mask = field_mask(&projection, filter.as_ref(), self.layout.dtype())?;

        let row_indices = self.row_indices.clone();
        let splits = self.split_by.splits(&self.layout, &field_mask)?;
        let row_masks = splits
            .into_iter()
            .filter_map(move |row_range| {
                let Some(row_indices) = &row_indices else {
                    // If there is no row indices filter, then take the whole range
                    return Some(RowMask::new_valid_between(row_range.start, row_range.end));
                };

                // Otherwise, find the indices that are within the row range.
                if row_indices
                    .first()
                    .is_some_and(|&first| first >= row_range.end)
                    || row_indices
                        .last()
                        .is_some_and(|&last| row_range.start >= last)
                {
                    return None;
                }

                // For the given row range, find the indices that are within the row_indices.
                let start_idx = row_indices
                    .binary_search(&row_range.start)
                    .unwrap_or_else(|x| x);
                let end_idx = row_indices
                    .binary_search(&row_range.end)
                    .unwrap_or_else(|x| x);

                if start_idx == end_idx {
                    // No rows in range
                    return None;
                }

                // Construct a row mask for the range.
                let filter_mask = Mask::from_indices(
                    usize::try_from(row_range.end - row_range.start)
                        .vortex_expect("Split ranges are within usize"),
                    row_indices[start_idx..end_idx]
                        .iter()
                        .map(|&idx| {
                            usize::try_from(idx - row_range.start)
                                .vortex_expect("index within range")
                        })
                        .collect(),
                );
                Some(RowMask::new(filter_mask, row_range.start))
            })
            .collect_vec();

        Ok(Scan {
            driver: self.driver,
            task_executor: self
                .task_executor
                .unwrap_or_else(|| Arc::new(InlineTaskExecutor)),
            layout: self.layout,
            ctx: self.ctx,
            projection,
            filter,
            row_masks,
            canonicalize: self.canonicalize,
            concurrency: self.concurrency,
        })
    }

    /// Perform the scan operation and return a stream of arrays.
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + 'static> {
        self.build()?.into_array_stream()
    }

    pub async fn into_array(self) -> VortexResult<Array> {
        self.into_array_stream()?.into_array().await
    }
}

pub struct ScanExecutor {
    segment_reader: Arc<dyn AsyncSegmentReader>,
    task_executor: Arc<dyn TaskExecutor>,
}

impl ScanExecutor {
    /// Mostly used for testing, this creates a ScanExecutor with an inline task executor.
    pub fn inline(segments: Arc<dyn AsyncSegmentReader>) -> Arc<Self> {
        Arc::new(Self {
            segment_reader: segments,
            task_executor: Arc::new(InlineTaskExecutor),
        })
    }

    pub async fn get_segment(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
        self.segment_reader.get(id).await
    }

    pub async fn evaluate(&self, array: &Array, tasks: &[ScanTask]) -> VortexResult<Array> {
        self.task_executor.execute(array, tasks).await
    }
}

pub struct Scan<D> {
    driver: D,
    task_executor: Arc<dyn TaskExecutor>,
    layout: Layout,
    ctx: ContextRef,
    // Guaranteed to be simplified
    projection: ExprRef,
    // Guaranteed to be simplified
    filter: Option<ExprRef>,
    row_masks: Vec<RowMask>,
    canonicalize: bool,
    concurrency: usize,
}

impl<D: ScanDriver> Scan<D> {
    /// Perform the scan operation and return a stream of arrays.
    ///
    /// The returned stream should be considered to perform I/O-bound operations and requires
    /// frequent polling to make progress.
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + 'static> {
        // Create a single LayoutReader that is reused for the entire scan.
        let executor = Arc::new(ScanExecutor {
            segment_reader: self.driver.segment_reader(),
            task_executor: self.task_executor.clone(),
        });
        let reader: Arc<dyn LayoutReader> =
            self.layout.reader(executor.clone(), self.ctx.clone())?;

        // We start with a stream of row masks
        let row_masks = stream::iter(self.row_masks);

        // If we have a filter expression, we set up a filter stream
        let row_masks: BoxStream<'static, VortexResult<RowMask>> =
            if let Some(filter) = self.filter.clone() {
                let reader = reader.clone();

                let pruning = Arc::new(FilterExpr::try_new(
                    reader
                        .dtype()
                        .as_struct()
                        .ok_or_else(|| {
                            vortex_err!("Vortex scan currently only works for struct arrays")
                        })?
                        .clone(),
                    filter.clone(),
                )?);

                row_masks
                    .map(move |row_mask| {
                        let reader = reader.clone();
                        let filter = filter.clone();
                        let pruning = pruning.clone();

                        log::debug!(
                            "Evaluating filter {} for row mask {}..{} {}",
                            &filter,
                            row_mask.begin(),
                            row_mask.end(),
                            row_mask.filter_mask().density()
                        );
                        async move { pruning.new_evaluation(&row_mask).evaluate(reader).await }
                    })
                    // Instead of buffering, we should be smarter where we poll the stream until
                    // the I/O queue has ~256MB of requests in it. Our working set size.
                    // We then return Pending and the I/O thread is free to spawn the requests.
                    .buffered(self.concurrency)
                    .filter_map(|r| async move {
                        r.map(|r| (!r.filter_mask().all_false()).then_some(r))
                            .transpose()
                    })
                    .boxed()
            } else {
                row_masks.map(Ok).boxed()
            };

        // Setup a projection stream
        let reader = reader.clone();
        let projection = self.projection.clone();
        let array_stream = row_masks
            .map(move |row_mask| {
                let reader = reader.clone();
                let projection = projection.clone();
                let executor = executor.clone();
                async move {
                    let row_mask = row_mask?;
                    let mut array = reader.evaluate_expr(row_mask, projection).await?;
                    if self.canonicalize {
                        array = executor.evaluate(&array, &[ScanTask::Canonicalize]).await?;
                    }
                    Ok(array)
                }
            })
            .buffered(self.concurrency);

        let io_stream = self.driver.io_stream();

        let unified = UnifiedDriverStream {
            exec_stream: array_stream,
            io_stream,
        };

        let result_dtype = self.projection.return_dtype(self.layout.dtype())?;
        Ok(ArrayStreamAdapter::new(result_dtype, unified))
    }

    pub async fn into_array(self) -> VortexResult<Array> {
        self.into_array_stream()?.into_array().await
    }
}

/// Compute a mask of field paths referenced by this scan.
///
/// Projection and filter must be pre-simplified.
fn field_mask(
    projection: &ExprRef,
    filter: Option<&ExprRef>,
    scope_dtype: &DType,
) -> VortexResult<Vec<FieldMask>> {
    let Some(struct_dtype) = scope_dtype.as_struct() else {
        return Ok(vec![FieldMask::All]);
    };

    let projection_mask = immediate_scope_access(projection, struct_dtype)?;
    let filter_mask = filter
        .map(|f| immediate_scope_access(f, struct_dtype))
        .transpose()?
        .unwrap_or_default();

    Ok(projection_mask
        .union(&filter_mask)
        .cloned()
        .map(|c| FieldMask::Prefix(FieldPath::from(Field::Name(c))))
        .collect_vec())
}
