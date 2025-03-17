use std::ops::BitAnd;
use std::sync::Arc;

use executor::{TaskExecutor, ThreadsExecutor};
use futures::{FutureExt, Stream, StreamExt, TryFutureExt, stream};
use itertools::Itertools;
pub use split_by::*;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter, ArrayStreamExt};
use vortex_array::{ArrayContext, ArrayRef, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Field, FieldMask, FieldName, FieldPath};
use vortex_error::{VortexError, VortexExpect, VortexResult};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_expr::{ExprRef, Identity};
use vortex_mask::Mask;

use crate::scan::unified::UnifiedDriverStream;
use crate::segments::{AsyncSegmentReader, RowRangePruner, SegmentCollector, SegmentStream};
use crate::{
    ExprEvaluator, Layout, LayoutReader, MaskFuture, RowMask, instrument, mask_future_ready,
    range_intersection,
};

pub mod executor;
pub(crate) mod filter;
mod split_by;
pub mod unified;

pub trait ScanDriver: 'static + Sized {
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
    fn io_stream(self, segments: SegmentStream) -> impl Stream<Item = VortexResult<()>>;
}

/// A struct for building a scan operation.
pub struct ScanBuilder<D: ScanDriver> {
    driver: D,
    task_executor: Option<TaskExecutor>,
    layout: Layout,
    ctx: ArrayContext, // TODO(ngates): store this on larger context on Layout
    projection: ExprRef,
    filter: Option<ExprRef>,
    row_indices: Option<Buffer<u64>>,
    split_by: SplitBy,
    canonicalize: bool,
    // The number of splits to make progress on concurrently.
    concurrency: usize,
    prefetch_conjuncts: bool,
}

impl<D: ScanDriver> ScanBuilder<D> {
    pub fn new(driver: D, layout: Layout, ctx: ArrayContext) -> Self {
        Self {
            driver,
            task_executor: None,
            layout,
            ctx,
            projection: Identity::new_expr(),
            filter: None,
            row_indices: None,
            split_by: SplitBy::Layout,
            canonicalize: false,
            prefetch_conjuncts: false,
            concurrency: 1024,
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

    /// The number of row splits to make progress on concurrently, must be greater than 0.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
        self
    }

    /// The number of row splits to make progress on concurrently, must be greater than 0.
    pub fn with_prefetch_conjuncts(mut self, prefetch: bool) -> Self {
        self.prefetch_conjuncts = prefetch;
        self
    }

    pub fn with_task_executor(mut self, task_executor: TaskExecutor) -> Self {
        self.task_executor = Some(task_executor);
        self
    }

    pub fn build(self) -> VortexResult<Scan<D>> {
        let projection = simplify_typed(self.projection.clone(), self.layout.dtype())?;
        let filter = self
            .filter
            .clone()
            .map(|f| simplify_typed(f, self.layout.dtype()))
            .transpose()?;
        let (filter_mask, projection_mask) =
            filter_and_projection_masks(&projection, filter.as_ref(), self.layout.dtype())?;

        let field_mask: Vec<_> = filter_mask
            .iter()
            .cloned()
            .chain(projection_mask.iter().cloned())
            .collect();

        let splits = self.split_by.splits(&self.layout, &field_mask)?;
        let collector = SegmentCollector::default();
        // FIXME(ngates): do not prefetch!!
        // self.layout
        //     .required_segments(0, &filter_mask, &projection_mask, &mut collector)?;
        let (mut row_range_pruner, segments) = collector.finish();
        let row_indices = self.row_indices.clone();
        if let Some(indices) = &row_indices {
            row_range_pruner.retain_matching(indices.clone());
        }

        let row_masks = splits
            .into_iter()
            .filter_map(move |row_range| {
                let Some(row_indices) = &row_indices else {
                    // If there is no row indices filter, then take the whole range
                    return Some(RowMask::new_valid_between(row_range.start, row_range.end));
                };

                // Otherwise, find the indices that are within the row range.
                let intersection = range_intersection(&row_range, row_indices)?;

                // Construct a row mask for the range.
                let filter_mask = Mask::from_indices(
                    usize::try_from(row_range.end - row_range.start)
                        .vortex_expect("Split ranges are within usize"),
                    row_indices[intersection]
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
                .unwrap_or(TaskExecutor::Threads(ThreadsExecutor::default())),
            layout: self.layout,
            ctx: self.ctx,
            projection,
            filter,
            row_masks,
            canonicalize: self.canonicalize,
            concurrency: self.concurrency,
            prefetch_conjuncts: self.prefetch_conjuncts,
            row_range_pruner,
            segments,
        })
    }

    /// Perform the scan operation and return a stream of arrays.
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + 'static> {
        self.build()?.into_array_stream()
    }

    pub async fn read_all(self) -> VortexResult<ArrayRef> {
        self.into_array_stream()?.read_all().await
    }
}

/// Compute masks of field paths referenced by the projection and filter in the scan.
///
/// Projection and filter must be pre-simplified.
fn filter_and_projection_masks(
    projection: &ExprRef,
    filter: Option<&ExprRef>,
    dtype: &DType,
) -> VortexResult<(Vec<FieldMask>, Vec<FieldMask>)> {
    let Some(struct_dtype) = dtype.as_struct() else {
        return Ok(match filter {
            Some(_) => (vec![FieldMask::All], vec![FieldMask::All]),
            None => (Vec::new(), vec![FieldMask::All]),
        });
    };
    let projection_mask = immediate_scope_access(projection, struct_dtype)?;
    Ok(match filter {
        None => (
            Vec::new(),
            projection_mask.into_iter().map(to_field_mask).collect_vec(),
        ),
        Some(f) => {
            let filter_mask = immediate_scope_access(f, struct_dtype)?;
            let only_projection_mask = projection_mask
                .difference(&filter_mask)
                .cloned()
                .map(to_field_mask)
                .collect_vec();
            (
                filter_mask.into_iter().map(to_field_mask).collect_vec(),
                only_projection_mask,
            )
        }
    })
}

fn to_field_mask(field: FieldName) -> FieldMask {
    FieldMask::Prefix(FieldPath::from(Field::Name(field)))
}

#[allow(dead_code)]
pub struct Scan<D> {
    driver: D,
    task_executor: TaskExecutor,
    layout: Layout,
    ctx: ArrayContext,
    // Guaranteed to be simplified
    projection: ExprRef,
    // Guaranteed to be simplified
    filter: Option<ExprRef>,
    row_masks: Vec<RowMask>,
    canonicalize: bool,
    //TODO(adam): bake this into the executors?
    concurrency: usize,
    prefetch_conjuncts: bool,
    row_range_pruner: RowRangePruner,
    segments: SegmentStream,
}

impl<D: ScanDriver> Scan<D> {
    /// Perform the scan operation and return a stream of arrays.
    ///
    /// The returned stream should be considered to perform I/O-bound operations and requires
    /// frequent polling to make progress.
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + 'static> {
        // Create a single LayoutReader that is reused for the entire scan.
        let segment_reader = self.driver.segment_reader();
        let _task_executor = self.task_executor.clone();
        let reader: Arc<dyn LayoutReader> = self.layout.reader(segment_reader, self.ctx.clone())?;
        //
        // let pruning = self
        //     .filter
        //     .as_ref()
        //     .map(|filter| {
        //         let pruning = Arc::new(FilterExpr::try_new(
        //             reader
        //                 .dtype()
        //                 .as_struct()
        //                 .ok_or_else(|| {
        //                     vortex_err!("Vortex scan currently only works for struct arrays")
        //                 })?
        //                 .clone(),
        //             filter.clone(),
        //             self.prefetch_conjuncts,
        //         )?);
        //
        //         VortexResult::Ok(pruning)
        //     })
        //     .transpose()?;

        // Initialize a stream of mask futures.
        let mut masks = self
            .row_masks
            .iter()
            .map(|row_mask| {
                let row_range = row_mask.begin()..row_mask.end();
                let mask = mask_future_ready(row_mask.filter_mask().clone());

                (row_range, mask)
            })
            .collect_vec();

        // Construct the filter expressions if necessary
        if let Some(filter) = self.filter.as_ref() {
            masks = masks
                .into_iter()
                .map(|(row_range, mask_future)| {
                    // FIXME(ngates): use the FilterEvaluator for conjunct short-circuiting.

                    let range_len = usize::try_from(
                        row_range
                            .end
                            .checked_sub(row_range.start)
                            .vortex_expect("row range overflow"),
                    )
                    .vortex_expect("row range overflow");

                    // NOTE that we currently pass an all-true mask to the filter expression.
                    let mask_future: MaskFuture = reader
                        .evaluate_expr2(
                            &row_range,
                            filter,
                            mask_future_ready(Mask::new_true(range_len)),
                        )?
                        .map_err(VortexError::from)
                        .and_then(async move |array: Option<ArrayRef>| {
                            // The array is a boolean array, so we extract the mask.
                            let filter_result = array
                                .map(|array| Mask::try_from(&array.to_bool()?))
                                .unwrap_or_else(|| Ok(Mask::new_false(range_len)))?;

                            // Intersect the filter result with the original mask.
                            let mask = mask_future.await?;
                            Ok(mask.bitand(&filter_result))
                        })
                        .map_err(Arc::new)
                        .boxed()
                        .shared();
                    Ok::<_, VortexError>((row_range, mask_future))
                })
                .try_collect()?;
        }

        // Project the masks into the final array futures.
        let projection = self.projection.clone();
        let arrays: Vec<_> = masks
            .into_iter()
            .map(move |(row_range, mask)| reader.evaluate_expr2(&row_range, &projection, mask))
            .try_collect()?;

        let exec_stream = stream::iter(arrays)
            .buffered(self.concurrency)
            .filter_map(|v| async move { v.transpose() });

        let exec_stream = instrument!("exec_stream", exec_stream);
        let io_stream = self.driver.io_stream(self.segments);

        let unified = UnifiedDriverStream {
            exec_stream,
            io_stream,
        };

        let result_dtype = self.projection.return_dtype(self.layout.dtype())?;
        Ok(ArrayStreamAdapter::new(result_dtype, unified))
    }

    pub async fn read_all(self) -> VortexResult<ArrayRef> {
        self.into_array_stream()?.read_all().await
    }
}
