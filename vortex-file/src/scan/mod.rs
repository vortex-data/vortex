use std::sync::Arc;

use executor::{TaskExecutor, ThreadsExecutor};
use futures::{StreamExt, stream};
use itertools::Itertools;
pub use row_mask::*;
pub use split_by::*;
use vortex_array::builders::builder_with_capacity;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter, ArrayStreamExt};
use vortex_array::{Array, ArrayRef};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Field, FieldMask, FieldName, FieldPath};
use vortex_error::{VortexError, VortexExpect, VortexResult};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_expr::{ExprRef, Identity};
use vortex_layout::layouts::filter::FilterLayoutReader;
use vortex_layout::segments::SegmentReader;
use vortex_layout::{ExprEvaluator, LayoutReader};
use vortex_mask::Mask;
use vortex_metrics::{VortexMetrics, instrument};

use crate::scan::executor::Executor;
pub mod executor;
mod row_mask;
mod split_by;
pub mod unified;

/// A struct for building a scan operation.
pub struct ScanBuilder {
    task_executor: Option<TaskExecutor>,
    layout_reader: Arc<dyn LayoutReader>,
    project_segment_reader: Arc<dyn SegmentReader>,
    approx_filter_segment_reader: Option<Arc<dyn SegmentReader>>,
    exact_filter_segment_reader: Option<Arc<dyn SegmentReader>>,
    projection: ExprRef,
    filter: Option<ExprRef>,
    row_indices: Option<Buffer<u64>>,
    split_by: SplitBy,
    canonicalize: bool,
    // The number of splits to make progress on concurrently.
    concurrency: usize,
    metrics: VortexMetrics,
}

impl ScanBuilder {
    pub fn new(
        layout_reader: Arc<dyn LayoutReader>,
        segment_reader: Arc<dyn SegmentReader>,
    ) -> Self {
        Self {
            task_executor: None,
            layout_reader,
            project_segment_reader: segment_reader,
            approx_filter_segment_reader: None,
            exact_filter_segment_reader: None,
            projection: Identity::new_expr(),
            filter: None,
            row_indices: None,
            split_by: SplitBy::Layout,
            canonicalize: false,
            // How many row splits to make progress on concurrently (not necessarily in parallel,
            // that is decided by the TaskExecutor).
            concurrency: 16,
            metrics: Default::default(),
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

    pub fn with_task_executor(mut self, task_executor: TaskExecutor) -> Self {
        self.task_executor = Some(task_executor);
        self
    }

    pub fn with_metrics(mut self, metrics: VortexMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn build(self) -> VortexResult<Scan> {
        let projection = simplify_typed(self.projection.clone(), self.layout_reader.dtype())?;
        let filter = self
            .filter
            .clone()
            .map(|f| simplify_typed(f, self.layout_reader.dtype()))
            .transpose()?;
        let (filter_mask, projection_mask) =
            filter_and_projection_masks(&projection, filter.as_ref(), self.layout_reader.dtype())?;

        let field_mask: Vec<_> = filter_mask
            .iter()
            .cloned()
            .chain(projection_mask.iter().cloned())
            .collect();

        let splits = self
            .split_by
            .splits(self.layout_reader.layout(), &field_mask)?;
        let row_indices = self.row_indices.clone();
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
            task_executor: self
                .task_executor
                .unwrap_or(TaskExecutor::Threads(ThreadsExecutor::default())),
            layout_reader: self.layout_reader,
            project_segment_reader: self.project_segment_reader.clone(),
            approx_filter_segment_reader: self
                .approx_filter_segment_reader
                .unwrap_or_else(|| self.project_segment_reader.clone()),
            exact_filter_segment_reader: self
                .exact_filter_segment_reader
                .unwrap_or_else(|| self.project_segment_reader.clone()),
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

pub struct Scan {
    task_executor: TaskExecutor,
    layout_reader: Arc<dyn LayoutReader>,

    // We allow the caller to configure segment readers for each stage of the scan, in case
    // they wish to prioritise or fetch the segments differently.
    project_segment_reader: Arc<dyn SegmentReader>,
    approx_filter_segment_reader: Arc<dyn SegmentReader>,
    exact_filter_segment_reader: Arc<dyn SegmentReader>,

    // Guaranteed to be simplified
    projection: ExprRef,
    // Guaranteed to be simplified
    filter: Option<ExprRef>,
    row_masks: Vec<RowMask>,
    canonicalize: bool,
    concurrency: usize,
}

impl Scan {
    /// Perform the scan operation and return a stream of arrays.
    ///
    /// The returned stream should be considered to perform I/O-bound operations and requires
    /// frequent polling to make progress.
    #[allow(clippy::unused_enumerate_index)]
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + 'static> {
        // Create a single LayoutReader that is reused for the entire scan.
        let result_dtype = self.projection.return_dtype(self.layout_reader.dtype())?;

        // If there's a filter expression, set up a shared FilterLayoutReader to store stats.
        let layout_reader = FilterLayoutReader::new(self.layout_reader.clone());
        //let filter_reader =
        //    FilterLayoutReader::new(layout_reader.clone(), self.task_executor.clone());

        let array_futures: Vec<_> = self
            .row_masks
            .into_iter()
            .enumerate()
            .map(|(_i, row_mask)| {
                let row_range = row_mask.begin()..row_mask.end();

                let approx_filter_eval = self
                    .filter
                    .as_ref()
                    .map(|expr| {
                        layout_reader.filter_evaluation(
                            &row_range,
                            expr,
                            self.approx_filter_segment_reader.as_ref(),
                        )
                    })
                    .transpose()?;
                let exact_filter_eval = self
                    .filter
                    .as_ref()
                    .map(|expr| {
                        layout_reader.filter_evaluation(
                            &row_range,
                            expr,
                            self.exact_filter_segment_reader.as_ref(),
                        )
                    })
                    .transpose()?;
                let project_eval = layout_reader.projection_evaluation(
                    &row_range,
                    &self.projection,
                    self.project_segment_reader.as_ref(),
                )?;

                Ok::<_, VortexError>(instrument!("split", { split = _i }, async move {
                    let mut mask = row_mask.filter_mask().clone();
                    if mask.all_false() {
                        return Ok(None);
                    }

                    if let Some(approx_filter_eval) = approx_filter_eval {
                        // First, we run an approximate evaluation to prune the row range.
                        mask = approx_filter_eval.invoke_approx(mask).await?;
                        if mask.all_false() {
                            return Ok(None);
                        }
                    }

                    if let Some(exact_filter_eval) = exact_filter_eval {
                        // Then, we run the full evaluation.
                        mask = exact_filter_eval.invoke(mask).await?;
                        if mask.all_false() {
                            return Ok(None);
                        }
                    }

                    let mut array = project_eval.invoke(mask).await?;
                    if self.canonicalize {
                        let mut builder = builder_with_capacity(array.dtype(), array.len());
                        array.append_to_builder(builder.as_mut())?;
                        array = builder.finish();
                    }

                    Ok(Some(array))
                }))
            })
            .try_collect()?;

        let task_executor = self.task_executor.clone();
        let exec_stream = stream::iter(array_futures)
            .map(move |task| task_executor.spawn(task))
            .buffered(self.concurrency)
            .filter_map(|v| async move { v.transpose() });

        Ok(ArrayStreamAdapter::new(
            result_dtype,
            instrument!("exec_stream", exec_stream),
        ))
    }

    pub async fn read_all(self) -> VortexResult<ArrayRef> {
        self.into_array_stream()?.read_all().await
    }
}
