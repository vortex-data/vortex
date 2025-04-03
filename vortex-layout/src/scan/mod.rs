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
use vortex_mask::Mask;
use vortex_metrics::{VortexMetrics, instrument};

use crate::layouts::filter::FilterLayoutReader;
use crate::scan::executor::Executor;
use crate::{ExprEvaluator, LayoutReader};

pub mod executor;
pub mod row_mask;
pub mod split_by;

/// A struct for building a scan operation.
pub struct ScanBuilder {
    layout_reader: Arc<dyn LayoutReader>,
    task_executor: Option<TaskExecutor>,
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
    pub fn new(layout_reader: Arc<dyn LayoutReader>) -> Self {
        Self {
            layout_reader,
            task_executor: None,
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

    #[allow(clippy::unused_enumerate_index)]
    pub fn build(self) -> VortexResult<impl ArrayStream + 'static> {
        // Spin up the root layout reader, and wrap it in a FilterLayoutReader to perform
        // conjunction splitting if a filter is provided.
        let mut layout_reader = self.layout_reader;
        if self.filter.is_some() {
            layout_reader = Arc::new(FilterLayoutReader::new(layout_reader));
        }

        // Normalize and simplify the expressions.
        let projection = simplify_typed(self.projection.clone(), layout_reader.dtype())?;
        let filter = self
            .filter
            .clone()
            .map(|f| simplify_typed(f, layout_reader.dtype()))
            .transpose()?;

        // Construct field masks and compute the row splits of the scan.
        let (filter_mask, projection_mask) =
            filter_and_projection_masks(&projection, filter.as_ref(), layout_reader.dtype())?;
        let field_mask: Vec<_> = filter_mask
            .iter()
            .cloned()
            .chain(projection_mask.iter().cloned())
            .collect();
        let splits = self.split_by.splits(layout_reader.layout(), &field_mask)?;
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

        let result_dtype = projection.return_dtype(layout_reader.dtype())?;

        // Create a future to process each row split of the scan.
        let array_futures: Vec<_> = row_masks
            .into_iter()
            .enumerate()
            .map(|(_i, row_mask)| {
                let row_range = row_mask.begin()..row_mask.end();

                let approx_filter_eval = filter
                    .as_ref()
                    .map(|expr| layout_reader.pruning_evaluation(&row_range, expr))
                    .transpose()?;
                let exact_filter_eval = filter
                    .as_ref()
                    .map(|expr| layout_reader.filter_evaluation(&row_range, expr))
                    .transpose()?;
                let project_eval = layout_reader.projection_evaluation(&row_range, &projection)?;

                Ok::<_, VortexError>(instrument!("split", [split = _i], async move {
                    let mut mask = row_mask.filter_mask().clone();
                    if mask.all_false() {
                        return Ok(None);
                    }

                    if let Some(approx_filter_eval) = approx_filter_eval {
                        // First, we run an approximate evaluation to prune the row range.
                        log::debug!("Pruning row range {:?}", row_range);
                        mask = approx_filter_eval.invoke(mask).await?;
                        if mask.all_false() {
                            return Ok(None);
                        }
                    }

                    if let Some(exact_filter_eval) = exact_filter_eval {
                        // Then, we run the full evaluation.
                        log::debug!("Filtering row range {:?}", row_range);
                        mask = exact_filter_eval.invoke(mask).await?;
                        if mask.all_false() {
                            return Ok(None);
                        }
                    }

                    log::debug!("Projecting row range {:?}", row_range);
                    let mut array = project_eval.invoke(mask).await?;
                    if self.canonicalize {
                        log::debug!("Canonicalizing row range {:?}", row_range);
                        let mut builder = builder_with_capacity(array.dtype(), array.len());
                        array.append_to_builder(builder.as_mut())?;
                        array = builder.finish();
                    }

                    Ok(Some(array))
                }))
            })
            .try_collect()?;

        // Spawn the array futures onto the executor and buffer some number of row splits.
        let task_executor = self
            .task_executor
            .unwrap_or_else(|| TaskExecutor::Threads(ThreadsExecutor::default()));
        let array_stream = stream::iter(array_futures)
            .map(move |task| task_executor.spawn(task))
            .buffered(self.concurrency)
            .filter_map(|v| async move { v.transpose() });

        Ok(ArrayStreamAdapter::new(
            result_dtype,
            instrument!("array_stream", array_stream),
        ))
    }

    /// Read all the data from the scan into a single (likely chunked) Vortex array.
    pub async fn read_all(self) -> VortexResult<ArrayRef> {
        self.build()?.read_all().await
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
