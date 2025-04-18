use std::iter;
use std::ops::Range;
use std::sync::Arc;

use futures::executor::LocalPool;
use futures::future::BoxFuture;
use futures::task::LocalSpawnExt;
use futures::{FutureExt, StreamExt, stream};
use itertools::Itertools;
pub use selection::*;
pub use split_by::*;
use vortex_array::builders::builder_with_capacity;
use vortex_array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::{Array, ArrayRef};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Field, FieldMask, FieldName, FieldPath};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_expr::{ExprRef, Identity};
use vortex_metrics::{VortexMetrics, instrument};

use crate::layouts::filter::FilterLayoutReader;
use crate::{ExprEvaluator, LayoutReader};

pub mod row_mask;
mod selection;
mod split_by;

/// A struct for building a scan operation.
pub struct ScanBuilder {
    layout_reader: Arc<dyn LayoutReader>,
    projection: ExprRef,
    filter: Option<ExprRef>,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    selection: Selection,
    /// How to split the file for concurrent processing.
    split_by: SplitBy,
    /// Whether the arrays returned by the scan should be in canonical form.
    canonicalize: bool,
    /// The number of splits to make progress on concurrently.
    concurrency: usize,
    metrics: VortexMetrics,
}

impl ScanBuilder {
    pub fn new(layout_reader: Arc<dyn LayoutReader>) -> Self {
        Self {
            layout_reader,
            projection: Identity::new_expr(),
            filter: None,
            row_range: None,
            selection: Default::default(),
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

    pub fn with_row_range(mut self, row_range: Range<u64>) -> Self {
        self.row_range = Some(row_range);
        self
    }

    pub fn with_some_row_range(mut self, row_range: Option<Range<u64>>) -> Self {
        self.row_range = row_range;
        self
    }

    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.selection = Selection::IncludeByIndex(row_indices);
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

    pub fn with_metrics(mut self, metrics: VortexMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    #[allow(clippy::unused_enumerate_index)]
    fn build_tasks(
        self,
    ) -> VortexResult<Vec<impl Future<Output = VortexResult<Option<ArrayRef>>>>> {
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

        let row_masks = splits
            .into_iter()
            .filter_map(|row_range| {
                if let Some(scan_range) = &self.row_range {
                    // If the row range is fully within the scan range, return it.
                    if row_range.start >= scan_range.end || row_range.end < scan_range.start {
                        return None;
                    }
                    // Otherwise, take the intersection of the range.
                    return Some(
                        row_range.start.max(scan_range.start)..row_range.end.min(scan_range.end),
                    );
                } else {
                    Some(row_range)
                }
            })
            .map(|row_range| self.selection.row_mask(&row_range))
            .filter(|mask| !mask.mask().all_false())
            .collect_vec();

        // Create a future to process each row split of the scan.
        row_masks
            .into_iter()
            .enumerate()
            .map(|(_i, row_mask)| {
                let row_range = row_mask.row_range();

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
                    let mut mask = row_mask.mask().clone();
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
            .try_collect()
    }

    /// Returns a stream over the scan with each CPU task spawned using the given spawn function.
    pub fn spawn_on<F, S>(self, mut spawner: S) -> VortexResult<impl ArrayStream + 'static>
    where
        F: Future<Output = VortexResult<Option<ArrayRef>>>,
        S: FnMut(BoxFuture<'static, VortexResult<Option<ArrayRef>>>) -> F + 'static,
    {
        let concurrency = self.concurrency;
        let dtype = self.projection.return_dtype(self.layout_reader.dtype())?;
        let tasks = self.build_tasks()?;

        let array_stream = stream::iter(tasks)
            .map(move |task| spawner(task.boxed()))
            .buffered(concurrency)
            .filter_map(|v| async move { v.transpose() });

        Ok(ArrayStreamAdapter::new(
            dtype,
            instrument!("array_stream", array_stream),
        ))
    }

    /// Returns a stream over the scan with each CPU task spawned onto the given Tokio runtime
    /// using [`tokio::runtime::Handle::spawn`].
    ///
    /// Note that this should only be used if the Tokio runtime is dedicated to CPU-bound tasks.
    #[cfg(feature = "tokio")]
    pub fn spawn_tokio(
        self,
        handle: tokio::runtime::Handle,
    ) -> VortexResult<impl ArrayStream + 'static> {
        self.spawn_on(move |task| {
            let handle = handle.clone();
            async move {
                handle
                    .spawn(task)
                    .await
                    .vortex_expect("Failed to join task")
            }
        })
    }

    /// Returns a stream over the scan with each CPU task spawned onto a Tokio worker thread
    /// using [`tokio::runtime::Handle::spawn_blocking`].
    #[cfg(feature = "tokio")]
    pub fn spawn_tokio_blocking(
        self,
        handle: tokio::runtime::Handle,
    ) -> VortexResult<impl ArrayStream + 'static> {
        self.spawn_on(move |task| {
            let handle = handle.clone();
            async move {
                handle
                    .spawn_blocking(|| futures::executor::block_on(task))
                    .await
                    .vortex_expect("Failed to join task")
            }
        })
    }

    /// Returns a stream over the scan with each CPU task polled on the current thread as per
    /// the behaviour of [`futures::stream::Buffered`].
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + 'static> {
        self.spawn_on(|task| task)
    }

    /// Returns a blocking iterator over the scan.
    ///
    /// All work will be performed on the current thread, with tasks interleaved per the
    /// configured concurrency.
    pub fn into_array_iter(self) -> VortexResult<impl ArrayIterator + 'static> {
        let mut local_pool = LocalPool::new();
        let spawner = local_pool.spawner();
        let array_stream = self.spawn_on(move |task| {
            spawner
                .spawn_local_with_handle(task)
                .map_err(|e| vortex_err!("Failed to spawn task: {e}"))
                .vortex_expect("Failed to spawn task")
        })?;

        let mut array_stream = Box::pin(array_stream);
        Ok(ArrayIteratorAdapter::new(
            array_stream.dtype().clone(),
            iter::from_fn(move || local_pool.run_until(array_stream.next())),
        ))
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
