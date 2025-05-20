use std::iter;
use std::ops::{Deref, Range};
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
pub use executor::*;
use futures::executor::LocalPool;
use futures::future::ok;
use futures::task::LocalSpawnExt;
use futures::{FutureExt, Stream, StreamExt, stream};
use itertools::Itertools;
pub use selection::*;
pub use split_by::*;
use vortex_array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::{ArrayRef, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Field, FieldMask, FieldName, FieldPath};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_expr::{ExprRef, Identity};
use vortex_metrics::VortexMetrics;

use crate::LayoutReader;
use crate::layouts::filter::FilterLayoutReader;
mod executor;
pub mod row_mask;
mod selection;
mod split_by;

/// A struct for building a scan operation.
pub struct ScanBuilder<A> {
    layout_reader: Arc<dyn LayoutReader>,
    projection: ExprRef,
    filter: Option<ExprRef>,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    selection: Selection,
    /// How to split the file for concurrent processing.
    split_by: SplitBy,
    /// The number of splits to make progress on concurrently.
    concurrency: usize,
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// The executor used to spawn each split task.
    executor: Option<Arc<dyn TaskExecutor>>,
    metrics: VortexMetrics,
}

impl<A: 'static + Send> ScanBuilder<A> {
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

    /// The number of row splits to make progress on concurrently, must be greater than 0.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
        self
    }

    /// Spawn each CPU task onto the given Tokio runtime.
    ///
    /// Note that this is an odd use of the Tokio runtime. Typically, it is used predominantly
    /// for I/O bound tasks.
    #[cfg(feature = "tokio")]
    pub fn with_tokio_executor(mut self, handle: tokio::runtime::Handle) -> Self {
        self.executor = Some(Arc::new(handle));
        self
    }

    pub fn with_executor(mut self, executor: Arc<dyn TaskExecutor>) -> Self {
        self.executor = Some(executor);
        self
    }

    pub fn with_metrics(mut self, metrics: VortexMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Map each split of the scan. The function will be run on the spawned task.
    pub fn map<B: 'static>(
        self,
        map_fn: impl Fn(A) -> VortexResult<B> + 'static + Send + Sync,
    ) -> ScanBuilder<B> {
        let old_map_fn = self.map_fn;
        ScanBuilder {
            layout_reader: self.layout_reader,
            projection: self.projection,
            filter: self.filter,
            row_range: self.row_range,
            selection: self.selection,
            split_by: self.split_by,
            concurrency: self.concurrency,
            map_fn: Arc::new(move |a| map_fn(old_map_fn(a)?)),
            executor: self.executor,
            metrics: self.metrics,
        }
    }

    /// Returns the output [`DType`] of the scan.
    pub fn dtype(&self) -> VortexResult<DType> {
        self.projection.return_dtype(self.layout_reader.dtype())
    }

    /// Constructs a task per row split of the scan, returned as a vector of futures.
    #[allow(clippy::unused_enumerate_index)]
    pub fn build(self) -> VortexResult<Vec<impl Future<Output = VortexResult<Option<A>>>>> {
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
        let splits = self.split_by.splits(layout_reader.deref(), &field_mask)?;

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
            .map(|row_mask| {
                let row_range = row_mask.row_range();
                (row_range, ok(row_mask.mask().clone()).boxed())
            })
            .collect_vec();

        // NOTE(ngates): since segment prefetching occurs in insertion order, we construct
        //  all pruning tasks, then all filter tasks, then all projection tasks. When a task
        //  explicitly polls a segment, it jumps to the front of the queue so this shouldn't
        //  impact the time-to-first-chunk latency.

        // If a filter expression is provided, then we setup pruning and filter evaluations.
        let row_masks = if let Some(filter) = &filter {
            // Map the row masks through the pruning evaluation
            let row_masks: Vec<_> = row_masks
                .into_iter()
                .map(|(row_range, mask_fut)| {
                    let eval = layout_reader.pruning_evaluation(&row_range, filter)?;
                    let mask_fut = async move {
                        let mask = mask_fut.await?;
                        if mask.all_false() {
                            Ok(mask)
                        } else {
                            eval.invoke(mask).await
                        }
                    }
                    .boxed();
                    Ok::<_, VortexError>((row_range, mask_fut))
                })
                .try_collect()?;

            // Map the row masks through the filter evaluation
            row_masks
                .into_iter()
                .map(|(row_range, mask_fut)| {
                    let eval = layout_reader.filter_evaluation(&row_range, filter)?;
                    let mask_fut = async move {
                        let mask = mask_fut.await?;
                        if mask.all_false() {
                            Ok(mask)
                        } else {
                            eval.invoke(mask).await
                        }
                    }
                    .boxed();
                    Ok::<_, VortexError>((row_range, mask_fut))
                })
                .try_collect()?
        } else {
            row_masks
        };

        // Finally, map the row masks through the projection evaluation and spawn.
        row_masks
            .into_iter()
            .map(|(row_range, mask_fut)| {
                let map_fn = self.map_fn.clone();
                let eval = layout_reader.projection_evaluation(&row_range, &projection)?;
                let array_fut = async move {
                    let mask = mask_fut.await?;
                    if mask.all_false() {
                        Ok(None)
                    } else {
                        map_fn(eval.invoke(mask).await?).map(Some)
                    }
                }
                .boxed();

                Ok(match &self.executor {
                    None => array_fut,
                    Some(executor) => executor.spawn(array_fut),
                })
            })
            .try_collect()
    }

    /// Returns a stream over the scan objects.
    pub fn into_stream(self) -> VortexResult<impl Stream<Item = VortexResult<A>> + 'static> {
        let concurrency = self.concurrency;
        Ok(stream::iter(self.build()?)
            .buffered(concurrency)
            .filter_map(|r| async move { r.transpose() }))
    }
}

impl ScanBuilder<ArrayRef> {
    pub fn new(layout_reader: Arc<dyn LayoutReader>) -> Self {
        Self {
            layout_reader,
            projection: Identity::new_expr(),
            filter: None,
            row_range: None,
            selection: Default::default(),
            split_by: SplitBy::Layout,
            // How many row splits to make progress on concurrently (not necessarily in parallel,
            // that is decided by the TaskExecutor).
            concurrency: 16,
            map_fn: Arc::new(Ok),
            executor: None,
            metrics: Default::default(),
        }
    }

    /// Map the scan into a stream of Arrow [`RecordBatch`].
    pub fn map_to_record_batch(self, schema: SchemaRef) -> ScanBuilder<RecordBatch> {
        self.map(move |array| {
            let st = array.to_struct()?;
            st.into_record_batch_with_schema(schema.as_ref())
        })
    }

    /// Returns a stream over the scan with each CPU task polled on the current thread as per
    /// the behaviour of [`futures::stream::Buffered`].
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + 'static> {
        let dtype = self.dtype()?;
        let stream = self.into_stream()?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
    }

    /// Returns a blocking iterator over the scan.
    ///
    /// All work will be performed on the current thread, with tasks interleaved per the
    /// configured concurrency. Any configured executor will be ignored.
    pub fn into_array_iter(self) -> VortexResult<impl ArrayIterator + 'static> {
        let dtype = self.dtype()?;
        let concurrency = self.concurrency;

        let mut local_pool = LocalPool::new();
        let spawner = local_pool.spawner();

        let mut stream = stream::iter(self.build()?)
            .map(move |task| {
                spawner
                    .spawn_local_with_handle(task)
                    .map_err(|e| vortex_err!("Failed to spawn task: {e}"))
                    .vortex_expect("Failed to spawn task")
            })
            .buffered(concurrency)
            .filter_map(|a| async move { a.transpose() })
            .boxed_local();

        Ok(ArrayIteratorAdapter::new(
            dtype,
            iter::from_fn(move || local_pool.run_until(stream.next())),
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
