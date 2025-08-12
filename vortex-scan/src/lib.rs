// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use itertools::Itertools;
pub use multi_scan::*;
pub use selection::*;
pub use split_by::*;
use tasks::{TaskContext, split_exec};
use vortex_array::ArrayRef;
use vortex_array::iter::ArrayIterator;
use vortex_array::stats::StatsSet;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Field, FieldMask, FieldName, FieldPath};
use vortex_error::{VortexResult, vortex_bail};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_expr::{ExprRef, root};
use vortex_layout::layouts::row_idx::RowIdxLayoutReader;
use vortex_layout::{LayoutReader, LayoutReaderRef};
pub use vortex_layout::{TaskExecutor, TaskExecutorExt};
use vortex_metrics::VortexMetrics;

use crate::filter::FilterExpr;
use crate::work_queue::{TaskFactory, WorkStealingQueue};
use crate::work_stealing_iter::{ArrayTask, WorkStealingArrayIterator};

mod arrow;
mod filter;
mod multi_scan;
#[cfg(feature = "tokio")]
mod multi_thread;
pub mod row_mask;
mod selection;
mod split_by;
mod tasks;
mod work_queue;
mod work_stealing_iter;

/// A struct for building a scan operation.
pub struct ScanBuilder<A> {
    layout_reader: LayoutReaderRef,
    projection: ExprRef,
    filter: Option<ExprRef>,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    // TODO(joe): replace this is usage of row_id selection, see
    selection: Selection,
    /// How to split the file f§    or concurrent processing.
    split_by: SplitBy,
    /// The number of splits to make progress on concurrently **per-thread**.
    concurrency: usize,
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    metrics: VortexMetrics,
    /// Should we try to prune the file (using stats) on open.
    file_stats: Option<Arc<[StatsSet]>>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<usize>,
    /// The row-offset assigned to the first row of the file. Used by the `row_idx` expression,
    /// but not by the scan [`Selection`] which remains relative.
    row_offset: u64,
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

    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.selection = Selection::IncludeByIndex(row_indices);
        self
    }

    pub fn with_row_offset(mut self, row_offset: u64) -> Self {
        self.row_offset = row_offset;
        self
    }

    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    /// The number of row splits to make progress on concurrently per-thread, must
    /// be greater than 0.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
        self
    }

    pub fn with_metrics(mut self, metrics: VortexMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// The [`DType`] returned by the scan, after applying the projection.
    pub fn dtype(&self) -> VortexResult<DType> {
        self.projection.return_dtype(self.layout_reader.dtype())
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
            metrics: self.metrics,
            file_stats: self.file_stats,
            limit: self.limit,
            row_offset: self.row_offset,
        }
    }

    /// Constructs a task per row split of the scan, returned as a vector of futures.
    pub fn build(mut self) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        if self.filter.is_some() && self.limit.is_some() {
            vortex_bail!("Vortex doesn't support scans with both a filter and a limit")
        }

        // The ultimate short circuit
        if self.limit.is_some_and(|l| l == 0) {
            return Ok(vec![]);
        }

        // Spin up the root layout reader, and wrap it in a FilterLayoutReader to perform
        // conjunction splitting if a filter is provided.
        let mut layout_reader = self.layout_reader;

        // Enrich the layout reader to support RowIdx expressions.
        // Note that this is applied below the filter layout reader since it can perform
        // better over individual conjunctions.
        layout_reader = Arc::new(RowIdxLayoutReader::new(self.row_offset, layout_reader));

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
        let field_mask: Vec<_> = [filter_mask, projection_mask].concat();
        let splits = self.split_by.splits(layout_reader.as_ref(), &field_mask)?;

        let ctx = Arc::new(TaskContext {
            row_range: self.row_range,
            selection: self.selection,
            filter: filter.map(|f| Arc::new(FilterExpr::new(f))),
            reader: layout_reader,
            projection,
            mapper: self.map_fn,
        });

        // Create a task that executes the full scan pipeline for each split.
        let split_tasks = splits
            .into_iter()
            .filter_map(|split_range| {
                if self.limit.is_some_and(|l| l == 0) {
                    None
                } else {
                    Some(split_exec(ctx.clone(), split_range, self.limit.as_mut()))
                }
            })
            .try_collect()?;

        Ok(split_tasks)
    }

    /// Returns a [`Stream`] with tasks spawned onto the current Tokio runtime.
    ///
    /// The stream performs CPU work on the polling thread, with I/O operations dispatched as
    /// per the Vortex I/O traits.
    ///
    /// Task concurrency is the product of the `concurrency` parameter and the number of worker
    /// threads in the Tokio runtime.
    #[cfg(feature = "tokio")]
    pub fn into_tokio_stream(
        self,
    ) -> VortexResult<impl futures::Stream<Item = VortexResult<A>> + Send + 'static> {
        use futures::StreamExt;
        use vortex_error::vortex_err;

        let handle = tokio::runtime::Handle::current();
        let num_workers = handle.metrics().num_workers();
        let concurrency = self.concurrency * num_workers;
        Ok(futures::stream::iter(self.build()?)
            .map(move |task| handle.spawn(task))
            .buffered(concurrency)
            .map(|task| {
                task.map_err(|e| vortex_err!("Failed to join task: {e}"))
                    .flatten()
            })
            .filter_map(|chunk| async move { chunk.transpose() }))
    }
}

impl ScanBuilder<ArrayRef> {
    pub fn new(layout_reader: Arc<dyn LayoutReader>) -> Self {
        Self {
            layout_reader,
            projection: root(),
            filter: None,
            row_range: None,
            selection: Default::default(),
            split_by: SplitBy::Layout,
            // We default to four tasks per worker thread, which allows for some I/O lookahead
            // without too much impact on work-stealing.
            concurrency: 4,
            map_fn: Arc::new(Ok),
            metrics: Default::default(),
            file_stats: None,
            limit: None,
            row_offset: 0,
        }
    }

    /// Returns a thread-safe [`ArrayIterator`] that can be cloned and passed
    /// to other threads to make progress on the same scan concurrently.
    ///
    /// Within each thread, the array chunks will be emitted in the original order they are within
    /// the scan. Between threads, the order is not guaranteed.
    pub fn into_array_iter(self) -> VortexResult<impl ArrayIterator + Send + Clone + 'static> {
        let dtype = self.dtype()?;
        let concurrency = self.concurrency;
        let tasks = self.build()?;
        let queue = WorkStealingQueue::new([Box::new(move || Ok(tasks)) as TaskFactory<ArrayTask>]);

        Ok(WorkStealingArrayIterator::new(
            queue,
            Arc::new(dtype),
            concurrency,
        ))
    }

    /// Returns an [`ArrayStream`] with tasks spawned onto the current Tokio runtime.
    ///
    /// See [`ScanBuilder::into_tokio_stream`] for more details.
    #[cfg(feature = "tokio")]
    pub fn into_tokio_array_stream(
        self,
    ) -> VortexResult<impl vortex_array::stream::ArrayStream + Send + 'static> {
        let dtype = self.dtype()?;
        let stream = self.into_tokio_stream()?;
        Ok(vortex_array::stream::ArrayStreamAdapter::new(dtype, stream))
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
    let projection_mask = immediate_scope_access(projection, struct_dtype);
    Ok(match filter {
        None => (
            Vec::new(),
            projection_mask.into_iter().map(to_field_mask).collect_vec(),
        ),
        Some(f) => {
            let filter_mask = immediate_scope_access(f, struct_dtype);
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
