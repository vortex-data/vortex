// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;
use std::{cmp, iter};

use futures::future::BoxFuture;
use itertools::Itertools;
pub use multi_scan::*;
pub use selection::*;
pub use split_by::*;
use tasks::{split_exec, TaskContext};
use vortex_array::iter::ArrayIterator;
use vortex_array::stats::StatsSet;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::ArrayRef;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Field, FieldMask, FieldName, FieldPath};
use vortex_error::{vortex_bail, VortexResult};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed;
use vortex_expr::{root, ExprRef};
use vortex_io::runtime::Handle;
use vortex_layout::layouts::row_idx::RowIdxLayoutReader;
use vortex_layout::{LayoutReader, LayoutReaderRef};
use vortex_metrics::VortexMetrics;

use crate::filter::FilterExpr;
use crate::work_queue::{TaskFactory, WorkStealingQueue};
use crate::work_stealing_iter::{ArrayTask, WorkStealingArrayIterator};

mod arrow;
mod filter;
mod multi_scan;
pub mod row_mask;
mod selection;
mod split_by;
mod tasks;
mod work_queue;
mod work_stealing_iter;

/// A struct for building a scan operation.
pub struct ScanBuilder<A> {
    handle: Option<Handle>,
    layout_reader: LayoutReaderRef,
    projection: ExprRef,
    filter: Option<ExprRef>,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    // TODO(joe): replace this is usage of row_id selection, see
    selection: Selection,
    /// How to split the file for concurrent processing.
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
    /// Provide a handle to the runtime on which to spawn tasks.
    pub fn with_handle(mut self, handle: Handle) -> Self {
        self.handle = Some(handle);
        self
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
            handle: self.handle,
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

    pub fn prepare(self) -> VortexResult<RepeatedScan<A>> {
        let dtype = self.dtype()?;

        let Some(handle) = self.handle else {
            vortex_bail!(
                "A runtime handle must be provided to the scan builder using `with_handle`"
            );
        };
        if self.filter.is_some() && self.limit.is_some() {
            vortex_bail!("Vortex doesn't support scans with both a filter and a limit")
        }

        // Spin up the root layout reader, and wrap it in a FilterLayoutReader to perform
        // conjunction splitting if a filter is provided.
        let mut layout_reader = self.layout_reader;

        // Enrich the layout reader to support RowIdx expressions.
        // Note that this is applied below the filter layout reader since it can perform
        // better over individual conjunctions.
        layout_reader = Arc::new(RowIdxLayoutReader::new(self.row_offset, layout_reader));

        // Normalize and simplify the expressions.
        let projection = simplify_typed(self.projection, layout_reader.dtype())?;
        let filter = self
            .filter
            .map(|f| simplify_typed(f, layout_reader.dtype()))
            .transpose()?;

        // Construct field masks and compute the row splits of the scan.
        let (filter_mask, projection_mask) =
            filter_and_projection_masks(&projection, filter.as_ref(), layout_reader.dtype())?;
        let field_mask: Vec<_> = [filter_mask, projection_mask].concat();
        let splits = self.split_by.splits(layout_reader.as_ref(), &field_mask)?;
        Ok(RepeatedScan {
            handle,
            layout_reader,
            projection,
            filter,
            row_range: self.row_range,
            selection: self.selection,
            splits,
            concurrency: self.concurrency,
            map_fn: self.map_fn,
            limit: self.limit,
            dtype,
        })
    }

    /// Constructs a task per row split of the scan, returned as a vector of futures.
    pub fn build(self) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        // The ultimate short circuit
        if self.limit.is_some_and(|l| l == 0) {
            return Ok(vec![]);
        }

        self.prepare()?.execute(None)
    }

    /// Returns a [`Stream`](futures::Stream) with tasks spawned onto the scan's [`Handle`].
    pub fn into_stream(
        self,
    ) -> VortexResult<impl futures::Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        self.prepare()?.execute_stream(None)
    }
}

impl ScanBuilder<ArrayRef> {
    pub fn new(layout_reader: Arc<dyn LayoutReader>) -> Self {
        Self {
            handle: Handle::find(),
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

    /// Returns an [`ArrayStream`] with tasks spawned onto the scan's [`Handle`].
    ///
    /// See [`ScanBuilder::into_stream`] for more details.
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype()?;
        let stream = self.into_stream()?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
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
    let Some(struct_dtype) = dtype.as_struct_fields_opt() else {
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

/// A projected subset (by indices, range, and filter) of rows from a Vortex data source.
///
/// The method of this struct enable, possibly concurrent, scanning of multiple row ranges of this
/// data source.
///
/// See also: [ScanBuilder].
pub struct RepeatedScan<A: 'static + Send> {
    handle: Handle,
    layout_reader: LayoutReaderRef,
    projection: ExprRef,
    filter: Option<ExprRef>,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    selection: Selection,
    /// The natural splits of the file.
    splits: BTreeSet<u64>,
    /// The number of splits to make progress on concurrently **per-thread**.
    concurrency: usize,
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<usize>,
    /// The dtype of the projected arrays.
    dtype: DType,
}

impl<A: 'static + Send> RepeatedScan<A> {
    pub fn execute(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        let ctx = Arc::new(TaskContext {
            selection: self.selection.clone(),
            filter: self.filter.clone().map(|f| Arc::new(FilterExpr::new(f))),
            reader: self.layout_reader.clone(),
            projection: self.projection.clone(),
            mapper: self.map_fn.clone(),
        });

        let row_range = intersect_ranges(self.row_range.as_ref(), row_range);
        let splits_iter: Box<dyn Iterator<Item = _>> = match row_range {
            None => Box::new(self.splits.iter().copied()),
            Some(range) => {
                if range.start > range.end {
                    return Ok(Vec::new());
                }
                Box::new(
                    iter::once(range.start)
                        .chain(self.splits.range(range.clone()).copied())
                        .chain(iter::once(range.end)),
                )
            }
        };

        // Create a task that executes the full scan pipeline for each split.
        let mut limit = self.limit;
        let split_tasks = splits_iter
            .tuple_windows()
            .filter_map(|(start, end)| {
                if limit.is_some_and(|l| l == 0) || start >= end {
                    None
                } else {
                    Some(split_exec(ctx.clone(), start..end, limit.as_mut()))
                }
            })
            .try_collect()?;

        Ok(split_tasks)
    }

    pub fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl futures::Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        use futures::StreamExt;
        // Multiply our per-thread concurrency by ~the number of available threads.
        let concurrency = self.concurrency
            * std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);

        let handle = self.handle.clone();
        Ok(futures::stream::iter(self.execute(row_range)?)
            .map(move |task| handle.spawn(task))
            .buffered(concurrency)
            .filter_map(|chunk| async move { chunk.transpose() }))
    }
}

impl RepeatedScan<ArrayRef> {
    pub fn execute_array_iter(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl ArrayIterator + Send + Clone + 'static> {
        let dtype = self.dtype.clone();
        let tasks = self.execute(row_range)?;
        let queue = WorkStealingQueue::new([Box::new(move || Ok(tasks)) as TaskFactory<ArrayTask>]);

        Ok(WorkStealingArrayIterator::new(
            queue,
            Arc::new(dtype),
            self.concurrency,
        ))
    }

    pub fn execute_array_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
    }
}

fn intersect_ranges(left: Option<&Range<u64>>, right: Option<Range<u64>>) -> Option<Range<u64>> {
    match (left, right) {
        (None, None) => None,
        (None, Some(r)) => Some(r),
        (Some(l), None) => Some(l.clone()),
        (Some(l), Some(r)) => Some(cmp::max(l.start, r.start)..cmp::min(l.end, r.end)),
    }
}

#[cfg(test)]
mod tests;
