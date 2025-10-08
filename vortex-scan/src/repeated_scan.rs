// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;
use std::{cmp, iter};

use futures::Stream;
use futures::future::BoxFuture;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_io::runtime::{BlockingRuntime, Handle};
use vortex_layout::LayoutReaderRef;

use crate::filter::FilterExpr;
use crate::selection::Selection;
use crate::tasks::{TaskContext, split_exec};

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
    ordered: bool,
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

impl RepeatedScan<ArrayRef> {
    pub fn execute_array_iter<B: BlockingRuntime>(
        &self,
        row_range: Option<Range<u64>>,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        let iter = runtime.block_on_stream(move |_h| stream);
        Ok(ArrayIteratorAdapter::new(dtype, iter))
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

impl<A: 'static + Send> RepeatedScan<A> {
    /// Constructor just to allow `scan_builder` to create a `RepeatedScan`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        handle: Handle,
        layout_reader: LayoutReaderRef,
        projection: ExprRef,
        filter: Option<ExprRef>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: BTreeSet<u64>,
        concurrency: usize,
        map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
        limit: Option<usize>,
        dtype: DType,
    ) -> Self {
        Self {
            handle,
            layout_reader,
            projection,
            filter,
            ordered,
            row_range,
            selection,
            splits,
            concurrency,
            map_fn,
            limit,
            dtype,
        }
    }

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

        // Create a task that executes the full scan operator for each split.
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
    ) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        use futures::StreamExt;
        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let concurrency = self.concurrency * num_workers;
        let handle = self.handle.clone();

        let stream =
            futures::stream::iter(self.execute(row_range)?).map(move |task| handle.spawn(task));

        let stream = if self.ordered {
            stream.buffered(concurrency).boxed()
        } else {
            stream.buffer_unordered(concurrency).boxed()
        };

        Ok(stream.filter_map(|chunk| async move { chunk.transpose() }))
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
