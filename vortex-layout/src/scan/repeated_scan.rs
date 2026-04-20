// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::iter;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use itertools::Either;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_error::VortexResult;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;
use crate::scan::limit::limit_array_stream;
use crate::scan::splits::Splits;
use crate::scan::tasks::TaskContext;
use crate::scan::tasks::TaskFuture;
use crate::scan::tasks::split_exec;

/// A projected subset (by indices, range, and filter) of rows from a Vortex data source.
///
/// The method of this struct enable, possibly concurrent, scanning of multiple row ranges of this
/// data source.
pub struct RepeatedScan {
    session: VortexSession,
    layout_reader: LayoutReaderRef,
    projection: Expression,
    filter: Option<Expression>,
    ordered: bool,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    selection: Selection,
    /// The natural splits of the file.
    splits: Splits,
    /// The number of splits to make progress on concurrently **per-thread**.
    concurrency: usize,
    /// Maximal number of rows to read (after filtering)
    limit: Option<u64>,
    /// The dtype of the projected arrays.
    dtype: DType,
}

impl RepeatedScan {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn execute_array_iter<B: BlockingRuntime>(
        &self,
        row_range: Option<Range<u64>>,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        let iter = runtime.block_on_stream(stream);
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
    /// Constructor just to allow `scan_builder` to create a `RepeatedScan`.
    #[expect(
        clippy::too_many_arguments,
        reason = "all arguments are needed for scan construction"
    )]
    pub fn new(
        session: VortexSession,
        layout_reader: LayoutReaderRef,
        projection: Expression,
        filter: Option<Expression>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: Splits,
        concurrency: usize,
        limit: Option<u64>,
        dtype: DType,
    ) -> Self {
        Self {
            session,
            layout_reader,
            projection,
            filter,
            ordered,
            row_range,
            selection,
            splits,
            concurrency,
            limit,
            dtype,
        }
    }

    fn split_ranges(&self, row_range: Option<Range<u64>>) -> Vec<Range<u64>> {
        let row_range = intersect_ranges(self.row_range.as_ref(), row_range);

        match &self.splits {
            Splits::Natural(btree_set) => {
                let splits_iter = match row_range {
                    None => Either::Left(btree_set.iter().copied()),
                    Some(range) => {
                        if range.is_empty() {
                            return Vec::new();
                        }
                        Either::Right(
                            iter::once(range.start)
                                .chain(btree_set.range(range.clone()).copied())
                                .chain(iter::once(range.end)),
                        )
                    }
                };

                splits_iter
                    .tuple_windows()
                    .map(|(start, end)| start..end)
                    .collect()
            }
            Splits::Ranges(ranges) => match row_range {
                None => ranges.to_vec(),
                Some(range) => {
                    if range.is_empty() {
                        return Vec::new();
                    }
                    ranges
                        .iter()
                        .filter_map(move |r| {
                            let start = cmp::max(r.start, range.start);
                            let end = cmp::min(r.end, range.end);
                            (start < end).then_some(start..end)
                        })
                        .collect()
                }
            },
        }
    }

    pub(crate) fn execute(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<TaskFuture<Option<ArrayRef>>>> {
        let ctx = Arc::new(TaskContext {
            selection: self.selection.clone(),
            filter: self.filter.clone().map(|f| Arc::new(FilterExpr::new(f))),
            reader: Arc::clone(&self.layout_reader),
            projection: self.projection.clone(),
        });

        let mut limit = self.limit;
        let mut tasks = Vec::new();

        for range in self.split_ranges(row_range) {
            if range.start >= range.end {
                continue;
            }

            if limit.is_some_and(|l| l == 0) {
                break;
            }

            tasks.push(split_exec(Arc::clone(&ctx), range, limit.as_mut())?);
        }

        Ok(tasks)
    }

    pub(crate) fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl Stream<Item = VortexResult<ArrayRef>> + Send + 'static> {
        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let handle = self.session.handle();

        if self.filter.is_some() && self.limit.is_some() {
            let ctx = Arc::new(TaskContext {
                selection: self.selection.clone(),
                filter: self.filter.clone().map(|f| Arc::new(FilterExpr::new(f))),
                reader: Arc::clone(&self.layout_reader),
                projection: self.projection.clone(),
            });
            let ordered = self.ordered;
            let limit = self.limit;
            let stream = futures::stream::iter(self.split_ranges(row_range)).map(move |range| {
                let ctx = Arc::clone(&ctx);
                let handle = handle.clone();
                async move {
                    let task = split_exec(ctx, range, None)?;
                    handle.spawn(task).await
                }
                .boxed()
            });
            let stream = if ordered {
                stream.buffered(1).boxed()
            } else {
                stream.buffer_unordered(1).boxed()
            };

            return Ok(limit_array_stream(
                stream.filter_map(|chunk| async move { chunk.transpose() }),
                limit,
            ));
        }

        let stream =
            futures::stream::iter(self.execute(row_range)?).map(move |task| handle.spawn(task));
        let concurrency = self.concurrency * num_workers;
        let stream = if self.ordered {
            stream.buffered(concurrency).boxed()
        } else {
            stream.buffer_unordered(concurrency).boxed()
        };

        Ok(limit_array_stream(
            stream.filter_map(|chunk| async move { chunk.transpose() }),
            self.limit,
        ))
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
