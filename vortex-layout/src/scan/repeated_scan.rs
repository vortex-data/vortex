// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::iter;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use futures::Stream;
use futures::StreamExt;
use futures::future;
use futures::stream::BoxStream;
use itertools::Either;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;
use crate::scan::limit::RowLimit;
use crate::scan::splits::Splits;
use crate::scan::tasks::TaskContext;
use crate::scan::tasks::TaskFuture;
use crate::scan::tasks::TaskResult;
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
    /// Maximal number of rows to read (after filtering).
    ///
    /// When no shared [`RowLimit`] (`row_limit`) is set, this limit is applied independently to
    /// each `execute_*` call: a `RepeatedScan` executed over several row ranges may therefore
    /// return up to `limit` rows *per call*. Supply a shared `row_limit` to cap the total across
    /// calls and sibling partitions.
    limit: Option<u64>,
    /// An optional row limit shared with sibling external partitions.
    row_limit: Option<RowLimit>,
    /// The dtype of the projected arrays.
    dtype: DType,
}

/// A source of split tasks that has not yet applied task concurrency or output error handling.
#[must_use = "task streams must be scheduled"]
struct TaskStream {
    inner: BoxStream<'static, Task<TaskResult>>,
}

impl TaskStream {
    fn new(stream: impl Stream<Item = Task<TaskResult>> + Send + 'static) -> Self {
        Self {
            inner: stream.boxed(),
        }
    }

    fn eager(handle: Handle, tasks: Vec<TaskFuture>) -> Self {
        Self::new(futures::stream::iter(tasks).map(move |task| handle.spawn(task)))
    }

    /// Build split tasks lazily, stopping as soon as `gate` has no budget left.
    ///
    /// `reserve` is the limit that each split reserves its filtered rows against before
    /// projecting; pass `None` to leave the limit to the caller (see
    /// [`ScheduledTaskStream::with_row_limit`]).
    fn lazy(
        handle: Handle,
        split_ranges: Vec<Range<u64>>,
        selection: Selection,
        ctx: Arc<TaskContext>,
        gate: RowLimit,
        reserve: Option<RowLimit>,
    ) -> Self {
        Self::new(
            futures::stream::iter(split_ranges)
                .take_while(move |_| future::ready(!gate.is_exhausted()))
                .filter_map(move |range| {
                    // Build the row mask and split task synchronously so the I/O system sees the
                    // split's ranges as soon as the buffered stream pulls it, without cloning
                    // `selection`.
                    let row_mask = selection.row_mask(&range);
                    let task = (!row_mask.mask().all_false()).then(|| {
                        // A synchronous split-construction failure happens before any row is
                        // reserved, so it is recoverable (yielded as a stream error) rather than
                        // terminal (which would abort the whole scan). See the `TaskResult` docs.
                        split_exec(Arc::clone(&ctx), row_mask, reserve.clone())
                            .unwrap_or_else(TaskFuture::recoverable)
                    });
                    future::ready(task.map(|task| handle.spawn(task)))
                }),
        )
    }
}

impl Stream for TaskStream {
    type Item = Task<TaskResult>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// A buffered task stream that exposes arrays and applies the task error policy.
pub(crate) struct ScheduledTaskStream {
    tasks: Option<BoxStream<'static, TaskResult>>,
    /// A limit applied to the emitted arrays, for scans that could not reserve rows per split.
    row_limit: Option<RowLimit>,
}

impl ScheduledTaskStream {
    fn new(tasks: TaskStream, ordered: bool, concurrency: usize) -> Self {
        let tasks = if ordered {
            tasks.buffered(concurrency).boxed()
        } else {
            tasks.buffer_unordered(concurrency).boxed()
        };
        Self {
            tasks: Some(tasks),
            row_limit: None,
        }
    }

    /// Trim emitted arrays to `row_limit`, ending the stream once its budget is spent.
    ///
    /// Used by ordered limited scans, where reserving per split would hand the budget to whichever
    /// split filters first rather than to the earliest split. Because this stream yields in split
    /// order, taking rows as they are emitted keeps the limit exact.
    fn with_row_limit(mut self, row_limit: RowLimit) -> Self {
        self.row_limit = Some(row_limit);
        self
    }

    /// Apply [`Self::row_limit`] to an emitted array, returning `None` once the budget is spent.
    fn take_rows(&mut self, array: ArrayRef) -> Option<VortexResult<ArrayRef>> {
        let Some(row_limit) = &self.row_limit else {
            return Some(Ok(array));
        };

        let granted = row_limit.take(array.len());
        let trimmed = granted < array.len();
        if row_limit.is_exhausted() {
            // No later split can contribute now, so drop the queued and in-flight task handles.
            // Their `Drop` implementations abort the work started for them.
            self.tasks = None;
        }

        match granted {
            0 => None,
            _ if trimmed => Some(array.slice(0..granted)),
            _ => Some(Ok(array)),
        }
    }
}

impl Stream for ScheduledTaskStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let Some(tasks) = self.tasks.as_mut() else {
                return Poll::Ready(None);
            };
            let result = ready!(tasks.as_mut().poll_next(cx));
            let Some(result) = result else {
                self.tasks = None;
                return Poll::Ready(None);
            };
            match result {
                TaskResult::Array(Some(array)) => {
                    if let Some(array) = self.take_rows(array) {
                        return Poll::Ready(Some(array));
                    }
                    return Poll::Ready(None);
                }
                TaskResult::Array(None) => {}
                TaskResult::Recoverable(error) => return Poll::Ready(Some(Err(error))),
                TaskResult::Terminal(error) => {
                    // Drop queued and in-flight task handles before yielding the error. Their
                    // `Drop` implementations abort work that can no longer contribute to this
                    // limited scan.
                    self.tasks = None;
                    return Poll::Ready(Some(Err(error)));
                }
            }
        }
    }
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
    pub(crate) fn new(
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
        row_limit: Option<RowLimit>,
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
            row_limit,
            dtype,
        }
    }

    fn has_filter(&self) -> bool {
        self.filter.is_some()
    }

    fn task_context(&self) -> Arc<TaskContext> {
        Arc::new(TaskContext {
            filter: self
                .filter
                .clone()
                .map(|filter| Arc::new(FilterExpr::new(filter))),
            reader: Arc::clone(&self.layout_reader),
            projection: self.projection.clone(),
        })
    }

    fn split_ranges(&self, row_range: Option<Range<u64>>) -> Vec<Range<u64>> {
        let selection_range: Option<Range<u64>> = match &self.selection {
            Selection::IncludeByIndex(buf) if !buf.is_empty() => {
                Some(buf[0]..buf[buf.len() - 1] + 1)
            }
            Selection::IncludeRoaring(roaring) if !roaring.is_empty() => {
                Some(roaring.min().vortex_expect("empty")..roaring.max().vortex_expect("empty") + 1)
            }
            _ => None,
        };
        let row_range = intersect_ranges(self.row_range.as_ref(), row_range);
        let row_range = intersect_ranges(row_range.as_ref(), selection_range);

        match &self.splits {
            Splits::Natural(vec) => {
                debug_assert!(vec.is_sorted());
                let splits_iter = match row_range {
                    None => Either::Left(vec.iter().copied()),
                    Some(range) => {
                        if range.is_empty() {
                            return Vec::new();
                        }
                        let lo = vec.partition_point(|&x| x <= range.start);
                        let hi = vec.partition_point(|&x| x < range.end);
                        Either::Right(
                            iter::once(range.start)
                                .chain(vec[lo..hi].iter().copied())
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
        row_limit: Option<RowLimit>,
    ) -> VortexResult<Vec<TaskFuture>> {
        let mut tasks = Vec::new();
        let ctx = self.task_context();

        for range in self.split_ranges(row_range) {
            if range.start >= range.end {
                continue;
            }
            if row_limit.as_ref().is_some_and(RowLimit::is_exhausted) {
                break;
            }

            let row_mask = self.selection.row_mask(&range);
            if row_mask.mask().all_false() {
                continue;
            }

            tasks.push(split_exec(Arc::clone(&ctx), row_mask, row_limit.clone())?);
        }

        Ok(tasks)
    }

    pub(crate) fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<ScheduledTaskStream> {
        let num_workers = get_available_parallelism().unwrap_or(1);
        let row_limit = self
            .row_limit
            .clone()
            .or_else(|| self.limit.map(RowLimit::new));
        let concurrency = self.concurrency * num_workers;
        let handle = self.session.handle();

        // With both a filter and a limit we cannot know each split's output row count ahead of
        // time, so split tasks are built lazily as the stream is polled. Once earlier work drains
        // the budget, `take_while` prevents further task creation.
        if self.has_filter()
            && let Some(row_limit) = row_limit
        {
            // Reserving rows per split hands the budget to whichever split finishes filtering
            // first, which an ordered scan cannot accept: it must return the *earliest* matching
            // rows. Ordered scans therefore filter and project without reserving, and take rows
            // from the (in-order) output instead. Unordered scans reserve before projecting, so
            // rows they cannot return are never decoded.
            let reserve = (!self.ordered).then(|| row_limit.clone());
            let tasks = TaskStream::lazy(
                handle,
                self.split_ranges(row_range),
                self.selection.clone(),
                self.task_context(),
                row_limit.clone(),
                reserve.clone(),
            );

            let stream = ScheduledTaskStream::new(tasks, self.ordered, concurrency);
            return Ok(match reserve {
                Some(_) => stream,
                None => stream.with_row_limit(row_limit),
            });
        }

        // No filter (or no limit): build every task eagerly so the IO system sees all split
        // ranges up front. A no-filter limit is applied to each selection mask inside `execute`,
        // which reserves in split order and so stays exact for ordered scans too.
        let tasks = TaskStream::eager(handle, self.execute(row_range, row_limit)?);

        let ordered = self.ordered;
        Ok(ScheduledTaskStream::new(tasks, ordered, concurrency))
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
