// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::iter;
use std::ops::Range;
use std::sync::Arc;
use std::task::Poll;

use futures::Future;
use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::future;
use futures::future::BoxFuture;
use futures::ready;
use futures::stream;
use futures::stream::BoxStream;
use itertools::Either;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::row_mask::RowMask;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;
use crate::scan::limit::RowLimit;
use crate::scan::limit::ScanLimit;
use crate::scan::splits::Splits;
use crate::scan::tasks::SplitFuture;
use crate::scan::tasks::TaskContext;
use crate::scan::tasks::split_exec;
use crate::scan::tasks::split_projection;

/// A projected subset (by indices, range, and filter) of rows from a Vortex data source.
///
/// The method of this struct enable, possibly concurrent, scanning of multiple row ranges of this
/// data source.
pub struct RepeatedScan<A> {
    session: VortexSession,
    layout_reader: LayoutReaderRef,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    ordered: bool,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    selection: Selection,
    /// The natural splits of the file.
    splits: Splits,
    /// The number of splits to make progress on concurrently **per-thread**.
    concurrency: usize,
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: MapFn<A>,
    /// Maximal number of rows to read (after filtering).
    limit: Option<ScanLimit>,
    /// The dtype of the projected arrays.
    dtype: DType,
}

/// A function applied to each projected split array.
type MapFn<A> = Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>;

impl RepeatedScan<ArrayRef> {
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
}

impl<A: 'static + Send> RepeatedScan<A> {
    /// Create split futures for an executor that schedules its own scan work.
    ///
    /// No tasks are spawned. Each future returns the mapped split, or `None` when its split is
    /// filtered out. The futures may be polled independently in any order.
    ///
    /// Scans with row limits must use [`Self::execute_stream`] so the scan can coordinate the
    /// limit across splits.
    pub fn execute(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        if self.limit.is_some() {
            vortex_bail!("Split futures do not support row limits; use a scan stream or iterator");
        }

        let ctx = self.task_context();
        self.split_masks(row_range)
            .map(|row_mask| Ok(map_split(split_exec(&ctx, row_mask)?, &self.map_fn)))
            .collect()
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Constructor just to allow `scan_builder` to create a `RepeatedScan`.
    #[expect(
        clippy::too_many_arguments,
        reason = "all arguments are needed for scan construction"
    )]
    pub(crate) fn new(
        session: VortexSession,
        layout_reader: LayoutReaderRef,
        projection: BoundExpression,
        filter: Option<BoundExpression>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: Splits,
        concurrency: usize,
        map_fn: MapFn<A>,
        limit: Option<ScanLimit>,
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
            map_fn,
            limit,
            dtype,
        }
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

    /// The non-empty row masks of the splits within `row_range`, in split order.
    fn split_masks(&self, row_range: Option<Range<u64>>) -> impl Iterator<Item = RowMask> + '_ {
        self.split_ranges(row_range)
            .into_iter()
            .filter(|range| range.start < range.end)
            .map(|range| self.selection.row_mask(&range))
            .filter(|row_mask| !row_mask.mask().all_false())
    }

    /// Execute the scan over `row_range` as a stream of the mapped split arrays.
    ///
    /// The stream ends after yielding its first error.
    pub fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        let concurrency = self.concurrency * get_available_parallelism().unwrap_or(1);
        let handle = self.session.handle();
        let ordered = self.ordered;
        let ctx = self.task_context();

        let map_fn = &self.map_fn;

        let arrays = match self.limit.as_ref().map(ScanLimit::budget) {
            // With a filter, a split's output row count is unknown until its filter has run.
            // Splits therefore filter and project ahead of the consumer, and the rows they
            // return are taken from the limit as they are emitted, discarding the excess. Split
            // tasks are built lazily so that no new splits start once the budget is spent. The
            // map function needs the trimmed arrays, so it runs in a second stage of CPU tasks.
            Some(limit) if ctx.filter.is_some() => {
                let gate = limit.clone();
                let selection = self.selection.clone();
                let split_handle = handle.clone();
                let tasks = stream::iter(self.split_ranges(row_range))
                    .take_while(move |_| future::ready(!gate.is_exhausted()))
                    .map(move |row_range| {
                        let ctx = Arc::clone(&ctx);
                        let selection = selection.clone();
                        split_handle.spawn_nested(move |handle| async move {
                            let task = handle
                                .spawn_cpu(move || {
                                    let row_mask = selection.row_mask(&row_range);
                                    if row_mask.mask().all_false() {
                                        return Ok(None);
                                    }
                                    split_exec(&ctx, row_mask).map(Some)
                                })
                                .await?;
                            match task {
                                Some(task) => task.await,
                                None => Ok(None),
                            }
                        })
                    });
                let map_fn = Arc::clone(map_fn);
                let arrays = buffer(tasks, ordered, concurrency)
                    .filter_map(|array| future::ready(array.transpose()))
                    .boxed();
                let mapped = limit_rows(arrays, limit).map(move |array| {
                    let map_fn = Arc::clone(&map_fn);
                    handle.spawn_cpu(move || map_fn(array?))
                });
                buffer(mapped, ordered, concurrency)
            }
            limit => {
                // Build eagerly to register I/O up front. Without a filter, selected row counts
                // are exact, so a limit can also be reserved here in split order.
                let mut tasks = Vec::new();
                for row_mask in self.split_masks(row_range) {
                    let task = if let Some(limit) = &limit {
                        if limit.is_exhausted() {
                            break;
                        }
                        let mask = limit.limit(row_mask.mask().clone());
                        split_projection(&ctx, &row_mask.row_range(), mask)?
                    } else {
                        split_exec(&ctx, row_mask)?
                    };
                    tasks.push(map_split(task, map_fn));
                }
                buffer(
                    stream::iter(tasks).map(move |task| handle.spawn(task)),
                    ordered,
                    concurrency,
                )
                .filter_map(|array| future::ready(array.transpose()))
                .boxed()
            }
        };

        Ok(stop_on_error(arrays))
    }
}

/// Apply `map_fn` to a split's array within the split's task.
fn map_split<A: 'static + Send>(
    task: SplitFuture,
    map_fn: &MapFn<A>,
) -> BoxFuture<'static, VortexResult<Option<A>>> {
    let map_fn = Arc::clone(map_fn);
    task.map(move |array| array?.map(|array| map_fn(array)).transpose())
        .boxed()
}

/// Trim arrays to the budget and drop upstream work as soon as it is exhausted.
fn limit_rows(
    arrays: BoxStream<'static, VortexResult<ArrayRef>>,
    limit: RowLimit,
) -> BoxStream<'static, VortexResult<ArrayRef>> {
    let mut arrays = Some(arrays);
    stream::poll_fn(move |cx| {
        loop {
            if limit.is_exhausted() {
                arrays = None;
            }
            let Some(stream) = arrays.as_mut() else {
                return Poll::Ready(None);
            };
            let array = match ready!(stream.poll_next_unpin(cx)) {
                Some(Ok(array)) => array,
                end => {
                    arrays = None;
                    return Poll::Ready(end);
                }
            };
            let granted = limit.take(array.len());
            if limit.is_exhausted() {
                arrays = None;
            }
            if granted != 0 {
                return Poll::Ready(Some(if granted < array.len() {
                    array.slice(0..granted)
                } else {
                    Ok(array)
                }));
            }
        }
    })
    .boxed()
}

/// Yield the first error and drop upstream without polling it again.
fn stop_on_error<T: Send + 'static>(
    stream: BoxStream<'static, VortexResult<T>>,
) -> BoxStream<'static, VortexResult<T>> {
    let mut stream = Some(stream);
    stream::poll_fn(move |cx| {
        let Some(input) = stream.as_mut() else {
            return Poll::Ready(None);
        };
        let item = ready!(input.poll_next_unpin(cx));
        if !matches!(item, Some(Ok(_))) {
            stream = None;
        }
        Poll::Ready(item)
    })
    .boxed()
}

/// Run up to `concurrency` futures of `tasks` at once, in order when `ordered` is set.
fn buffer<T: Send + 'static>(
    tasks: impl Stream<Item = impl Future<Output = T> + Send + 'static> + Send + 'static,
    ordered: bool,
    concurrency: usize,
) -> BoxStream<'static, T> {
    if ordered {
        tasks.buffered(concurrency).boxed()
    } else {
        tasks.buffer_unordered(concurrency).boxed()
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
mod tests {
    use std::sync::Arc;
    use std::task::Poll;

    use futures::FutureExt;
    use futures::StreamExt;
    use futures::stream;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::limit_rows;
    use super::stop_on_error;
    use crate::scan::limit::RowLimit;

    #[rstest]
    #[case::whole_array(4)]
    #[case::trimmed_array(2)]
    fn limit_ends_before_pending_work(#[case] rows: usize) -> VortexResult<()> {
        let array = PrimitiveArray::from_iter(0..4i32).into_array();
        let alive = Arc::new(());
        let dropped = Arc::downgrade(&alive);
        let pending = stream::poll_fn(move |_| {
            let _ = &alive;
            Poll::Pending
        });
        let arrays = stream::iter([Ok(array)]).chain(pending).boxed();
        let mut limited = limit_rows(arrays, RowLimit::new(rows as u64));

        let array = limited
            .next()
            .now_or_never()
            .expect("ready")
            .expect("array")?;
        assert_eq!(array.len(), rows);
        assert!(dropped.upgrade().is_none());
        assert!(matches!(limited.next().now_or_never(), Some(None)));
        Ok(())
    }

    #[test]
    fn exhausted_shared_limit_ends_before_pending_work() {
        let limit = RowLimit::new(1);
        let mut limited = limit_rows(stream::pending().boxed(), limit.clone());
        assert!(limited.next().now_or_never().is_none());

        assert_eq!(limit.take(1), 1);
        assert!(matches!(limited.next().now_or_never(), Some(None)));
    }

    #[test]
    fn zero_limit_does_not_poll_input() {
        let arrays = stream::poll_fn(|_| panic!("zero limit must not poll input")).boxed();
        let mut limited = limit_rows(arrays, RowLimit::new(0));
        assert!(matches!(limited.next().now_or_never(), Some(None)));
    }

    #[test]
    fn error_ends_before_pending_work() {
        let arrays = stream::iter([Err::<(), _>(vortex_err!("scan failed"))])
            .chain(stream::pending())
            .boxed();
        let mut stream = stop_on_error(arrays);

        assert!(matches!(stream.next().now_or_never(), Some(Some(Err(_)))));
        assert!(matches!(stream.next().now_or_never(), Some(None)));
    }
}
