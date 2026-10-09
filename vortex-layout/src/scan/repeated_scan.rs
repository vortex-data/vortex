// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::iter;
use std::ops::Range;
use std::sync::Arc;

use futures::Future;
use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::future;
use futures::future::BoxFuture;
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
use crate::scan::tasks::TaskContext;
use crate::scan::tasks::split_exec;
use crate::scan::tasks::split_projection;

/// A projected subset (by indices, range, and filter) of rows from a Vortex data source.
///
/// The method of this struct enable, possibly concurrent, scanning of multiple row ranges of this
/// data source.
pub struct RepeatedScan {
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
    /// Maximal number of rows to read (after filtering).
    limit: Option<ScanLimit>,
    /// The dtype of the projected arrays.
    dtype: DType,
}

impl RepeatedScan {
    /// Create split futures for an executor that schedules its own scan work.
    ///
    /// No tasks are spawned. Each future returns the projected array, or `None` when its split
    /// is filtered out. The futures may be polled independently in any order.
    ///
    /// Scans with row limits must use [`Self::execute_array_stream`] or
    /// [`Self::execute_array_iter`] so the scan can coordinate the limit across splits.
    pub fn execute(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<ArrayRef>>>>> {
        if self.limit.is_some() {
            vortex_bail!("Split futures do not support row limits; use a scan stream or iterator");
        }

        let ctx = self.task_context();
        self.split_masks(row_range)
            .map(|row_mask| split_exec(&ctx, row_mask))
            .collect()
    }

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
        projection: BoundExpression,
        filter: Option<BoundExpression>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: Splits,
        concurrency: usize,
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

    /// Execute the scan over `row_range` as a stream of the projected split arrays.
    ///
    /// The stream ends after yielding its first error.
    pub(crate) fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<BoxStream<'static, VortexResult<ArrayRef>>> {
        let concurrency = self.concurrency * get_available_parallelism().unwrap_or(1);
        let handle = self.session.handle();
        let ordered = self.ordered;
        let ctx = self.task_context();

        // The limit applied to the emitted arrays, for scans that cannot reserve rows up front.
        let mut trim = None;
        let tasks = match self.limit.as_ref().map(ScanLimit::budget) {
            // Without a limit, build every task eagerly so the I/O system sees all split ranges
            // up front.
            None => {
                let tasks = self
                    .split_masks(row_range)
                    .map(|row_mask| split_exec(&ctx, row_mask))
                    .collect::<VortexResult<Vec<_>>>()?;
                buffer(
                    stream::iter(tasks).map(move |task| handle.spawn(task)),
                    ordered,
                    concurrency,
                )
            }
            // Without a filter, each split returns exactly its selected rows, so the limit can be
            // reserved eagerly in split order.
            Some(limit) if ctx.filter.is_none() => {
                let mut tasks = Vec::new();
                for row_mask in self.split_masks(row_range) {
                    if limit.is_exhausted() {
                        break;
                    }
                    let mask = limit.limit(row_mask.mask().clone());
                    tasks.push(split_projection(&ctx, &row_mask.row_range(), mask)?);
                }
                buffer(
                    stream::iter(tasks).map(move |task| handle.spawn(task)),
                    ordered,
                    concurrency,
                )
            }
            // With a filter, a split's output row count is unknown until its filter has run.
            // Splits therefore filter and project ahead of the consumer, and the rows they
            // return are taken from the limit as they are emitted, discarding the excess. Split
            // tasks are built lazily so that no new splits start once the budget is spent.
            Some(limit) => {
                let gate = limit.clone();
                let selection = self.selection.clone();
                let tasks = stream::iter(self.split_ranges(row_range))
                    .take_while(move |_| future::ready(!gate.is_exhausted()))
                    .filter_map(move |row_range| {
                        let row_mask = selection.row_mask(&row_range);
                        let task = (!row_mask.mask().all_false()).then(|| {
                            match split_exec(&ctx, row_mask) {
                                Ok(task) => handle.spawn(task).boxed(),
                                Err(err) => future::ready(Err(err)).boxed(),
                            }
                        });
                        future::ready(task)
                    });
                trim = Some(limit);
                buffer(tasks, ordered, concurrency)
            }
        };

        let arrays = tasks.filter_map(|array| future::ready(array.transpose()));
        let arrays = match trim {
            Some(limit) => {
                let gate = limit.clone();
                arrays
                    .take_while(move |_| future::ready(!gate.is_exhausted()))
                    .filter_map(move |array| future::ready(take_rows(&limit, array)))
                    .boxed()
            }
            None => arrays.boxed(),
        };

        let mut errored = false;
        Ok(arrays
            .take_while(move |array| {
                let take = !errored;
                errored |= array.is_err();
                future::ready(take)
            })
            .boxed())
    }
}

/// Take `array`'s rows from `limit`, slicing off the rows past the budget.
///
/// Returns `None` when the budget is already spent.
fn take_rows(limit: &RowLimit, array: VortexResult<ArrayRef>) -> Option<VortexResult<ArrayRef>> {
    let Ok(array) = array else {
        return Some(array);
    };
    match limit.take(array.len()) {
        0 => None,
        granted if granted < array.len() => Some(array.slice(0..granted)),
        _ => Some(Ok(array)),
    }
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
