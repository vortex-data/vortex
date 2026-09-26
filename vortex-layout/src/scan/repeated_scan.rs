// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::iter;
use std::ops::Range;
use std::sync::Arc;

use futures::Stream;
use futures::future::BoxFuture;
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
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;
use crate::scan::limit::LimitBudget;
use crate::scan::limit::LimitClaim;
use crate::scan::splits::Splits;
use crate::scan::tasks::SplitLimit;
use crate::scan::tasks::TaskContext;
use crate::scan::tasks::split_exec;

pub(crate) type ScanTasks<A> = Vec<BoxFuture<'static, VortexResult<Option<A>>>>;

/// The split tasks of a scan, along with the shared limit budget when the scan has both a filter
/// and a limit.
pub(crate) struct PreparedTasks<A> {
    pub tasks: ScanTasks<A>,
    pub limit_budget: Option<Arc<LimitBudget>>,
}

impl<A: 'static + Send> PreparedTasks<A> {
    /// Iterate the tasks, stopping once the limit budget, if any, is exhausted.
    ///
    /// Tasks yielded before the budget ran out still resolve normally, while the remaining tasks
    /// are never started.
    pub fn into_limited_iter(
        self,
    ) -> impl Iterator<Item = BoxFuture<'static, VortexResult<Option<A>>>> {
        let budget = self.limit_budget;
        self.tasks
            .into_iter()
            .take_while(move |_| !budget.as_ref().is_some_and(|b| b.is_exhausted()))
    }
}

/// A projected subset (by indices, range, and filter) of rows from a Vortex data source.
///
/// The method of this struct enable, possibly concurrent, scanning of multiple row ranges of this
/// data source.
pub struct RepeatedScan<A: 'static + Send> {
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
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<u64>,
    /// The dtype of the projected arrays.
    dtype: DType,
}

impl RepeatedScan<ArrayRef> {
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
}

impl<A: 'static + Send> RepeatedScan<A> {
    /// Constructor just to allow `scan_builder` to create a `RepeatedScan`.
    #[expect(
        clippy::too_many_arguments,
        reason = "all arguments are needed for scan construction"
    )]
    pub fn new(
        session: VortexSession,
        layout_reader: LayoutReaderRef,
        projection: BoundExpression,
        filter: Option<BoundExpression>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: Splits,
        concurrency: usize,
        map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
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
            map_fn,
            limit,
            dtype,
        }
    }

    /// Construct a task per row split of the scan.
    ///
    /// When the scan is ordered and has both a filter and a limit, a task near the limit boundary
    /// may wait for the tasks before it to finish filtering, so the tasks must be driven
    /// concurrently (for example by spawning them) and earlier tasks must not be held back.
    pub fn execute(&self, row_range: Option<Range<u64>>) -> VortexResult<ScanTasks<A>> {
        Ok(self.execute_tasks(row_range)?.tasks)
    }

    pub(crate) fn execute_tasks(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<PreparedTasks<A>> {
        let no_tasks = || PreparedTasks {
            tasks: Vec::new(),
            limit_budget: None,
        };
        if self.limit == Some(0) {
            return Ok(no_tasks());
        }

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

        let ranges = match &self.splits {
            Splits::Natural(vec) => {
                debug_assert!(vec.is_sorted());
                let splits_iter = match row_range {
                    None => Either::Left(vec.iter().copied()),
                    Some(range) => {
                        if range.is_empty() {
                            return Ok(no_tasks());
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

                Either::Left(splits_iter.tuple_windows().map(|(start, end)| start..end))
            }
            Splits::Ranges(ranges) => Either::Right(match row_range {
                None => Either::Left(ranges.iter().cloned()),
                Some(range) => {
                    if range.is_empty() {
                        return Ok(no_tasks());
                    }
                    Either::Right(ranges.iter().filter_map(move |r| {
                        let start = cmp::max(r.start, range.start);
                        let end = cmp::min(r.end, range.end);
                        (start < end).then_some(start..end)
                    }))
                }
            }),
        };

        let ctx = Arc::new(TaskContext {
            filter: self.filter.clone().map(|f| Arc::new(FilterExpr::new(f))),
            reader: Arc::clone(&self.layout_reader),
            projection: self.projection.clone(),
            mapper: Arc::clone(&self.map_fn),
        });

        let row_masks: Vec<_> = ranges
            .map(|range| self.selection.row_mask(&range))
            .filter(|row_mask| !row_mask.mask().all_false())
            .collect();

        // Without a filter, each split's row count is known up front and the limit is applied
        // eagerly. With a filter, splits claim rows from a shared budget after filtering.
        let mut eager_limit = self.limit.filter(|_| self.filter.is_none());
        let limit_budget = self.limit.filter(|_| self.filter.is_some()).map(|limit| {
            let split_rows: Vec<u64> = row_masks
                .iter()
                .map(|row_mask| row_mask.mask().true_count() as u64)
                .collect();
            LimitBudget::new(limit, self.ordered, &split_rows)
        });

        let mut tasks = Vec::with_capacity(row_masks.len());
        for (split, row_mask) in row_masks.into_iter().enumerate() {
            let limit = match (eager_limit.as_mut(), limit_budget.as_ref()) {
                (Some(l), _) => SplitLimit::Eager(l),
                (None, Some(budget)) => SplitLimit::Filtered(LimitClaim::new(budget, split)),
                (None, None) => SplitLimit::None,
            };
            tasks.push(split_exec(Arc::clone(&ctx), row_mask, limit)?);
            if eager_limit.is_some_and(|l| l == 0) {
                break;
            }
        }

        Ok(PreparedTasks {
            tasks,
            limit_budget,
        })
    }

    pub fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        use futures::StreamExt;
        let num_workers = get_available_parallelism().unwrap_or(1);
        let concurrency = self.concurrency * num_workers;
        let handle = self.session.handle();

        let stream = futures::stream::iter(self.execute_tasks(row_range)?.into_limited_iter())
            .map(move |task| handle.spawn(task));

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
