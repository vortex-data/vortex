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
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::plan::PlanExecutionContext;
use vortex_layout::plan::PlanRef;
use vortex_scan::selection::Selection;
use vortex_utils::parallelism::get_available_parallelism;

use crate::filter::FilterPlan;
use crate::splits::Splits;
use crate::tasks::TaskContext;
use crate::tasks::split_exec;

/// A prepared plan-native scan that can be executed repeatedly over narrower row ranges.
pub struct RepeatedScan<A: 'static + Send> {
    execution: PlanExecutionContext,
    projection: PlanRef,
    pruning: Option<PlanRef>,
    filter: Option<FilterPlan>,
    ordered: bool,
    row_range: Option<Range<u64>>,
    selection: Selection,
    splits: Splits,
    concurrency: usize,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    limit: Option<u64>,
    dtype: DType,
}

impl RepeatedScan<ArrayRef> {
    /// Returns the dtype produced by this scan.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Executes the scan as a blocking array iterator.
    pub fn execute_array_iter<B: BlockingRuntime>(
        &self,
        row_range: Option<Range<u64>>,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        Ok(ArrayIteratorAdapter::new(
            dtype,
            runtime.block_on_stream(stream),
        ))
    }

    /// Executes the scan as an asynchronous array stream.
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
    #[expect(clippy::too_many_arguments, reason = "scan construction state")]
    pub(crate) fn new(
        execution: PlanExecutionContext,
        projection: PlanRef,
        pruning: Option<PlanRef>,
        filter: Option<FilterPlan>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: Splits,
        concurrency: usize,
        map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
        limit: Option<u64>,
    ) -> Self {
        let dtype = projection.dtype().clone();
        Self {
            execution,
            projection,
            pruning,
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

    /// Constructs one execution future per selected row split.
    pub fn execute(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        let selection_range = match &self.selection {
            Selection::IncludeByIndex(indices) if !indices.is_empty() => {
                Some(indices[0]..indices[indices.len() - 1] + 1)
            }
            Selection::IncludeRoaring(indices) if !indices.is_empty() => Some(
                indices.min().vortex_expect("non-empty selection")
                    ..indices.max().vortex_expect("non-empty selection") + 1,
            ),
            _ => None,
        };
        let row_range = intersect_ranges(self.row_range.as_ref(), row_range);
        let row_range = intersect_ranges(row_range.as_ref(), selection_range);

        let ranges = match &self.splits {
            Splits::Natural(boundaries) => {
                let boundaries = match row_range {
                    None => Either::Left(boundaries.iter().copied()),
                    Some(range) => {
                        if range.is_empty() {
                            return Ok(Vec::new());
                        }
                        let start = boundaries.partition_point(|&point| point < range.start);
                        let end = boundaries.partition_point(|&point| point < range.end);
                        Either::Right(
                            iter::once(range.start)
                                .chain(boundaries[start..end].iter().copied())
                                .chain(iter::once(range.end)),
                        )
                    }
                };
                Either::Left(boundaries.tuple_windows().map(|(start, end)| start..end))
            }
            Splits::Ranges(ranges) => Either::Right(match row_range {
                None => Either::Left(ranges.iter().cloned()),
                Some(range) => {
                    if range.is_empty() {
                        return Ok(Vec::new());
                    }
                    Either::Right(ranges.iter().filter_map(move |candidate| {
                        let start = cmp::max(candidate.start, range.start);
                        let end = cmp::min(candidate.end, range.end);
                        (start < end).then_some(start..end)
                    }))
                }
            }),
        };

        let ctx = Arc::new(TaskContext {
            execution: self.execution.clone(),
            pruning: self.pruning.clone(),
            filter: self.filter.clone(),
            projection: self.projection.clone(),
            mapper: Arc::clone(&self.map_fn),
        });
        let mut limit = self.limit;
        let mut tasks = Vec::new();
        for range in ranges {
            let row_mask = self.selection.row_mask(&range);
            if row_mask.mask().all_false() {
                continue;
            }
            tasks.push(split_exec(Arc::clone(&ctx), row_mask, limit.as_mut())?);
            if limit.is_some_and(|limit| limit == 0) {
                break;
            }
        }
        Ok(tasks)
    }

    /// Executes all selected row splits with the configured ordering and concurrency.
    pub fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        use futures::StreamExt;

        let concurrency = self.concurrency * get_available_parallelism().unwrap_or(1);
        let handle = self.execution.session().handle();
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
        (None, Some(right)) => Some(right),
        (Some(left), None) => Some(left.clone()),
        (Some(left), Some(right)) => {
            Some(cmp::max(left.start, right.start)..cmp::min(left.end, right.end))
        }
    }
}
