// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutRef;
use vortex_layout::plan::EvalPlan;
use vortex_layout::plan::PlanExecutionContext;
use vortex_layout::plan::PlanRef;
use vortex_layout::plan::RowIdxPlan;
use vortex_layout::plan::lower;
use vortex_layout::plan::optimize;
use vortex_layout::segments::SegmentSource;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::RepeatedScan;
use crate::splits::SplitBy;
use crate::splits::Splits;
use crate::splits::attempt_split_ranges;

/// Builds a plan-native scan without constructing a layout reader.
pub struct ScanBuilder<A> {
    execution: PlanExecutionContext,
    base_plan: PlanRef,
    projection: Expression,
    filter: Option<Expression>,
    ordered: bool,
    row_range: Option<Range<u64>>,
    selection: Selection,
    split_by: SplitBy,
    concurrency: usize,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    limit: Option<u64>,
    row_offset: u64,
}

impl ScanBuilder<ArrayRef> {
    /// Creates a plan-native scan directly from a stored layout.
    pub fn try_new(
        layout: &LayoutRef,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> VortexResult<Self> {
        tracing::debug!(
            target: "vortex_scan_v2::planner",
            layout = %layout.display_tree(),
            "building a plan-native scan from a layout"
        );
        let plan = lower(layout)?;
        tracing::debug!(
            target: "vortex_scan_v2::planner",
            plan = %plan.display_tree(),
            "constructed the source physical plan"
        );
        Ok(Self::from_plan(
            plan,
            PlanExecutionContext::new(segment_source, session),
        ))
    }

    /// Creates a scan from an already constructed physical plan.
    pub fn from_plan(base_plan: PlanRef, execution: PlanExecutionContext) -> Self {
        Self {
            execution,
            base_plan,
            projection: root(),
            filter: None,
            ordered: true,
            row_range: None,
            selection: Selection::default(),
            split_by: SplitBy::default(),
            concurrency: 4,
            map_fn: Arc::new(Ok),
            limit: None,
            row_offset: 0,
        }
    }

    /// Returns an asynchronous stream of Vortex arrays.
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype()?;
        Ok(ArrayStreamAdapter::new(dtype, self.into_stream()?))
    }

    /// Returns a blocking iterator of Vortex arrays.
    pub fn into_array_iter<B: BlockingRuntime>(
        self,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let stream = self.into_array_stream()?;
        let dtype = stream.dtype().clone();
        Ok(ArrayIteratorAdapter::new(
            dtype,
            runtime.block_on_stream(stream),
        ))
    }
}

impl<A: 'static + Send> ScanBuilder<A> {
    /// Sets the filter expression.
    pub fn with_filter(mut self, filter: Expression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets or clears the filter expression.
    pub fn with_some_filter(mut self, filter: Option<Expression>) -> Self {
        self.filter = filter;
        self
    }

    /// Sets the projection expression.
    pub fn with_projection(mut self, projection: Expression) -> Self {
        self.projection = projection;
        self
    }

    /// Returns whether output splits retain row order.
    pub fn ordered(&self) -> bool {
        self.ordered
    }

    /// Configures whether output splits retain row order.
    pub fn with_ordered(mut self, ordered: bool) -> Self {
        self.ordered = ordered;
        self
    }

    /// Restricts the scan to a contiguous row range.
    pub fn with_row_range(mut self, row_range: Range<u64>) -> Self {
        self.row_range = Some(row_range);
        self
    }

    /// Applies an additional row selection.
    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// Selects strictly sorted absolute row indices relative to the scan input.
    pub fn with_row_indices(mut self, row_indices: StrictSortedBuffer<u64>) -> Self {
        self.selection = Selection::IncludeByIndex(row_indices);
        self
    }

    /// Sets the global offset used by row-index expressions.
    pub fn with_row_offset(mut self, row_offset: u64) -> Self {
        self.row_offset = row_offset;
        self
    }

    /// Configures how scan work is split into tasks.
    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    /// Returns the per-worker split concurrency.
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// Sets the per-worker split concurrency.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0, "scan concurrency must be non-zero");
        self.concurrency = concurrency;
        self
    }

    /// Sets a maximum number of output rows.
    pub fn with_limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Sets or clears the maximum number of output rows.
    pub fn with_some_limit(mut self, limit: Option<u64>) -> Self {
        self.limit = limit;
        self
    }

    /// Returns the dtype produced by the projection expression.
    pub fn dtype(&self) -> VortexResult<DType> {
        self.projection.return_dtype(self.base_plan.dtype())
    }

    /// Returns the session used by plan execution.
    pub fn session(&self) -> &VortexSession {
        self.execution.session()
    }

    /// Maps every output array into another result type.
    pub fn map<B: 'static>(
        self,
        map_fn: impl Fn(A) -> VortexResult<B> + 'static + Send + Sync,
    ) -> ScanBuilder<B> {
        let old_map_fn = self.map_fn;
        ScanBuilder {
            execution: self.execution,
            base_plan: self.base_plan,
            projection: self.projection,
            filter: self.filter,
            ordered: self.ordered,
            row_range: self.row_range,
            selection: self.selection,
            split_by: self.split_by,
            concurrency: self.concurrency,
            map_fn: Arc::new(move |array| old_map_fn(array).and_then(&map_fn)),
            limit: self.limit,
            row_offset: self.row_offset,
        }
    }

    /// Constructs and optimizes the projection and filter plans.
    pub fn prepare(self) -> VortexResult<RepeatedScan<A>> {
        if self.filter.is_some() && self.limit.is_some() {
            vortex_bail!("Vortex doesn't support scans with both a filter and a limit")
        }

        let source = RowIdxPlan::new(self.row_offset, self.base_plan.clone()).into_plan();
        tracing::debug!(
            target: "vortex_scan_v2::planner",
            row_offset = self.row_offset,
            plan = %source.display_tree(),
            "planning expressions over the row-index-aware source"
        );
        tracing::debug!(
            target: "vortex_scan_v2::planner",
            expression = %self.projection,
            "optimizing the projection expression"
        );
        let projection = self
            .projection
            .optimize_recursive(source.dtype())?
            .bind(source.dtype())?;
        let projection: PlanRef = EvalPlan::try_new(projection, source.clone())?.into_plan();
        let projection = optimize(projection)?;
        tracing::debug!(
            target: "vortex_scan_v2::planner",
            plan = %projection.display_tree(),
            "optimized the projection physical plan"
        );

        let filter = self
            .filter
            .map(|filter| -> VortexResult<PlanRef> {
                tracing::debug!(
                    target: "vortex_scan_v2::planner",
                    expression = %filter,
                    "optimizing the filter expression"
                );
                let filter = filter
                    .optimize_recursive(source.dtype())?
                    .bind(source.dtype())?;
                let filter: PlanRef = EvalPlan::try_new(filter, source.clone())?.into_plan();
                let filter = optimize(filter)?;
                vortex_ensure!(
                    filter.dtype().is_boolean(),
                    "Filter plan must produce booleans"
                );
                Ok(filter)
            })
            .transpose()?;
        if let Some(filter) = &filter {
            tracing::debug!(
                target: "vortex_scan_v2::planner",
                plan = %filter.display_tree(),
                "optimized the filter physical plan"
            );
        }

        let splits =
            if let Some(ranges) = attempt_split_ranges(&self.selection, self.row_range.as_ref()) {
                Splits::Ranges(ranges)
            } else {
                let row_range = self
                    .row_range
                    .clone()
                    .unwrap_or_else(|| 0..self.base_plan.row_count());
                let mut plans = vec![&projection];
                plans.extend(filter.as_ref());
                Splits::Natural(self.split_by.splits(&plans, &row_range)?)
            };
        match &splits {
            Splits::Natural(boundaries) => tracing::debug!(
                target: "vortex_scan_v2::planner",
                split_count = boundaries.len().saturating_sub(1),
                ?boundaries,
                "selected natural plan scan splits"
            ),
            Splits::Ranges(ranges) => tracing::debug!(
                target: "vortex_scan_v2::planner",
                split_count = ranges.len(),
                ?ranges,
                "selected sparse plan scan ranges"
            ),
        }

        Ok(RepeatedScan::new(
            self.execution,
            projection,
            filter,
            self.ordered,
            self.row_range,
            self.selection,
            splits,
            self.concurrency,
            self.map_fn,
            self.limit,
        ))
    }

    /// Builds one future per scan split.
    pub fn build(self) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        if self.limit.is_some_and(|limit| limit == 0) {
            return Ok(Vec::new());
        }
        self.prepare()?.execute(None)
    }

    /// Returns an asynchronous stream that schedules scan splits on the session runtime.
    pub fn into_stream(
        self,
    ) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        Ok(LazyScanStream::new(self))
    }

    /// Returns a blocking iterator over mapped scan outputs.
    pub fn into_iter<B: BlockingRuntime>(
        self,
        runtime: &B,
    ) -> VortexResult<impl Iterator<Item = VortexResult<A>> + 'static> {
        Ok(runtime.block_on_stream(self.into_stream()?))
    }
}

enum LazyScanState<A: 'static + Send> {
    Builder(Option<Box<ScanBuilder<A>>>),
    Preparing(PreparingScan<A>),
    Stream(BoxStream<'static, VortexResult<A>>),
    Error(Option<vortex_error::VortexError>),
}

type PreparedScanTasks<A> = Vec<BoxFuture<'static, VortexResult<Option<A>>>>;

struct PreparingScan<A: 'static + Send> {
    ordered: bool,
    concurrency: usize,
    handle: Handle,
    task: Task<VortexResult<PreparedScanTasks<A>>>,
}

struct LazyScanStream<A: 'static + Send> {
    state: LazyScanState<A>,
}

impl<A: 'static + Send> LazyScanStream<A> {
    fn new(builder: ScanBuilder<A>) -> Self {
        Self {
            state: LazyScanState::Builder(Some(Box::new(builder))),
        }
    }
}

impl<A: 'static + Send> Unpin for LazyScanStream<A> {}

impl<A: 'static + Send> Stream for LazyScanStream<A> {
    type Item = VortexResult<A>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                LazyScanState::Builder(builder) => {
                    let builder = builder.take().vortex_expect("polled after completion");
                    let ordered = builder.ordered;
                    let concurrency =
                        builder.concurrency * get_available_parallelism().unwrap_or(1);
                    let handle = builder.execution.session().handle();
                    let task = handle
                        .spawn_cpu(move || builder.prepare().and_then(|scan| scan.execute(None)));
                    self.state = LazyScanState::Preparing(PreparingScan {
                        ordered,
                        concurrency,
                        handle,
                        task,
                    });
                }
                LazyScanState::Preparing(preparing) => {
                    match ready!(Pin::new(&mut preparing.task).poll(cx)) {
                        Ok(tasks) => {
                            let handle = preparing.handle.clone();
                            let stream =
                                futures::stream::iter(tasks).map(move |task| handle.spawn(task));
                            let stream = if preparing.ordered {
                                stream.buffered(preparing.concurrency).boxed()
                            } else {
                                stream.buffer_unordered(preparing.concurrency).boxed()
                            };
                            self.state = LazyScanState::Stream(
                                stream
                                    .filter_map(|chunk| async move { chunk.transpose() })
                                    .boxed(),
                            );
                        }
                        Err(error) => self.state = LazyScanState::Error(Some(error)),
                    }
                }
                LazyScanState::Stream(stream) => return stream.as_mut().poll_next(cx),
                LazyScanState::Error(error) => return Poll::Ready(error.take().map(Err)),
            }
        }
    }
}
