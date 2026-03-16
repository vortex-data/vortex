// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use futures::stream::FuturesUnordered;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
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
use vortex_layout::LayoutReaderRef;
use vortex_session::VortexSession;

use crate::filter::FilterExpr;
use crate::selection::Selection;
use crate::splits::Splits;
use crate::tasks::FilteredSplit;
use crate::tasks::TaskContext;
use crate::tasks::filter_split;
use crate::tasks::project_filtered_split;
use crate::tasks::split_exec;

const ADAPTIVE_SELECTIVITY_SAMPLE_SPLITS: usize = 4;
const HIGH_SURVIVOR_RATIO: f64 = 0.75;
const MAX_PROJECTION_SUBSPLITS_PER_SPLIT: usize = 8;
const IMMEDIATE_PROJECTION_FILTER_AHEAD_MULTIPLIER: usize = 2;

fn should_prefer_immediate_projection(
    observed_filter_splits: usize,
    observed_filter_rows: usize,
    observed_surviving_rows: usize,
) -> bool {
    observed_filter_splits >= ADAPTIVE_SELECTIVITY_SAMPLE_SPLITS
        && observed_filter_rows > 0
        && (observed_surviving_rows as f64 / observed_filter_rows as f64) >= HIGH_SURVIVOR_RATIO
}

/// A projected subset (by indices, range, and filter) of rows from a Vortex data source.
///
/// The method of this struct enable, possibly concurrent, scanning of multiple row ranges of this
/// data source.
pub struct RepeatedScan<A: 'static + Send> {
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
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<u64>,
    /// The dtype of the projected arrays.
    dtype: DType,
    projection_field_mask: Vec<FieldMask>,
    /// Preferred lower bound for rows accumulated before starting projection.
    target_output_rows: usize,
    /// Preferred lower bound for projected payload bytes accumulated before starting projection.
    target_output_bytes: usize,
    /// Estimated incremental projection payload bytes per surviving row.
    projection_row_cost_bytes: usize,
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
        let stream = self.default_stream(row_range)?;
        let iter = runtime.block_on_stream(stream);
        Ok(ArrayIteratorAdapter::new(dtype, iter))
    }

    pub fn execute_array_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.default_stream(row_range)?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
    }
}

impl<A: 'static + Send> RepeatedScan<A> {
    fn task_context(&self) -> Arc<TaskContext<A>> {
        Arc::new(TaskContext {
            selection: self.selection.clone(),
            filter: self.filter.clone().map(|f| Arc::new(FilterExpr::new(f))),
            reader: self.layout_reader.clone(),
            projection: self.projection.clone(),
            projection_field_mask: self.projection_field_mask.clone(),
            mapper: self.map_fn.clone(),
        })
    }

    fn effective_row_range(&self, row_range: Option<Range<u64>>) -> Option<Range<u64>> {
        intersect_ranges(self.row_range.as_ref(), row_range)
    }

    fn split_ranges(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Box<dyn Iterator<Item = Range<u64>> + Send>> {
        let Some(row_range) = self
            .effective_row_range(row_range)
            .or_else(|| Some(0..self.layout_reader.row_count()))
        else {
            return Ok(Box::new(std::iter::empty()));
        };

        self.splits.iter(self.layout_reader.as_ref(), row_range)
    }

    fn default_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let concurrency = self.concurrency * num_workers;
        self.execute_stream(row_range, concurrency, self.ordered, self.session.handle())
    }

    fn legacy_stream_from_ranges(
        &self,
        ctx: Arc<TaskContext<A>>,
        split_ranges: Vec<Range<u64>>,
        concurrency: usize,
        ordered: bool,
        handle: Handle,
    ) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        let mut limit = self.limit;
        let mut tasks = Vec::with_capacity(split_ranges.len());

        for range in split_ranges {
            if range.start >= range.end {
                continue;
            }

            if limit.is_some_and(|value| value == 0) {
                break;
            }

            tasks.push(split_exec(ctx.clone(), range, limit.as_mut())?);
        }

        let spawned = tasks.into_iter().map(move |task| handle.spawn(task));
        let stream = if ordered {
            stream::iter(spawned).buffered(concurrency).left_stream()
        } else {
            stream::iter(spawned)
                .buffer_unordered(concurrency)
                .right_stream()
        };

        Ok(stream
            .filter_map(|result| async move {
                match result {
                    Ok(Some(value)) => Some(Ok(value)),
                    Ok(None) => None,
                    Err(err) => Some(Err(err)),
                }
            })
            .boxed())
    }

    /// Constructor just to allow `scan_builder` to create a `RepeatedScan`.
    #[expect(
        clippy::too_many_arguments,
        reason = "all arguments are needed for scan construction"
    )]
    pub(super) fn new(
        session: VortexSession,
        layout_reader: LayoutReaderRef,
        projection: Expression,
        filter: Option<Expression>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: Splits,
        concurrency: usize,
        map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
        limit: Option<u64>,
        dtype: DType,
        projection_field_mask: Vec<FieldMask>,
        target_output_rows: usize,
        target_output_bytes: usize,
        projection_row_cost_bytes: usize,
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
            projection_field_mask,
            target_output_rows,
            target_output_bytes,
            projection_row_cost_bytes,
        }
    }

    pub fn execute(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        let ctx = self.task_context();
        let ranges = self.split_ranges(row_range)?;
        let mut limit = self.limit;
        let mut tasks = Vec::new();

        for range in ranges {
            if range.start >= range.end {
                continue;
            }

            if limit.is_some_and(|l| l == 0) {
                break;
            }

            tasks.push(split_exec(ctx.clone(), range, limit.as_mut())?);
        }

        Ok(tasks)
    }

    pub(crate) fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
        concurrency: usize,
        ordered: bool,
        handle: Handle,
    ) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        let ctx = self.task_context();
        let concurrency = concurrency.max(1);
        let filter_ahead = filter_ahead_for(concurrency, self.filter.is_some());
        let mut split_ranges = self.split_ranges(row_range)?;
        let mut prefetched_ranges = Vec::with_capacity(filter_ahead.saturating_add(1));
        let mut split_ranges_exhausted = true;

        while prefetched_ranges.len() <= filter_ahead {
            let Some(range) = split_ranges.next() else {
                break;
            };
            if range.start >= range.end {
                continue;
            }
            prefetched_ranges.push(range);
            if prefetched_ranges.len() > filter_ahead {
                split_ranges_exhausted = false;
                break;
            }
        }

        if should_fallback_to_legacy_stream(
            prefetched_ranges.len(),
            split_ranges_exhausted,
            filter_ahead,
        ) {
            return self.legacy_stream_from_ranges(
                ctx,
                prefetched_ranges,
                concurrency,
                ordered,
                handle,
            );
        }

        let split_ranges = Box::new(prefetched_ranges.into_iter().chain(split_ranges))
            as Box<dyn Iterator<Item = Range<u64>> + Send>;
        let mut staged = StagedSplitStream::new(
            ctx,
            split_ranges,
            self.limit,
            concurrency,
            ordered,
            handle,
            self.filter.is_some(),
            self.target_output_rows,
            self.target_output_bytes,
            self.projection_row_cost_bytes,
        );

        Ok(stream::poll_fn(move |cx| staged.poll_next(cx)).boxed())
    }
}

/// Two-phase concurrent split processor.
///
/// Splits flow through a pipeline:
///
///   split_ranges → filter.in_flight → filter.ready → projection.in_flight → emit
///
/// Filter tasks run ahead (up to `filter_ahead` splits) to discover which rows
/// survive before committing to projection IO. Projection starts once enough
/// filtered rows accumulate (by row count or byte estimate) or the filter
/// frontier is full. A fraction of the concurrency budget is reserved for filter
/// tasks to keep the pipeline fed.
struct StagedSplitStream<A: 'static + Send> {
    ctx: Arc<TaskContext<A>>,
    split_ranges: Box<dyn Iterator<Item = Range<u64>> + Send>,
    limit: Option<u64>,
    concurrency: usize,
    handle: Handle,
    filter_enabled: bool,
    filter_ahead: usize,
    split_ranges_exhausted: bool,
    next_split_idx: usize,
    prefer_immediate_projection: bool,
    observed_filter_splits: usize,
    observed_filter_rows: usize,
    observed_surviving_rows: usize,
    filter: FilterQueue,
    projection: ProjectionQueue<A>,
    emit: EmitQueue<A>,
}

type FilterTaskResult = (usize, usize, VortexResult<Option<FilteredSplit>>);
type ProjectionTaskResult<A> = (usize, usize, usize, VortexResult<A>);

struct FilterQueue {
    in_flight: FuturesUnordered<Task<FilterTaskResult>>,
    ready: BTreeMap<usize, FilteredSplit>,
    ready_rows: usize,
    ready_projection_bytes: usize,
}

impl FilterQueue {
    fn frontier_len(&self) -> usize {
        self.in_flight.len() + self.ready.len()
    }

    fn push_ready(&mut self, idx: usize, filtered: FilteredSplit, row_cost_bytes: usize) {
        let rows = filtered.mask.true_count();
        self.ready_rows += rows;
        self.ready_projection_bytes = self
            .ready_projection_bytes
            .saturating_add(rows.saturating_mul(row_cost_bytes));
        self.ready.insert(idx, filtered);
    }

    fn take_ready(&mut self, row_cost_bytes: usize) -> Option<(usize, FilteredSplit)> {
        let (idx, filtered) = self.ready.pop_first()?;
        let rows = filtered.mask.true_count();
        self.ready_rows = self.ready_rows.saturating_sub(rows);
        self.ready_projection_bytes = self
            .ready_projection_bytes
            .saturating_sub(rows.saturating_mul(row_cost_bytes));
        Some((idx, filtered))
    }
}

struct ProjectionQueue<A: 'static + Send> {
    in_flight: FuturesUnordered<Task<ProjectionTaskResult<A>>>,
    target_rows: usize,
    target_bytes: usize,
    row_cost_bytes: usize,
    subsplit_rows: usize,
}

struct EmitQueue<A> {
    ordered: bool,
    next_split_idx: usize,
    next_part_idx: usize,
    unordered: VecDeque<VortexResult<A>>,
    ordered_map: BTreeMap<usize, OrderedSplitOutput<A>>,
}

struct OrderedSplitOutput<A> {
    total_parts: usize,
    parts: BTreeMap<usize, VortexResult<A>>,
}

impl<A> EmitQueue<A> {
    fn queue(&mut self, idx: usize, value: Option<VortexResult<A>>) {
        self.queue_split(idx, value.into_iter().collect());
    }

    fn queue_split(&mut self, idx: usize, values: Vec<VortexResult<A>>) {
        if self.ordered {
            let parts: BTreeMap<usize, VortexResult<A>> = values.into_iter().enumerate().collect();
            self.ordered_map.insert(
                idx,
                OrderedSplitOutput {
                    total_parts: parts.len(),
                    parts,
                },
            );
        } else {
            self.unordered.extend(values);
        }
    }

    fn queue_part(
        &mut self,
        idx: usize,
        part_idx: usize,
        total_parts: usize,
        value: VortexResult<A>,
    ) {
        if self.ordered {
            let split = self
                .ordered_map
                .entry(idx)
                .or_insert_with(|| OrderedSplitOutput {
                    total_parts,
                    parts: BTreeMap::new(),
                });
            split.total_parts = total_parts;
            split.parts.insert(part_idx, value);
        } else {
            self.unordered.push_back(value);
        }
    }

    fn pop(&mut self) -> Option<VortexResult<A>> {
        if self.ordered {
            loop {
                let current = self.ordered_map.get_mut(&self.next_split_idx)?;
                if self.next_part_idx < current.total_parts {
                    if let Some(value) = current.parts.remove(&self.next_part_idx) {
                        self.next_part_idx += 1;
                        return Some(value);
                    }
                    return None;
                }

                self.ordered_map.remove(&self.next_split_idx);
                self.next_split_idx += 1;
                self.next_part_idx = 0;
            }
        } else {
            self.unordered.pop_front()
        }
    }

    fn is_empty(&self) -> bool {
        self.unordered.is_empty() && self.ordered_map.is_empty()
    }
}

impl<A: 'static + Send> StagedSplitStream<A> {
    #[expect(
        clippy::too_many_arguments,
        reason = "all arguments are part of staged scheduler construction"
    )]
    fn new(
        ctx: Arc<TaskContext<A>>,
        split_ranges: Box<dyn Iterator<Item = Range<u64>> + Send>,
        limit: Option<u64>,
        concurrency: usize,
        ordered: bool,
        handle: Handle,
        filter_enabled: bool,
        target_output_rows: usize,
        target_output_bytes: usize,
        projection_row_cost_bytes: usize,
    ) -> Self {
        let concurrency = concurrency.max(1);
        let filter_ahead = filter_ahead_for(concurrency, filter_enabled);

        Self {
            ctx,
            split_ranges,
            limit,
            concurrency,
            handle,
            filter_enabled,
            filter_ahead,
            split_ranges_exhausted: false,
            next_split_idx: 0,
            prefer_immediate_projection: false,
            observed_filter_splits: 0,
            observed_filter_rows: 0,
            observed_surviving_rows: 0,
            filter: FilterQueue {
                in_flight: FuturesUnordered::new(),
                ready: BTreeMap::new(),
                ready_rows: 0,
                ready_projection_bytes: 0,
            },
            projection: ProjectionQueue {
                in_flight: FuturesUnordered::new(),
                target_rows: target_output_rows.max(1),
                target_bytes: target_output_bytes,
                row_cost_bytes: projection_row_cost_bytes,
                subsplit_rows: target_output_rows.div_ceil(4).max(1),
            },
            emit: EmitQueue {
                ordered,
                next_split_idx: 0,
                next_part_idx: 0,
                unordered: VecDeque::new(),
                ordered_map: BTreeMap::new(),
            },
        }
    }

    fn effective_filter_ahead(&self) -> usize {
        if self.prefer_immediate_projection {
            immediate_projection_filter_ahead(self.filter_ahead, self.concurrency)
        } else {
            self.filter_ahead
        }
    }

    fn record_filter_observation(&mut self, candidate_rows: usize, surviving_rows: usize) {
        self.observed_filter_splits = self.observed_filter_splits.saturating_add(1);
        self.observed_filter_rows = self.observed_filter_rows.saturating_add(candidate_rows);
        self.observed_surviving_rows = self.observed_surviving_rows.saturating_add(surviving_rows);

        if self.prefer_immediate_projection || !self.filter_enabled {
            return;
        }

        if should_prefer_immediate_projection(
            self.observed_filter_splits,
            self.observed_filter_rows,
            self.observed_surviving_rows,
        ) {
            self.prefer_immediate_projection = true;
        }
    }

    /// Compute how many projection tasks can be spawned right now.
    ///
    /// Reserves ceil(concurrency/4) slots for filter tasks while the filter frontier
    /// is below the lookahead threshold, to keep the pipeline fed.
    fn available_projection_slots(&self) -> usize {
        let in_flight = self.filter.in_flight.len() + self.projection.in_flight.len();
        let available = self.concurrency.saturating_sub(in_flight);

        let needs_reserve = !self.prefer_immediate_projection
            && self.filter_enabled
            && !self.split_ranges_exhausted
            && self.filter.frontier_len() < self.effective_filter_ahead();
        if !needs_reserve {
            return available;
        }
        let reserved = self
            .concurrency
            .div_ceil(4)
            .saturating_sub(self.filter.in_flight.len());
        available.saturating_sub(reserved)
    }

    fn should_start_projection(&self) -> bool {
        if self.filter.ready.is_empty() {
            return false;
        }

        if !self.filter_enabled {
            return true;
        }

        if self.prefer_immediate_projection {
            return true;
        }

        if self.filter.ready_rows >= self.projection.target_rows {
            return true;
        }

        if self.filter.ready_projection_bytes >= self.projection.target_bytes {
            return true;
        }

        if self.filter.frontier_len() >= self.effective_filter_ahead() {
            return true;
        }

        self.split_ranges_exhausted && self.filter.in_flight.is_empty()
    }

    fn spawn_projection_tasks(&mut self) -> bool {
        let mut progress = false;
        while self.available_projection_slots() > 0 && self.should_start_projection() {
            let Some((idx, filtered)) = self.filter.take_ready(self.projection.row_cost_bytes)
            else {
                break;
            };
            let parts = split_filtered_for_projection(filtered, self.projection.subsplit_rows);
            let total_parts = parts.len();

            for (part_idx, part) in parts.into_iter().enumerate() {
                progress |= self.spawn_projection_part(idx, part_idx, total_parts, part);
            }
        }
        progress
    }

    fn spawn_projection_part(
        &mut self,
        idx: usize,
        part_idx: usize,
        total_parts: usize,
        filtered: FilteredSplit,
    ) -> bool {
        match project_filtered_split(self.ctx.clone(), filtered) {
            Ok(task) => {
                self.projection.in_flight.push(
                    self.handle
                        .spawn(async move { (idx, part_idx, total_parts, task.await) }),
                );
                true
            }
            Err(err) => {
                self.emit.queue_part(idx, part_idx, total_parts, Err(err));
                true
            }
        }
    }

    fn spawn_filter_tasks(&mut self) -> bool {
        let mut progress = false;

        while !self.split_ranges_exhausted
            && self.filter.frontier_len() < self.effective_filter_ahead()
            && self.filter.in_flight.len() + self.projection.in_flight.len() < self.concurrency
        {
            let Some(range) = self.split_ranges.next() else {
                self.split_ranges_exhausted = true;
                break;
            };

            if range.start >= range.end {
                continue;
            }

            if self.limit.is_some_and(|value| value == 0) {
                self.split_ranges_exhausted = true;
                break;
            }

            let idx = self.next_split_idx;
            self.next_split_idx += 1;
            let split_rows =
                usize::try_from(range.end.saturating_sub(range.start)).unwrap_or(usize::MAX);

            match filter_split(self.ctx.clone(), range, self.limit.as_mut()) {
                Ok(task) => {
                    self.filter.in_flight.push(
                        self.handle
                            .spawn(async move { (idx, split_rows, task.await) }),
                    );
                }
                Err(err) => self.emit.queue(idx, Some(Err(err))),
            }
            progress = true;
        }

        progress
    }

    fn poll_projection_completions(&mut self, cx: &mut Context<'_>) -> bool {
        let mut progress = false;
        while let Poll::Ready(Some((idx, part_idx, total_parts, value))) =
            self.projection.in_flight.poll_next_unpin(cx)
        {
            self.emit.queue_part(idx, part_idx, total_parts, value);
            progress = true;
        }
        progress
    }

    fn poll_filter_completions(&mut self, cx: &mut Context<'_>) -> bool {
        let mut progress = false;
        while let Poll::Ready(Some(result)) = self.filter.in_flight.poll_next_unpin(cx) {
            match result {
                (idx, split_rows, Ok(Some(filtered))) => {
                    self.record_filter_observation(split_rows, filtered.mask.true_count());
                    self.filter
                        .push_ready(idx, filtered, self.projection.row_cost_bytes);
                }
                (idx, split_rows, Ok(None)) => {
                    self.record_filter_observation(split_rows, 0);
                    self.emit.queue(idx, None);
                }
                (idx, split_rows, Err(err)) => {
                    self.record_filter_observation(split_rows, 0);
                    self.emit.queue(idx, Some(Err(err)));
                }
            }
            progress = true;
        }
        progress
    }

    fn is_finished(&self) -> bool {
        self.split_ranges_exhausted
            && self.filter.in_flight.is_empty()
            && self.filter.ready.is_empty()
            && self.projection.in_flight.is_empty()
            && self.emit.is_empty()
    }

    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<Option<VortexResult<A>>> {
        // Each step below can produce ready items: poll_* via completed futures, and
        // spawn_* via synchronous error paths. We check emit.pop() after every step
        // so we yield results as early as possible rather than doing unnecessary work.
        loop {
            if let Some(value) = self.emit.pop() {
                return Poll::Ready(Some(value));
            }

            let mut progress = false;
            progress |= self.poll_projection_completions(cx);
            if let Some(value) = self.emit.pop() {
                return Poll::Ready(Some(value));
            }

            progress |= self.poll_filter_completions(cx);
            if let Some(value) = self.emit.pop() {
                return Poll::Ready(Some(value));
            }

            progress |= self.spawn_projection_tasks();
            if let Some(value) = self.emit.pop() {
                return Poll::Ready(Some(value));
            }

            progress |= self.spawn_filter_tasks();
            if let Some(value) = self.emit.pop() {
                return Poll::Ready(Some(value));
            }

            if self.is_finished() {
                return Poll::Ready(None);
            }

            if !progress {
                return Poll::Pending;
            }
        }
    }
}

fn filter_ahead_for(concurrency: usize, filter_enabled: bool) -> usize {
    if filter_enabled {
        concurrency.clamp(4, 16)
    } else {
        concurrency
    }
}

fn should_fallback_to_legacy_stream(
    prefetched_split_count: usize,
    split_ranges_exhausted: bool,
    filter_ahead: usize,
) -> bool {
    split_ranges_exhausted && prefetched_split_count <= filter_ahead
}

fn immediate_projection_filter_ahead(filter_ahead: usize, concurrency: usize) -> usize {
    filter_ahead
        .saturating_mul(IMMEDIATE_PROJECTION_FILTER_AHEAD_MULTIPLIER)
        .min(concurrency)
        .max(1)
}

fn split_filtered_for_projection(
    filtered: FilteredSplit,
    _target_rows: usize,
) -> Vec<FilteredSplit> {
    if filtered.projection_ranges.len() <= 1 || filtered.mask.true_count() == 0 {
        return vec![filtered];
    }

    let FilteredSplit {
        row_range,
        mask,
        projection_ranges,
    } = filtered;
    let group_size = projection_ranges
        .len()
        .div_ceil(MAX_PROJECTION_SUBSPLITS_PER_SPLIT)
        .max(1);
    let mut parts = Vec::with_capacity(projection_ranges.len().div_ceil(group_size));
    for grouped_ranges in projection_ranges.chunks(group_size) {
        let part_row_range = grouped_ranges
            .first()
            .vortex_expect("grouped projection ranges are non-empty")
            .start
            ..grouped_ranges
                .last()
                .vortex_expect("grouped projection ranges are non-empty")
                .end;
        let start_offset = usize::try_from(part_row_range.start.saturating_sub(row_range.start))
            .unwrap_or(usize::MAX);
        let end_offset = usize::try_from(part_row_range.end.saturating_sub(row_range.start))
            .unwrap_or(usize::MAX);
        let part_mask = mask.slice(start_offset..end_offset);
        if part_mask.all_false() {
            continue;
        }
        parts.push(FilteredSplit {
            row_range: part_row_range.clone(),
            mask: part_mask,
            projection_ranges: vec![part_row_range],
        });
    }

    if parts.is_empty() {
        vec![FilteredSplit {
            row_range: row_range.clone(),
            mask,
            projection_ranges: vec![row_range],
        }]
    } else {
        parts
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
mod test {
    use std::collections::BTreeMap;

    use futures::stream::FuturesUnordered;
    use vortex_mask::Mask;

    use crate::tasks::FilteredSplit;

    fn projection_slots_for(
        concurrency: usize,
        filter_enabled: bool,
        split_ranges_exhausted: bool,
        filter_ahead: usize,
        filter_in_flight: usize,
        filtered_ready: usize,
        projection_in_flight: usize,
    ) -> usize {
        let available_slots = concurrency.saturating_sub(filter_in_flight + projection_in_flight);
        if !filter_enabled
            || split_ranges_exhausted
            || filter_in_flight + filtered_ready >= filter_ahead
        {
            return available_slots;
        }

        let desired_filter_slots = concurrency.div_ceil(4);
        let reserved_filter_slots = desired_filter_slots.saturating_sub(filter_in_flight);
        available_slots.saturating_sub(reserved_filter_slots)
    }

    #[test]
    fn projection_reserves_capacity_for_filter_frontier() {
        assert_eq!(projection_slots_for(8, true, false, 8, 0, 0, 6), 0);
        assert_eq!(projection_slots_for(8, true, false, 8, 1, 0, 5), 1);
        assert_eq!(projection_slots_for(8, true, false, 8, 2, 0, 4), 2);
    }

    #[test]
    fn projection_uses_all_slots_when_filter_no_longer_needs_reserve() {
        assert_eq!(projection_slots_for(8, false, false, 8, 0, 0, 6), 2);
        assert_eq!(projection_slots_for(8, true, true, 8, 0, 0, 6), 2);
        assert_eq!(projection_slots_for(8, true, false, 2, 0, 2, 6), 2);
    }

    #[test]
    fn small_split_sets_fallback_to_legacy_stream() {
        assert!(super::should_fallback_to_legacy_stream(0, true, 4));
        assert!(super::should_fallback_to_legacy_stream(4, true, 4));
        assert!(!super::should_fallback_to_legacy_stream(5, true, 4));
        assert!(!super::should_fallback_to_legacy_stream(4, false, 4));
    }

    #[test]
    fn high_survivor_ratio_prefers_immediate_projection() {
        assert!(!super::should_prefer_immediate_projection(3, 300, 300));
        assert!(super::should_prefer_immediate_projection(4, 400, 400));
        assert!(super::should_prefer_immediate_projection(4, 400, 320));
    }

    #[test]
    fn immediate_projection_softens_filter_ahead_expansion() {
        assert_eq!(super::immediate_projection_filter_ahead(16, 32), 32);
        assert_eq!(super::immediate_projection_filter_ahead(8, 12), 12);
        assert_eq!(super::immediate_projection_filter_ahead(0, 0), 1);
    }

    #[test]
    fn low_survivor_ratio_keeps_filter_ahead() {
        assert!(!super::should_prefer_immediate_projection(4, 400, 200));
        assert!(!super::should_prefer_immediate_projection(8, 800, 300));
    }

    #[test]
    fn projection_splitter_subdivides_dense_masks() {
        let parts = super::split_filtered_for_projection(
            FilteredSplit {
                row_range: 100..164,
                mask: Mask::new_true(64),
                projection_ranges: vec![100..116, 116..132, 132..148, 148..164],
            },
            16,
        );

        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0].row_range, 100..116);
        assert_eq!(parts[1].row_range, 116..132);
        assert_eq!(parts[2].row_range, 132..148);
        assert_eq!(parts[3].row_range, 148..164);
        assert!(parts.iter().all(|part| part.mask.all_true()));
    }

    #[test]
    fn projection_splitter_subdivides_by_surviving_rows() {
        let parts = super::split_filtered_for_projection(
            FilteredSplit {
                row_range: 200..216,
                mask: Mask::from_indices(16, vec![1, 2, 3, 10, 12, 15]),
                projection_ranges: vec![200..203, 203..212, 212..216],
            },
            2,
        );

        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].row_range, 200..203);
        assert_eq!(parts[0].mask.true_count(), 2);
        assert_eq!(parts[1].row_range, 203..212);
        assert_eq!(parts[1].mask.true_count(), 2);
        assert_eq!(parts[2].row_range, 212..216);
        assert_eq!(parts[2].mask.true_count(), 2);
    }

    #[test]
    fn projection_splitter_keeps_all_ranges_when_capped() {
        let projection_ranges = (0..10)
            .map(|idx| {
                let start = idx * 10;
                start..start + 10
            })
            .collect::<Vec<_>>();
        let parts = super::split_filtered_for_projection(
            FilteredSplit {
                row_range: 0..100,
                mask: Mask::new_true(100),
                projection_ranges,
            },
            10,
        );

        assert!(parts.len() <= super::MAX_PROJECTION_SUBSPLITS_PER_SPLIT);
        assert_eq!(parts.first().expect("at least one part").row_range.start, 0);
        assert_eq!(parts.last().expect("at least one part").row_range.end, 100);
        assert_eq!(
            parts
                .iter()
                .map(|part| part.mask.true_count())
                .sum::<usize>(),
            100
        );
    }

    #[test]
    fn filter_queue_take_ready_pops_first_split() {
        let mut filter = super::FilterQueue {
            in_flight: FuturesUnordered::new(),
            ready: BTreeMap::from([
                (
                    0,
                    FilteredSplit {
                        row_range: 0..10,
                        mask: Mask::new_true(10),
                        projection_ranges: std::iter::once(0..10).collect(),
                    },
                ),
                (
                    1,
                    FilteredSplit {
                        row_range: 10..20,
                        mask: Mask::new_true(10),
                        projection_ranges: std::iter::once(10..20).collect(),
                    },
                ),
            ]),
            ready_rows: 20,
            ready_projection_bytes: 20,
        };

        let (idx, filtered) = filter.take_ready(1).expect("expected one split");
        assert_eq!(idx, 0);
        assert_eq!(filtered.row_range, 0..10);
        assert_eq!(filter.ready.keys().copied().collect::<Vec<_>>(), vec![1]);
        assert_eq!(filter.ready_rows, 10);
        assert_eq!(filter.ready_projection_bytes, 10);
    }
}
