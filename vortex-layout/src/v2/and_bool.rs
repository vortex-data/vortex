// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ConjunctPlan`] — k-way per-element AND of bool-stream children.
//!
//! Used by `Scan::build` to combine the per-conjunct mask streams
//! into a single mask that `FilterPlan` can apply.
//!
//! See `LAYOUT_PLAN.md` § Scan construction.

use std::ops::Range;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;
use std::time::Duration;
use std::time::Instant;

use async_stream::try_stream;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use kanal::AsyncReceiver;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;

use crate::mask_debug::mask_coordinate_summary;
use crate::v2::aligned::AlignedArrayStream;
use crate::v2::dataflow::DomainId;
use crate::v2::dataflow::FrontierSource;
use crate::v2::dataflow::GrantKey;
use crate::v2::dataflow::OrdinalDemand;
use crate::v2::dataflow::OutputEstimate;
use crate::v2::dataflow::OutputFrontier;
use crate::v2::dataflow::OutputGrant;
use crate::v2::dataflow::OutputGrantReason;
use crate::v2::dataflow::OutputGrantRequest;
use crate::v2::dataflow::SubplanId;
use crate::v2::demand::DemandSource;
use crate::v2::demand::Resource;
use crate::v2::demand::RowDemand;
use crate::v2::experiment::bool_var;
use crate::v2::experiment::usize_var;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Combines N bool-stream children into a single bool stream by
/// progressively AND-ing per row. Each conjunct window sees the mask
/// produced by earlier conjuncts for the same window, while the final
/// mask stays in the original row coordinate space.
pub struct ConjunctPlan {
    children: Vec<LayoutPlanRef>,
    conjuncts: Vec<ConjunctInfo>,
    output_dtype: DType,
    row_count: u64,
}

/// Backwards-compatible name for callers that still refer to the old
/// k-way AND plan. New code should use [`ConjunctPlan`].
pub type AndBoolStreamsPlan = ConjunctPlan;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ConjunctInfo {
    pub original_idx: usize,
    pub cost: u8,
    pub expr: String,
}

/// Default to natural child stream boundaries. Benchmark experiments
/// can raise this via `VORTEX_V2_CONJUNCT_MIN_ROWS` to quantify the
/// tradeoff between per-batch overhead and row-demand responsiveness.
const CONJUNCT_MIN_ROWS: usize = 1;

impl ConjunctPlan {
    pub fn new(children: Vec<LayoutPlanRef>, row_count: u64) -> Self {
        Self::with_conjuncts(children, Vec::new(), row_count)
    }

    pub fn with_conjuncts(
        children: Vec<LayoutPlanRef>,
        conjuncts: Vec<ConjunctInfo>,
        row_count: u64,
    ) -> Self {
        debug_assert!(
            !children.is_empty(),
            "ConjunctPlan needs at least one child"
        );
        debug_assert!(
            children
                .iter()
                .all(|c| matches!(c.schema(), DType::Bool(_))),
            "ConjunctPlan: every child must produce a Bool stream",
        );
        debug_assert!(
            conjuncts.is_empty() || conjuncts.len() == children.len(),
            "ConjunctPlan conjunct metadata must match child count",
        );
        // The result is always a non-nullable Bool — input nulls
        // are absorbed into the mask (None values are treated as
        // not-matching, same as the v1 filter pipeline).
        let output_dtype = DType::Bool(Nullability::NonNullable);
        Self {
            children,
            conjuncts,
            output_dtype,
            row_count,
        }
    }
}

impl PartialEq for ConjunctPlan {
    fn eq(&self, other: &Self) -> bool {
        self.children == other.children
            && self.output_dtype == other.output_dtype
            && self.row_count == other.row_count
    }
}

impl Eq for ConjunctPlan {}

impl std::hash::Hash for ConjunctPlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.children.hash(state);
        self.output_dtype.hash(state);
        self.row_count.hash(state);
    }
}

impl LayoutPlan for ConjunctPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        1
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition >= 1 {
            vortex_bail!("ConjunctPlan partition out of range: {partition}");
        }
        Ok(PartitionStats::for_range(0..self.row_count))
    }

    fn output_ordered(&self) -> bool {
        self.children.iter().all(|c| c.output_ordered())
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![true; self.children.len()]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true; self.children.len()]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        &self.children
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != self.children.len() {
            vortex_bail!(
                "ConjunctPlan::with_new_children expected {} children, got {}",
                self.children.len(),
                children.len()
            );
        }
        Ok(Arc::new(Self {
            children,
            conjuncts: self.conjuncts.clone(),
            output_dtype: self.output_dtype.clone(),
            row_count: self.row_count,
        }))
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        frontier: &OutputFrontier,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        if bool_var("VORTEX_V2_CONJUNCT_FRONTIER") {
            return self.execute_with_frontiers(row_range, demand, frontier, ctx);
        }

        let dtype = self.output_dtype.clone();
        let children = self.children.clone();
        let conjuncts = self.conjuncts.clone();
        let mut child_streams = Vec::with_capacity(children.len());
        for child in &children {
            child_streams.push(child.execute(row_range.clone(), demand, frontier, ctx)?);
        }
        let demand = demand.clone();
        let session = ctx.session().clone();
        let min_rows = usize_var("VORTEX_V2_CONJUNCT_MIN_ROWS").unwrap_or_else(|| {
            if bool_var("VORTEX_V2_ADAPTIVE_CONJUNCT_MIN_ROWS") {
                adaptive_conjunct_min_rows(conjuncts.as_slice())
            } else {
                CONJUNCT_MIN_ROWS
            }
        });
        let aligned =
            AlignedArrayStream::new_labeled(child_streams, ctx.session().handle(), "conjunct")
                .with_min_rows(min_rows);
        let debug_label = ctx.debug_label().map(str::to_owned);
        let dynamic_order = !bool_var("VORTEX_V2_STATIC_CONJUNCT_ORDER");
        let conjunct_demand = !bool_var("VORTEX_V2_DISABLE_CONJUNCT_DEMAND");
        let stream = try_stream! {
            let mut scheduler = ConjunctScheduler::new(children.len(), conjuncts.as_slice(), dynamic_order);
            let mut window_start = row_range.start;
            let mut aligned = Box::pin(aligned);

            while let Some(arrays) = aligned.next().await {
                let arrays = arrays?;
                let Some(window_len) = arrays.first().map(|array| array.len()) else {
                    continue;
                };
                if window_len == 0 {
                    continue;
                }
                let window_end = window_start + u64::try_from(window_len)?;
                if window_end > row_range.end {
                    Err(vortex_error::vortex_err!(
                        "ConjunctPlan aligned streams produced rows past requested range: {window_end} > {}",
                        row_range.end
                    ))?;
                }
                let window = window_start..window_end;
                window_start = window_end;
                let coord_range = demand.global_range(&window);

                // The aligned stream has already driven each child to
                // the next smallest chunk boundary. This preserves
                // dynamic child chunking without collecting the full
                // execute range. A later scheduler can replace
                // AlignedArrayStream's fixed producer buffers with
                // demand-aware permits so less selective children do
                // not run this far ahead.
                let arrays = arrays;
                for array in &arrays {
                    if array.len() != window_len {
                        Err(vortex_error::vortex_err!(
                            "ConjunctPlan aligned child length {} does not match window length {window_len}",
                            array.len()
                        ))?;
                    }
                }

                let mut exec_ctx = session.create_execution_ctx();
                let mut acc = Mask::new_true(window_len);
                let order = scheduler.order(conjuncts.as_slice());
                for child_idx in order {
                    if acc.true_count() == 0 {
                        break;
                    }

                    let input_mask = acc.clone();
                    let input_rows = input_mask.true_count();
                    let start = Instant::now();
                    let mask_result = if conjunct_demand {
                        execute_mask_with_demand(arrays[child_idx].clone(), &input_mask, &mut exec_ctx)?
                    } else {
                        execute_mask_without_demand(arrays[child_idx].clone(), &input_mask, &mut exec_ctx)?
                    };
                    let elapsed = start.elapsed();
                    let output_mask = mask_result.mask;
                    let compute_input_rows = mask_result.compute_input_rows;
                    let compute_output_rows = mask_result.compute_output_rows;
                    let output_rows = output_mask.true_count();
                    scheduler.observe(child_idx, compute_input_rows, compute_output_rows, elapsed);
                    if tracing::enabled!(tracing::Level::DEBUG) {
                        log_conjunct_eval(
                            child_idx,
                            conjuncts.as_slice(),
                            debug_label.as_deref(),
                            &window,
                            &coord_range,
                            input_rows,
                            output_rows,
                            compute_input_rows,
                            compute_output_rows,
                            elapsed,
                            &input_mask,
                            &output_mask,
                        );
                    }

                    acc = output_mask;
                }

                let bits = acc.to_bit_buffer();
                yield BoolArray::new(bits, Validity::NonNullable).into_array();
            }
            if window_start != row_range.end {
                Err(vortex_error::vortex_err!(
                    "ConjunctPlan aligned streams produced {} rows, expected {}",
                    window_start - row_range.start,
                    row_range.end - row_range.start
                ))?;
            }
        };

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }
}

impl ConjunctPlan {
    fn execute_with_frontiers(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        _frontier: &OutputFrontier,
        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        let dtype = self.output_dtype.clone();
        let children = self.children.clone();
        let conjuncts = self.conjuncts.clone();
        let demand = demand.clone();
        let ctx = ctx.clone();
        let session = ctx.session().clone();
        let debug_label = ctx.debug_label().map(str::to_owned);
        let dynamic_order = !bool_var("VORTEX_V2_STATIC_CONJUNCT_ORDER");
        let conjunct_demand = !bool_var("VORTEX_V2_DISABLE_CONJUNCT_DEMAND");
        let output_rows = usize_var("VORTEX_V2_CONJUNCT_FRONTIER_OUTPUT_ROWS")
            .unwrap_or(64 * 1024)
            .max(1);

        let stream = try_stream! {
            let total_len = usize::try_from(row_range.end - row_range.start)?;
            let total_len_u64 = u64::try_from(total_len)?;
            let coord_range = demand.global_range(&row_range);
            let mut scheduler = ConjunctScheduler::new(children.len(), conjuncts.as_slice(), dynamic_order);
            let lead_rows = u64::try_from(
                usize_var("VORTEX_V2_CONJUNCT_WEIGHTED_LEAD_ROWS")
                    .unwrap_or(output_rows)
                    .max(1),
            )?;

            let grant_source = Arc::new(ConjunctGrantSource::new(
                total_len_u64,
                scheduler.weights(conjuncts.as_slice()),
                lead_rows,
            ));
            let demand_source = Arc::new(ConjunctRuntimeDemand::new(
                demand.root_total_rows(),
                DomainId::new(0),
            ));
            let demand_source_trait: Arc<dyn DemandSource> =
                Arc::clone(&demand_source) as Arc<dyn DemandSource>;
            let staged_demand = demand.with_source(demand_source_trait);
            let frontier_source_trait: Arc<dyn FrontierSource> =
                Arc::clone(&grant_source) as Arc<dyn FrontierSource>;

            let mut raw_child_streams = Vec::with_capacity(children.len());
            for child_idx in 0..children.len() {
                let child_frontier = OutputFrontier::new(
                    Arc::clone(&frontier_source_trait),
                    GrantKey::new(DomainId::new(0), SubplanId::new(u32::try_from(child_idx + 1)?)),
                    total_len_u64,
                );
                raw_child_streams.push(children[child_idx].execute(
                    row_range.clone(),
                    &staged_demand,
                    &child_frontier,
                    &ctx,
                )?);
            }
            let buffer_depth = usize_var("VORTEX_V2_CONJUNCT_BUFFER_DEPTH").unwrap_or(2).max(1);
            let (mut child_streams, _producers) =
                spawn_conjunct_children(raw_child_streams, ctx.session().handle(), buffer_depth);

            let mut buffers: Vec<Option<ArrayRef>> = vec![None; children.len()];
            let mut child_positions = vec![0usize; children.len()];
            let mut acc = Mask::new_true(total_len);
            let mut exec_ctx = session.create_execution_ctx();

            let mut cursor = 0usize;
            while cursor < total_len {
                let window_end = (cursor + output_rows).min(total_len);
                let window_len = window_end - cursor;
                grant_source.update_weights(scheduler.weights(conjuncts.as_slice()))?;
                let order = scheduler.order(conjuncts.as_slice());
                let leader_idx = *order
                    .first()
                    .ok_or_else(|| vortex_error::vortex_err!("ConjunctPlan frontier path has no children"))?;
                let leader = next_child_window_exact(
                    leader_idx,
                    &mut child_streams,
                    &mut buffers,
                    &mut child_positions,
                    cursor..window_end,
                )
                .await?;
                if window_len == 0 {
                    continue;
                }
                let window = (row_range.start + u64::try_from(cursor)?)
                    ..(row_range.start + u64::try_from(window_end)?);
                let window_coord = (coord_range.start + u64::try_from(cursor)?)
                    ..(coord_range.start + u64::try_from(window_end)?);
                let mut window_acc = acc.slice(cursor..window_end);

                let input_mask = window_acc.clone();
                let input_rows = input_mask.true_count();
                let start = Instant::now();
                let result = if conjunct_demand {
                    execute_mask_with_demand(leader, &input_mask, &mut exec_ctx)?
                } else {
                    execute_mask_without_demand(leader, &input_mask, &mut exec_ctx)?
                };
                let elapsed = start.elapsed();
                scheduler.observe(
                    leader_idx,
                    result.compute_input_rows,
                    result.compute_output_rows,
                    elapsed,
                );
                if tracing::enabled!(tracing::Level::DEBUG) {
                    log_conjunct_eval(
                        leader_idx,
                        conjuncts.as_slice(),
                        debug_label.as_deref(),
                        &window,
                        &window_coord,
                        input_rows,
                        result.mask.true_count(),
                        result.compute_input_rows,
                        result.compute_output_rows,
                        elapsed,
                        &input_mask,
                        &result.mask,
                    );
                }
                grant_source.resolve(stage_key(leader_idx), u64::try_from(window_end)?)?;
                window_acc = result.mask;
                demand_source.publish(window_coord.clone(), window_acc.clone())?;
                grant_source.update_weights(scheduler.weights(conjuncts.as_slice()))?;

                let mut order = scheduler.order(conjuncts.as_slice());
                order.retain(|&idx| idx != leader_idx);
                for (order_idx, &child_idx) in order.iter().enumerate() {
                    if window_acc.all_false() {
                        for &skipped_idx in &order[order_idx..] {
                            grant_source
                                .resolve(stage_key(skipped_idx), u64::try_from(window_end)?)?;
                        }
                        break;
                    }
                    let child = next_child_window_exact(
                        child_idx,
                        &mut child_streams,
                        &mut buffers,
                        &mut child_positions,
                        cursor..window_end,
                    )
                    .await?;
                    let input_mask = window_acc.clone();
                    let input_rows = input_mask.true_count();
                    let start = Instant::now();
                    let result = if conjunct_demand {
                        execute_mask_with_demand(child, &input_mask, &mut exec_ctx)?
                    } else {
                        execute_mask_without_demand(child, &input_mask, &mut exec_ctx)?
                    };
                    let elapsed = start.elapsed();
                    scheduler.observe(
                        child_idx,
                        result.compute_input_rows,
                        result.compute_output_rows,
                        elapsed,
                    );
                    if tracing::enabled!(tracing::Level::DEBUG) {
                        log_conjunct_eval(
                            child_idx,
                            conjuncts.as_slice(),
                            debug_label.as_deref(),
                            &window,
                            &window_coord,
                            input_rows,
                            result.mask.true_count(),
                            result.compute_input_rows,
                            result.compute_output_rows,
                            elapsed,
                            &input_mask,
                            &result.mask,
                        );
                    }
                    grant_source.resolve(stage_key(child_idx), u64::try_from(window_end)?)?;
                    window_acc = result.mask;
                    demand_source.publish(window_coord.clone(), window_acc.clone())?;
                    grant_source.update_weights(scheduler.weights(conjuncts.as_slice()))?;
                }

                acc = replace_mask_range(acc, cursor..window_end, &window_acc)?;
                for start in (0..window_len).step_by(output_rows) {
                    let end = (start + output_rows).min(window_len);
                    let chunk = window_acc.slice(start..end);
                    yield BoolArray::new(chunk.to_bit_buffer(), Validity::NonNullable).into_array();
                }
                cursor = window_end;
            }
        };

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }
}

fn adaptive_conjunct_min_rows(conjuncts: &[ConjunctInfo]) -> usize {
    if conjuncts.len() <= 1 {
        return CONJUNCT_MIN_ROWS;
    }

    let expensive_count = conjuncts
        .iter()
        .filter(|info| info.cost >= 40 || info.expr.contains("like"))
        .count();
    if expensive_count >= 2 {
        64 * 1024
    } else if expensive_count == 1 {
        16 * 1024
    } else {
        CONJUNCT_MIN_ROWS
    }
}

fn stage_key(child_idx: usize) -> GrantKey {
    GrantKey::new(
        DomainId::new(0),
        SubplanId::new(u32::try_from(child_idx + 1).unwrap_or(u32::MAX)),
    )
}

fn spawn_conjunct_children(
    children: Vec<SendableArrayStream>,
    handle: Handle,
    buffer_depth: usize,
) -> (Vec<SendableArrayStream>, Vec<Task<()>>) {
    let mut streams = Vec::with_capacity(children.len());
    let mut tasks = Vec::with_capacity(children.len());
    for (child_idx, child) in children.into_iter().enumerate() {
        let dtype = child.dtype().clone();
        let (sender, receiver) = kanal::bounded_async(buffer_depth);
        tasks.push(handle.spawn(conjunct_producer_task(child_idx, child, sender)));
        streams.push(Box::pin(ArrayStreamAdapter::new(
            dtype,
            futures::stream::unfold(receiver, recv_conjunct_child),
        )) as SendableArrayStream);
    }
    (streams, tasks)
}

async fn conjunct_producer_task(
    child_idx: usize,
    mut source: SendableArrayStream,
    sender: kanal::AsyncSender<VortexResult<ArrayRef>>,
) {
    while let Some(item) = source.next().await {
        if sender.send(item).await.is_err() {
            return;
        }
    }
    tracing::trace!(
        target: "vortex_layout::v2::flow",
        child_idx,
        "conjunct producer eof"
    );
}

async fn recv_conjunct_child(
    recv: AsyncReceiver<VortexResult<ArrayRef>>,
) -> Option<(
    VortexResult<ArrayRef>,
    AsyncReceiver<VortexResult<ArrayRef>>,
)> {
    match recv.recv().await {
        Ok(item) => Some((item, recv)),
        Err(_) => None,
    }
}

async fn next_child_chunk(
    child_idx: usize,
    child_streams: &mut [SendableArrayStream],
    buffers: &mut [Option<ArrayRef>],
) -> VortexResult<ArrayRef> {
    if let Some(buffered) = buffers[child_idx].take() {
        return Ok(buffered);
    }
    child_streams[child_idx]
        .next()
        .await
        .ok_or_else(|| vortex_error::vortex_err!("conjunct child {child_idx} ended early"))?
}

async fn next_child_window_exact(
    child_idx: usize,
    child_streams: &mut [SendableArrayStream],
    buffers: &mut [Option<ArrayRef>],
    child_positions: &mut [usize],
    target: Range<usize>,
) -> VortexResult<ArrayRef> {
    let position = &mut child_positions[child_idx];
    if target.start < *position {
        vortex_bail!(
            "conjunct child {child_idx} has already advanced to {}, past requested window {:?}",
            *position,
            target
        );
    }

    while *position < target.start {
        let chunk = next_child_chunk(child_idx, child_streams, buffers).await?;
        let skip = (target.start - *position).min(chunk.len());
        *position += skip;
        if skip < chunk.len() {
            buffers[child_idx] = Some(chunk.slice(skip..chunk.len())?);
        }
    }

    let len = target.end - target.start;
    let mut chunks = Vec::new();
    let mut collected = 0usize;
    while collected < len {
        let chunk = next_child_chunk(child_idx, child_streams, buffers).await?;
        let take = (len - collected).min(chunk.len());
        let head = if take == chunk.len() {
            chunk
        } else {
            let head = chunk.slice(0..take)?;
            buffers[child_idx] = Some(chunk.slice(take..chunk.len())?);
            head
        };
        *position += take;
        collected += take;
        chunks.push(head);
    }

    if chunks.len() == 1 {
        return Ok(chunks.remove(0));
    }

    let dtype = chunks
        .first()
        .map(|chunk| chunk.dtype().clone())
        .ok_or_else(|| vortex_error::vortex_err!("empty conjunct child window"))?;
    Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
}

fn replace_mask_range(
    mut base: Mask,
    range: Range<usize>,
    replacement: &Mask,
) -> VortexResult<Mask> {
    if range.end > base.len() || replacement.len() != range.end - range.start {
        vortex_bail!(
            "invalid mask replacement range {range:?} for base len {} and replacement len {}",
            base.len(),
            replacement.len()
        );
    }
    let mut bits = BitBufferMut::with_capacity(base.len());
    for idx in 0..base.len() {
        let value = if range.contains(&idx) {
            replacement.value(idx - range.start)
        } else {
            base.value(idx)
        };
        bits.append(value);
    }
    base = Mask::from_buffer(bits.freeze());
    Ok(base)
}

struct ConjunctGrantSource {
    total_rows: u64,
    keys: std::collections::BTreeMap<GrantKey, usize>,
    lead_rows: f64,
    state: parking_lot::Mutex<ConjunctGrantState>,
}

struct ConjunctGrantState {
    weights: Vec<f64>,
    granted: Vec<u64>,
    resolved: Vec<u64>,
    waiters: Vec<Waker>,
}

impl ConjunctGrantSource {
    fn new(total_rows: u64, weights: Vec<f64>, lead_rows: u64) -> Self {
        let keys = (0..weights.len())
            .map(|idx| (stage_key(idx), idx))
            .collect::<std::collections::BTreeMap<_, _>>();
        Self {
            total_rows,
            keys,
            lead_rows: lead_rows.max(1) as f64,
            state: parking_lot::Mutex::new(ConjunctGrantState {
                granted: vec![0; weights.len()],
                resolved: vec![0; weights.len()],
                weights,
                waiters: Vec::new(),
            }),
        }
    }

    fn resolve(&self, key: GrantKey, frontier: u64) -> VortexResult<()> {
        if frontier > self.total_rows {
            vortex_bail!(
                "conjunct resolved frontier {frontier} exceeds total rows {}",
                self.total_rows
            );
        }
        let child_idx = self.child_idx(key)?;
        let waiters = {
            let mut state = self.state.lock();
            if state.resolved[child_idx] < frontier {
                state.resolved[child_idx] = frontier;
                std::mem::take(&mut state.waiters)
            } else {
                Vec::new()
            }
        };
        wake_all(waiters);
        Ok(())
    }

    fn update_weights(&self, weights: Vec<f64>) -> VortexResult<()> {
        if weights.len() != self.keys.len() {
            vortex_bail!(
                "conjunct weight count {} does not match child count {}",
                weights.len(),
                self.keys.len()
            );
        }
        let waiters = {
            let mut state = self.state.lock();
            state.weights = weights;
            std::mem::take(&mut state.waiters)
        };
        wake_all(waiters);
        Ok(())
    }

    fn child_idx(&self, key: GrantKey) -> VortexResult<usize> {
        self.keys
            .get(&key)
            .copied()
            .ok_or_else(|| vortex_error::vortex_err!("unknown conjunct grant key {key:?}"))
    }

    fn try_grant(
        &self,
        request: OutputGrantRequest,
        waiter: Option<&Waker>,
    ) -> VortexResult<Option<OutputGrant>> {
        let child_idx = self.child_idx(request.key())?;
        let mut waiters = Vec::new();
        let result = {
            let mut state = self.state.lock();
            let allowed_end = self.allowed_end(&state, child_idx);
            if request.target().start >= allowed_end {
                if let Some(waiter) = waiter {
                    state.waiters.push(waiter.clone());
                }
                return Ok(None);
            }

            let end = request.target().end.min(allowed_end);
            let rows = end - request.target().start;
            if state.granted[child_idx] < end {
                state.granted[child_idx] = end;
                waiters = std::mem::take(&mut state.waiters);
            }
            OutputGrant::new(
                request.key(),
                request.target().start..end,
                request.estimate().scale_to_rows(rows),
                allowed_end,
                OutputGrantReason::Granted,
            )
        };
        wake_all(waiters);
        Ok(Some(result))
    }

    fn allowed_end(&self, state: &ConjunctGrantState, child_idx: usize) -> u64 {
        if state
            .resolved
            .iter()
            .all(|&resolved| resolved >= self.total_rows)
        {
            return self.total_rows;
        }

        let min_weighted_outstanding = state
            .granted
            .iter()
            .zip(&state.resolved)
            .zip(&state.weights)
            .map(|((&granted, &resolved), &weight)| {
                granted.saturating_sub(resolved) as f64 / weight.max(1.0)
            })
            .fold(f64::INFINITY, f64::min);
        let weight = state.weights[child_idx].max(1.0);
        let allowed = state.resolved[child_idx]
            .saturating_add(((min_weighted_outstanding + self.lead_rows) * weight).ceil() as u64);
        allowed.min(self.total_rows)
    }
}

impl FrontierSource for ConjunctGrantSource {
    fn grant_now(&self, request: OutputGrantRequest) -> VortexResult<OutputGrant> {
        if let Some(grant) = self.try_grant(request.clone(), None)? {
            return Ok(grant);
        }
        let allowed_end = {
            let state = self.state.lock();
            self.allowed_end(&state, self.child_idx(request.key())?)
        };
        Ok(OutputGrant::new(
            request.key(),
            request.target().start..request.target().start,
            OutputEstimate::new(0, 0),
            allowed_end,
            OutputGrantReason::BlockedAtFrontier,
        ))
    }

    fn poll_grant(
        &self,
        request: &OutputGrantRequest,
        cx: &mut Context<'_>,
    ) -> Poll<VortexResult<OutputGrant>> {
        match self.try_grant(request.clone(), Some(cx.waker())) {
            Ok(Some(grant)) => Poll::Ready(Ok(grant)),
            Ok(None) => Poll::Pending,
            Err(err) => Poll::Ready(Err(err)),
        }
    }
}

fn wake_all(waiters: Vec<Waker>) {
    for waiter in waiters {
        waiter.wake();
    }
}

struct ConjunctRuntimeDemand {
    state: parking_lot::Mutex<ConjunctRuntimeDemandState>,
}

struct ConjunctRuntimeDemandState {
    demand: OrdinalDemand,
    waiters: Vec<Waker>,
}

impl ConjunctRuntimeDemand {
    fn new(total_rows: u64, domain: DomainId) -> Self {
        Self {
            state: parking_lot::Mutex::new(ConjunctRuntimeDemandState {
                demand: OrdinalDemand::new(domain, total_rows),
                waiters: Vec::new(),
            }),
        }
    }

    fn publish(&self, range: Range<u64>, mask: Mask) -> VortexResult<()> {
        let waiters = {
            let mut state = self.state.lock();
            state.demand.publish(range, mask)?;
            std::mem::take(&mut state.waiters)
        };
        for waiter in waiters {
            waiter.wake();
        }
        Ok(())
    }
}

impl Resource for ConjunctRuntimeDemand {
    fn version(&self) -> u64 {
        self.state.lock().demand.version()
    }

    fn ensure_ready(&self) -> BoxFuture<'_, VortexResult<()>> {
        async move { Ok(()) }.boxed()
    }
}

impl DemandSource for ConjunctRuntimeDemand {
    fn mask_for(&self, range: Range<u64>) -> BoxFuture<'_, VortexResult<Mask>> {
        async move {
            // Unknown rows remain demanded. That lets weighted
            // conjunct streams speculate up to their grant budget
            // without inserting a storage round trip at every mask
            // boundary; once a prior conjunct publishes a mask the
            // resource version changes and later pulls see the
            // narrower demand.
            self.state.lock().demand.mask_for(&range)
        }
        .boxed()
    }
}

struct DemandMaskResult {
    mask: Mask,
    compute_input_rows: usize,
    compute_output_rows: usize,
}

fn execute_mask_with_demand(
    array: ArrayRef,
    demand_mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DemandMaskResult> {
    if demand_mask.all_false() {
        return Ok(DemandMaskResult {
            mask: Mask::new_false(demand_mask.len()),
            compute_input_rows: 0,
            compute_output_rows: 0,
        });
    }

    if demand_mask.all_true() {
        let mask = array.execute::<Mask>(ctx)?;
        return Ok(DemandMaskResult {
            compute_input_rows: mask.len(),
            compute_output_rows: mask.true_count(),
            mask,
        });
    }

    let compact_array = array.filter(demand_mask.clone())?;
    let compact_mask = compact_array.execute::<Mask>(ctx)?;
    let mask = demand_mask.intersect_by_rank(&compact_mask);
    Ok(DemandMaskResult {
        compute_input_rows: compact_mask.len(),
        compute_output_rows: compact_mask.true_count(),
        mask,
    })
}

fn execute_mask_without_demand(
    array: ArrayRef,
    demand_mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DemandMaskResult> {
    let raw_mask = array.execute::<Mask>(ctx)?;
    let compute_input_rows = raw_mask.len();
    let compute_output_rows = raw_mask.true_count();
    let mask = if demand_mask.all_true() {
        raw_mask
    } else {
        demand_mask & &raw_mask
    };
    Ok(DemandMaskResult {
        mask,
        compute_input_rows,
        compute_output_rows,
    })
}

#[allow(clippy::too_many_arguments)]
fn log_conjunct_eval(
    child_idx: usize,
    conjuncts: &[ConjunctInfo],
    debug_label: Option<&str>,
    row_range: &Range<u64>,
    coord_range: &Range<u64>,
    input_rows: usize,
    output_rows: usize,
    compute_input_rows: usize,
    compute_output_rows: usize,
    elapsed: Duration,
    input_mask: &Mask,
    output_mask: &Mask,
) {
    let selectivity = if input_rows == 0 {
        0.0
    } else {
        output_rows as f64 / input_rows as f64
    };
    let input_coords = mask_coordinate_summary(input_mask, coord_range);
    let output_coords = mask_coordinate_summary(output_mask, coord_range);
    let conjunct = conjuncts.get(child_idx);
    tracing::debug!(
        child_idx,
        scan_label = debug_label.unwrap_or(""),
        original_idx = conjunct.map(|info| info.original_idx),
        cost = conjunct.map(|info| info.cost),
        conjunct = conjunct.map(|info| info.expr.as_str()),
        row_start = row_range.start,
        row_end = row_range.end,
        coord_start = coord_range.start,
        coord_end = coord_range.end,
        input_rows,
        output_rows,
        compute_input_rows,
        compute_output_rows,
        selectivity,
        elapsed_ms = elapsed.as_secs_f64() * 1000.0,
        input_coord_rows = input_coords.rows,
        input_coord_true_rows = input_coords.true_rows,
        input_coord_density = input_coords.density,
        input_coord_first_row = ?input_coords.first_row,
        input_coord_last_row = ?input_coords.last_row,
        input_coord_hash = input_coords.coord_hash,
        input_coord_sum = input_coords.coord_sum,
        input_coord_xor = input_coords.coord_xor,
        input_coord_sample = input_coords.sample_ranges.as_str(),
        output_coord_rows = output_coords.rows,
        output_coord_true_rows = output_coords.true_rows,
        output_coord_density = output_coords.density,
        output_coord_first_row = ?output_coords.first_row,
        output_coord_last_row = ?output_coords.last_row,
        output_coord_hash = output_coords.coord_hash,
        output_coord_sum = output_coords.coord_sum,
        output_coord_xor = output_coords.coord_xor,
        output_coord_sample = output_coords.sample_ranges.as_str(),
        "v2 conjunct mask evaluated"
    );
}

struct ConjunctScheduler {
    stats: Vec<ConjunctRuntimeStats>,
    dynamic_order: bool,
}

#[derive(Clone, Copy, Debug)]
struct ConjunctRuntimeStats {
    selectivity: f64,
    ns_per_input_row: f64,
    observed: bool,
}

impl ConjunctScheduler {
    fn new(conjunct_count: usize, conjuncts: &[ConjunctInfo], dynamic_order: bool) -> Self {
        let stats = (0..conjunct_count)
            .map(|idx| {
                let cost = conjuncts.get(idx).map_or(50.0, |info| f64::from(info.cost));
                ConjunctRuntimeStats {
                    selectivity: 1.0,
                    ns_per_input_row: cost,
                    observed: false,
                }
            })
            .collect();
        Self {
            stats,
            dynamic_order,
        }
    }

    fn order(&self, conjuncts: &[ConjunctInfo]) -> Vec<usize> {
        let mut order: Vec<_> = (0..self.stats.len()).collect();
        if !self.dynamic_order {
            return order;
        }
        order.sort_unstable_by(|&left, &right| {
            let left_score = self.score(left);
            let right_score = self.score(right);
            left_score
                .total_cmp(&right_score)
                .then_with(|| {
                    let left_cost = conjuncts.get(left).map_or(50, |info| info.cost);
                    let right_cost = conjuncts.get(right).map_or(50, |info| info.cost);
                    left_cost.cmp(&right_cost)
                })
                .then_with(|| left.cmp(&right))
        });
        order
    }

    fn weights(&self, _conjuncts: &[ConjunctInfo]) -> Vec<f64> {
        let scores: Vec<_> = (0..self.stats.len())
            .map(|idx| self.score(idx).max(1.0))
            .collect();
        let max_score = scores.iter().copied().fold(1.0, f64::max);
        scores
            .into_iter()
            .map(|score| (max_score / score).clamp(1.0, 8.0))
            .collect()
    }

    fn observe(
        &mut self,
        conjunct_idx: usize,
        input_rows: usize,
        output_rows: usize,
        elapsed: Duration,
    ) {
        if input_rows == 0 {
            return;
        }
        if !self.dynamic_order {
            return;
        }
        let sample_selectivity = output_rows as f64 / input_rows as f64;
        let sample_ns = elapsed.as_nanos() as f64 / input_rows as f64;
        let Some(stat) = self.stats.get_mut(conjunct_idx) else {
            return;
        };
        if stat.observed {
            const ALPHA: f64 = 0.25;
            stat.selectivity = (1.0 - ALPHA) * stat.selectivity + ALPHA * sample_selectivity;
            stat.ns_per_input_row = (1.0 - ALPHA) * stat.ns_per_input_row + ALPHA * sample_ns;
        } else {
            stat.selectivity = sample_selectivity;
            stat.ns_per_input_row = sample_ns.max(1.0);
            stat.observed = true;
        }
    }

    fn score(&self, conjunct_idx: usize) -> f64 {
        let stat = self.stats[conjunct_idx];
        // Prefer predicates that are both cheap and selective. Once a
        // conjunct has observations this is a dynamic read-ahead/order
        // proxy; the later producer-permit scheduler can use the same
        // score to assign row budgets rather than choosing a total order.
        stat.ns_per_input_row * stat.selectivity.max(0.001)
    }
}
