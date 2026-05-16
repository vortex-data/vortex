// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FilterPlan`] — applies a per-row mask stream to a value stream.
//!
//! `FilterPlan(value_plan, mask_plan)` is the only node that actually
//! filters in the v2 design. At execute time it consumes value and
//! mask batches in row-aligned lockstep (via [`AlignedArrayStream`])
//! and emits filtered value batches.
//!
//! The PR4 invariant of "no plan-internal caches" still applies.
//! The default path is lockstep and cardinality-changing. The
//! `VORTEX_V2_FILTER_DOMAIN_DEMAND` experiment first materialises
//! the mask, publishes it as row-domain demand for value children,
//! and only then compacts at this filter boundary.
//!
//! See `LAYOUT_PLAN.md` § FilterPlan and its pushdown.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::task::Poll;
use std::task::Waker;
use std::time::Instant;

use async_stream::try_stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::future::poll_fn;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;

use crate::mask_debug::log_mask_batch;
use crate::v2::aligned::AlignedArrayStream;
use crate::v2::dataflow::DomainId;
use crate::v2::dataflow::LayoutLoweringCtx;
use crate::v2::dataflow::OrdinalDemand;
use crate::v2::dataflow::OutputFrontier;
use crate::v2::dataflow::PermitPolicy;
use crate::v2::dataflow::PermitReason;
use crate::v2::dataflow::WorkEstimate;
use crate::v2::demand::DemandSource;
use crate::v2::demand::Resource;
use crate::v2::demand::RowDemand;
use crate::v2::demand::RowRange;
use crate::v2::experiment::bool_var;
use crate::v2::experiment::trace_flow;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Applies `mask` to `values` per row. Output dtype matches the
/// value plan's schema; output row count is the number of `true`
/// rows in the mask, not the input row count.
pub struct FilterPlan {
    values: LayoutPlanRef,
    mask: LayoutPlanRef,
    output_dtype: DType,
}

impl FilterPlan {
    /// Construct a `FilterPlan` over `values` and `mask`. Always
    /// returns a real `FilterPlan`; use [`Self::new_or_pushdown`] to
    /// give the values plan a chance to absorb the mask first.
    pub fn new(values: LayoutPlanRef, mask: LayoutPlanRef) -> Self {
        debug_assert!(
            matches!(mask.schema(), DType::Bool(_)),
            "FilterPlan: mask plan must produce a Bool stream",
        );
        let output_dtype = values.schema().clone();
        Self {
            values,
            mask,
            output_dtype,
        }
    }

    /// Try to push `mask` into `values` via
    /// [`LayoutPlan::try_pushdown_mask`]. If the values plan absorbs
    /// it, return the rewrite directly (no `FilterPlan` wrapper). If
    /// not, fall back to wrapping with `FilterPlan::new`.
    pub fn new_or_pushdown(values: LayoutPlanRef, mask: LayoutPlanRef) -> LayoutPlanRef {
        debug_assert!(
            matches!(mask.schema(), DType::Bool(_)),
            "FilterPlan: mask plan must produce a Bool stream",
        );
        if let Some(pushed) = Arc::clone(&values).try_pushdown_mask(Arc::clone(&mask)) {
            if trace_flow() {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    values_schema = %values.schema(),
                    mask_schema = %mask.schema(),
                    "filter pushdown succeeded"
                );
            }
            return pushed;
        }
        if trace_flow() {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                values_schema = %values.schema(),
                mask_schema = %mask.schema(),
                "filter pushdown failed"
            );
        }
        Arc::new(Self::new(values, mask))
    }
}

impl PartialEq for FilterPlan {
    fn eq(&self, other: &Self) -> bool {
        crate::v2::plan::plans_eq(&self.values, &other.values)
            && crate::v2::plan::plans_eq(&self.mask, &other.mask)
            && self.output_dtype == other.output_dtype
    }
}

impl Eq for FilterPlan {}

impl std::hash::Hash for FilterPlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::v2::plan::hash_plan(&self.values, state);
        crate::v2::plan::hash_plan(&self.mask, state);
        self.output_dtype.hash(state);
    }
}

impl LayoutPlan for FilterPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        // Row-range partitioning passes through. We don't merge
        // values/mask partitions because both are derived from the
        // same Layout::plan and share the same partitioning shape.
        self.values.partition_count()
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        // The row range is in the *input* row coordinate space. The
        // actual emitted row count after filtering is data-dependent
        // and unknown without executing.
        self.values.partition_stats(partition)
    }

    fn output_ordered(&self) -> bool {
        self.values.output_ordered() && self.mask.output_ordered()
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![true, true]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true, false]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        // Children order: [values, mask].
        // Returning an empty slice here would be safe (we just won't
        // be visited by the pushdown walker), but exposing the real
        // children lets PR6 pushdown rules find them.
        // We can't return `&[values, mask]` because they're not
        // contiguous in memory — would need an owning vec on each
        // call. Skip for now; PR6 can add a `children_arc` accessor
        // if it needs them.
        &[]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if !children.is_empty() {
            vortex_bail!("FilterPlan does not yet expose its children for replacement");
        }
        Ok(self)
    }

    fn lower_to_scheduler(
        &self,
        row_range: Range<u64>,
        ctx: &mut LayoutLoweringCtx,
    ) -> VortexResult<()> {
        ctx.register_plan_node(row_range.clone(), self.schema(), 2);
        self.mask.lower_to_scheduler(row_range.clone(), ctx)?;
        self.values.lower_to_scheduler(row_range, ctx)
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        frontier: &OutputFrontier,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        if bool_var("VORTEX_V2_FILTER_DATAFLOW") {
            return self.execute_dataflow(row_range, demand, frontier, ctx);
        }
        if bool_var("VORTEX_V2_FILTER_DOMAIN_DEMAND") {
            return self.execute_domain_demand(row_range, demand, frontier, ctx);
        }
        if bool_var("VORTEX_V2_FILTER_MASK_FIRST") {
            return self.execute_mask_first(row_range, demand, frontier, ctx);
        }

        let local_start = row_range.start;
        let coord_start = demand.global_range(&row_range).start;
        let trace = trace_flow();
        if trace {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                row_start = row_range.start,
                row_end = row_range.end,
                coord_start,
                "filter execute"
            );
        }
        let values_stream = self
            .values
            .execute(row_range.clone(), demand, frontier, ctx)?;
        let mask_stream = self.mask.execute(row_range, demand, frontier, ctx)?;

        let session = ctx.session().clone();
        let dtype = self.output_dtype.clone();
        let debug_label = ctx.debug_label().map(str::to_owned);
        let mut local_cursor = local_start;
        let mut coord_cursor = coord_start;
        let aligned = AlignedArrayStream::new_labeled(
            vec![values_stream, mask_stream],
            ctx.session().handle(),
            "filter",
        );
        let mapped = aligned.map(move |result| {
            let mut arrays = result?.into_iter();
            let values = arrays
                .next()
                .vortex_expect("FilterPlan: values stream missing from aligned output");
            let mask = arrays
                .next()
                .vortex_expect("FilterPlan: mask stream missing from aligned output");
            // Convert the bool array to a `Mask` and apply. Same
            // round-trip the v1 `FlatReader::projection_evaluation`
            // does when it has a non-trivial mask.
            let mut ctx = session.create_execution_ctx();
            let mask: Mask = mask.execute::<Mask>(&mut ctx)?;
            let input_rows = mask.len() as u64;
            let local_range = local_cursor..local_cursor + input_rows;
            let coord_range = coord_cursor..coord_cursor + input_rows;
            if trace {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    row_start = local_range.start,
                    row_end = local_range.end,
                    coord_start = coord_range.start,
                    coord_end = coord_range.end,
                    input_rows,
                    true_count = mask.true_count(),
                    value_rows = values.len(),
                    "filter aligned batch"
                );
            }
            let output_mask_for_log =
                tracing::enabled!(tracing::Level::DEBUG).then(|| mask.clone());
            if mask.all_true() {
                let output_rows = values.len();
                if let Some(mask) = output_mask_for_log.as_ref() {
                    log_mask_batch(
                        "v2 filter batch projected",
                        debug_label.as_deref(),
                        &local_range,
                        &coord_range,
                        mask,
                        None,
                        Some(output_rows),
                    );
                }
                local_cursor += input_rows;
                coord_cursor += input_rows;
                Ok(values)
            } else {
                let output = values.filter(mask)?;
                if let Some(mask) = output_mask_for_log.as_ref() {
                    log_mask_batch(
                        "v2 filter batch projected",
                        debug_label.as_deref(),
                        &local_range,
                        &coord_range,
                        mask,
                        None,
                        Some(output.len()),
                    );
                }
                local_cursor += input_rows;
                coord_cursor += input_rows;
                Ok(output)
            }
        });

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, mapped)))
    }
}

impl FilterPlan {
    fn execute_dataflow(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        frontier: &OutputFrontier,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        if !bool_var("VORTEX_V2_FILTER_DATAFLOW_WINDOWED") {
            return self.execute_dataflow_long_lived(row_range, demand, frontier, ctx);
        }

        let values = Arc::clone(&self.values);
        let mut mask_stream = self
            .mask
            .execute(row_range.clone(), demand, frontier, ctx)?;
        let dtype = self.output_dtype.clone();
        let session = ctx.session().clone();
        let parent_demand = demand.clone();
        let frontier = frontier.clone();
        let ctx = ctx.clone();
        let debug_label = ctx.debug_label().map(str::to_owned);
        let trace = trace_flow();
        let published = Arc::new(DataflowMaskDemand::new(parent_demand.root_total_rows()));
        let published_source: Arc<dyn DemandSource> =
            Arc::clone(&published) as Arc<dyn DemandSource>;
        let value_demand = parent_demand.with_source(published_source);
        let min_value_rows =
            crate::v2::experiment::usize_var("VORTEX_V2_FILTER_DATAFLOW_MIN_VALUE_ROWS")
                .map(|value| u64::try_from(value).unwrap_or(u64::MAX))
                .unwrap_or(64 * 1024);
        let producer_rows =
            crate::v2::experiment::usize_var("VORTEX_V2_FILTER_DATAFLOW_PRODUCER_ROWS")
                .map(|value| u64::try_from(value).unwrap_or(u64::MAX))
                .unwrap_or(64 * 1024);
        let speculative_rows =
            crate::v2::experiment::usize_var("VORTEX_V2_FILTER_DATAFLOW_SPECULATIVE_ROWS")
                .map(|value| u64::try_from(value).unwrap_or(u64::MAX))
                .unwrap_or(8 * 1024);
        let policy = PermitPolicy::new(producer_rows, speculative_rows, 1.0);
        // This first integrated prototype is intentionally conservative:
        // unknown value work waits for mask coverage unless the isolated
        // policy is changed later. Speculation needs a value buffer so it
        // can execute before the final mask is known but emit only after
        // coverage catches up.
        let estimate = WorkEstimate::new(1.0, 100.0, 0.95, 0.9);

        let stream = try_stream! {
            let mut local_demand = OrdinalDemand::new(DomainId::new(0), row_range.end);
            let mut pending_masks = VecDeque::new();
            let mut mask_cursor = row_range.start;
            let mut value_cursor = row_range.start;
            let mut mask_done = false;

            loop {
                if value_cursor >= row_range.end {
                    break;
                }

                if let Some(output) = maybe_run_dataflow_value_window(
                    &values,
                    &value_demand,
                    &frontier,
                    &ctx,
                    &mut pending_masks,
                    &local_demand,
                    &policy,
                    estimate,
                    value_cursor..row_range.end,
                    min_value_rows,
                    mask_done,
                    debug_label.as_deref(),
                    trace,
                )
                .await? {
                    value_cursor = output.next_cursor;
                    if let Some(array) = output.array {
                        yield array;
                    }
                    continue;
                }

                if mask_done {
                    Err(vortex_error::vortex_err!(
                        "dataflow FilterPlan stopped with uncovered value range {}..{}",
                        value_cursor,
                        row_range.end
                    ))?;
                }

                let mask_start = Instant::now();
                let Some(mask_array) = mask_stream.next().await else {
                    mask_done = true;
                    if mask_cursor != row_range.end {
                        Err(vortex_error::vortex_err!(
                            "dataflow FilterPlan mask stream produced {} rows, expected {}",
                            mask_cursor - row_range.start,
                            row_range.end - row_range.start
                        ))?;
                    }
                    continue;
                };
                let mask_array = mask_array?;
                if mask_array.is_empty() {
                    continue;
                }

                let mut exec_ctx = session.create_execution_ctx();
                let mask: Mask = mask_array.execute::<Mask>(&mut exec_ctx)?;
                let mask_elapsed = mask_start.elapsed();
                let input_rows = u64::try_from(mask.len())?;
                let local_range = mask_cursor..mask_cursor + input_rows;
                if local_range.end > row_range.end {
                    Err(vortex_error::vortex_err!(
                        "dataflow FilterPlan mask stream produced rows past requested range: {} > {}",
                        local_range.end,
                        row_range.end
                    ))?;
                }
                let coord_range = parent_demand.global_range(&local_range);
                mask_cursor = local_range.end;

                if trace {
                    tracing::debug!(
                        target: "vortex_layout::v2::flow",
                        row_start = local_range.start,
                        row_end = local_range.end,
                        coord_start = coord_range.start,
                        coord_end = coord_range.end,
                        true_count = mask.true_count(),
                        mask_elapsed_ms = mask_elapsed.as_secs_f64() * 1000.0,
                        "filter dataflow mask published"
                    );
                }
                published.publish(coord_range.clone(), mask.clone())?;
                local_demand.publish(local_range.clone(), mask.clone())?;
                pending_masks.push_back(FilterMaskBatch {
                    local_range,
                    coord_range,
                    mask,
                    mask_elapsed,
                });
            }
        };

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }

    fn execute_dataflow_long_lived(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        frontier: &OutputFrontier,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        let dtype = self.output_dtype.clone();
        let published = Arc::new(DataflowMaskDemand::new(demand.root_total_rows()));
        let published_source: Arc<dyn DemandSource> =
            Arc::clone(&published) as Arc<dyn DemandSource>;
        let value_demand = demand.with_source(published_source);
        let mut values_stream =
            self.values
                .execute(row_range.clone(), &value_demand, frontier, ctx)?;

        let mask_stream = self
            .mask
            .execute(row_range.clone(), demand, frontier, ctx)?;
        let mask_task = ctx.session().handle().spawn(publish_dataflow_masks(
            mask_stream,
            Arc::clone(&published),
            demand.clone(),
            row_range.clone(),
            ctx.session().clone(),
            ctx.debug_label().map(str::to_owned),
            trace_flow(),
        ));

        let parent_demand = demand.clone();
        let debug_label = ctx.debug_label().map(str::to_owned);
        let trace = trace_flow();
        let stream = try_stream! {
            let _mask_task = mask_task;
            let mut local_cursor = row_range.start;

            while let Some(values) = values_stream.next().await {
                let values = values?;
                if values.is_empty() {
                    continue;
                }

                let input_rows = u64::try_from(values.len())?;
                let local_range = local_cursor..local_cursor + input_rows;
                if local_range.end > row_range.end {
                    Err(vortex_err!(
                        "dataflow long-lived FilterPlan values produced rows past requested range: {} > {}",
                        local_range.end,
                        row_range.end
                    ))?;
                }
                let coord_range = parent_demand.global_range(&local_range);
                let mask_start = Instant::now();
                let mask = published.mask_for_covered(coord_range.clone()).await?;
                let mask_elapsed = mask_start.elapsed();
                let output_mask_for_log =
                    tracing::enabled!(tracing::Level::DEBUG).then(|| mask.clone());

                let output = if mask.all_true() {
                    values
                } else {
                    values.filter(mask.clone())?
                };
                let output_rows = output.len();
                if let Some(mask) = output_mask_for_log.as_ref() {
                    log_mask_batch(
                        "v2 filter dataflow long-lived projected",
                        debug_label.as_deref(),
                        &local_range,
                        &coord_range,
                        mask,
                        Some(mask_elapsed),
                        Some(output_rows),
                    );
                }
                if trace {
                    tracing::debug!(
                        target: "vortex_layout::v2::flow",
                        row_start = local_range.start,
                        row_end = local_range.end,
                        coord_start = coord_range.start,
                        coord_end = coord_range.end,
                        true_count = mask.true_count(),
                        value_rows = input_rows,
                        output_rows,
                        mask_wait_ms = mask_elapsed.as_secs_f64() * 1000.0,
                        "filter dataflow long-lived value emitted"
                    );
                }

                local_cursor = local_range.end;
                yield output;
            }

            if local_cursor != row_range.end {
                Err(vortex_err!(
                    "dataflow long-lived FilterPlan values produced {} rows, expected {}",
                    local_cursor - row_range.start,
                    row_range.end - row_range.start
                ))?;
            }
        };

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }

    fn execute_domain_demand(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        frontier: &OutputFrontier,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        let values = Arc::clone(&self.values);
        let mask_stream = self
            .mask
            .execute(row_range.clone(), demand, frontier, ctx)?;
        let dtype = self.output_dtype.clone();
        let session = ctx.session().clone();
        let demand = demand.clone();
        let frontier = frontier.clone();
        let ctx = ctx.clone();
        let debug_label = ctx.debug_label().map(str::to_owned);
        let coord_range = demand.global_range(&row_range);
        let local_range = row_range;

        let stream = try_stream! {
            let mask_array = mask_stream.read_all().await?;
            let mut exec_ctx = session.create_execution_ctx();
            let mask: Mask = mask_array.execute::<Mask>(&mut exec_ctx)?;
            let output_mask_for_log =
                tracing::enabled!(tracing::Level::DEBUG).then(|| mask.clone());

            if mask.all_false() {
                if let Some(mask) = output_mask_for_log.as_ref() {
                    log_mask_batch(
                        "v2 filter domain-demand projected",
                        debug_label.as_deref(),
                        &local_range,
                        &coord_range,
                        mask,
                        None,
                        Some(0),
                    );
                }
                return;
            }

            let published = Arc::new(PublishedMaskDemand::new(demand.root_total_rows()));
            published.publish(coord_range.clone(), mask.clone())?;
            let published_source: Arc<dyn DemandSource> = published;
            let demand = demand.with_source(published_source);
            let values_stream = values.execute(local_range.clone(), &demand, &frontier, &ctx)?;
            let values = values_stream.read_all().await?;
            let output = if mask.all_true() {
                values
            } else {
                values.filter(mask.clone())?
            };

            if let Some(mask) = output_mask_for_log.as_ref() {
                log_mask_batch(
                    "v2 filter domain-demand projected",
                    debug_label.as_deref(),
                    &local_range,
                    &coord_range,
                    mask,
                    None,
                    Some(output.len()),
                );
            }
            yield output;
        };

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }

    fn execute_mask_first(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        frontier: &OutputFrontier,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        let values = Arc::clone(&self.values);
        let mut mask_stream = self
            .mask
            .execute(row_range.clone(), demand, frontier, ctx)?;
        let dtype = self.output_dtype.clone();
        let session = ctx.session().clone();
        let demand = demand.clone();
        let frontier = frontier.clone();
        let ctx = ctx.clone();
        let debug_label = ctx.debug_label().map(str::to_owned);
        let mut local_cursor = row_range.start;
        let mut coord_cursor = demand.global_range(&row_range).start;

        let stream = try_stream! {
            while let Some(mask_array) = mask_stream.next().await {
                let mask_array = mask_array?;
                if mask_array.is_empty() {
                    continue;
                }

                let mut exec_ctx = session.create_execution_ctx();
                let mask: Mask = mask_array.execute::<Mask>(&mut exec_ctx)?;
                let input_rows = mask.len() as u64;
                let local_range = local_cursor..local_cursor + input_rows;
                let coord_range = coord_cursor..coord_cursor + input_rows;
                let output_mask_for_log =
                    tracing::enabled!(tracing::Level::DEBUG).then(|| mask.clone());

                if mask.all_false() {
                    if let Some(mask) = output_mask_for_log.as_ref() {
                        log_mask_batch(
                            "v2 filter batch projected",
                            debug_label.as_deref(),
                            &local_range,
                            &coord_range,
                            mask,
                            None,
                            Some(0),
                        );
                    }
                    local_cursor += input_rows;
                    coord_cursor += input_rows;
                    continue;
                }

                let values_stream = values.execute(local_range.clone(), &demand, &frontier, &ctx)?;
                let values = values_stream.read_all().await?;
                let output = if mask.all_true() {
                    values
                } else {
                    values.filter(mask.clone())?
                };

                if let Some(mask) = output_mask_for_log.as_ref() {
                    log_mask_batch(
                        "v2 filter batch projected",
                        debug_label.as_deref(),
                        &local_range,
                        &coord_range,
                        mask,
                        None,
                        Some(output.len()),
                    );
                }
                local_cursor += input_rows;
                coord_cursor += input_rows;
                yield output;
            }
        };

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }
}

struct DataflowValueOutput {
    next_cursor: u64,
    array: Option<ArrayRef>,
}

#[allow(clippy::too_many_arguments)]
async fn maybe_run_dataflow_value_window(
    values: &LayoutPlanRef,
    demand: &RowDemand,
    frontier: &OutputFrontier,

    ctx: &ScanCtx,
    pending_masks: &mut VecDeque<FilterMaskBatch>,
    local_demand: &OrdinalDemand,
    policy: &PermitPolicy,
    estimate: WorkEstimate,
    target: RowRange,
    min_value_rows: u64,
    mask_done: bool,
    debug_label: Option<&str>,
    trace: bool,
) -> VortexResult<Option<DataflowValueOutput>> {
    if target.start >= target.end {
        return Ok(Some(DataflowValueOutput {
            next_cursor: target.start,
            array: None,
        }));
    }

    let permit = policy.value_consumer_permit(local_demand, &target, estimate)?;
    let reason = permit.reason();
    let permit_range = permit.range().clone();
    let permit_rows = permit.rows_to_poll();
    if trace {
        tracing::debug!(
            target: "vortex_layout::v2::flow",
            row_start = target.start,
            row_end = target.end,
            permit_start = permit_range.start,
            permit_end = permit_range.end,
            permit_rows,
            ?reason,
            mask_done,
            "filter dataflow permit"
        );
    }

    match reason {
        PermitReason::SkipAllFalse => {
            let mask = drain_pending_mask(pending_masks, &permit_range)?;
            log_mask_batch(
                "v2 filter dataflow skipped",
                debug_label,
                &permit_range,
                &permit_range,
                &mask,
                None,
                Some(0),
            );
            Ok(Some(DataflowValueOutput {
                next_cursor: permit_range.end,
                array: None,
            }))
        }
        PermitReason::ProceedWithKnownDemand => {
            if !mask_done && permit_rows < min_value_rows {
                return Ok(None);
            }
            let value_start = Instant::now();
            let mask = drain_pending_mask(pending_masks, &permit_range)?;
            let values_stream = values.execute(permit_range.clone(), demand, frontier, ctx)?;
            let values = values_stream.read_all().await?;
            let output = if mask.all_true() {
                values
            } else {
                values.filter(mask.clone())?
            };
            let output_rows = output.len();
            log_mask_batch(
                "v2 filter dataflow projected",
                debug_label,
                &permit_range,
                &permit_range,
                &mask,
                Some(value_start.elapsed()),
                Some(output_rows),
            );
            if trace {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    row_start = permit_range.start,
                    row_end = permit_range.end,
                    true_count = mask.true_count(),
                    output_rows,
                    value_elapsed_ms = value_start.elapsed().as_secs_f64() * 1000.0,
                    "filter dataflow value emitted"
                );
            }
            Ok(Some(DataflowValueOutput {
                next_cursor: permit_range.end,
                array: Some(output),
            }))
        }
        PermitReason::AlreadyCovered if target.start >= target.end => {
            Ok(Some(DataflowValueOutput {
                next_cursor: target.start,
                array: None,
            }))
        }
        PermitReason::Speculate => {
            // The policy supports speculation, but this FilterPlan
            // prototype does not yet buffer speculative values until
            // the mask frontier catches up. Treat it as "wait" for
            // correctness and to keep the first experiment simple.
            Ok(None)
        }
        PermitReason::WaitForDemand
        | PermitReason::DriveDemandProducer
        | PermitReason::AlreadyCovered => Ok(None),
    }
}

fn drain_pending_mask(
    pending_masks: &mut VecDeque<FilterMaskBatch>,
    target: &RowRange,
) -> VortexResult<Mask> {
    let mut cursor = target.start;
    let mut masks = Vec::new();
    while cursor < target.end {
        let Some(batch) = pending_masks.pop_front() else {
            vortex_bail!("missing pending mask for range {target:?} at row {cursor}");
        };
        if batch.local_range.start != cursor {
            vortex_bail!(
                "pending mask range {:?} did not start at expected row {cursor}",
                batch.local_range
            );
        }
        if batch.local_range.end <= target.end {
            cursor = batch.local_range.end;
            masks.push(batch.mask);
            continue;
        }

        let take_len = usize::try_from(target.end - batch.local_range.start)?;
        let rest_start = batch.local_range.start + u64::try_from(take_len)?;
        let left = batch.mask.slice(0..take_len);
        let right = batch.mask.slice(take_len..batch.mask.len());
        let rest_coord_start = batch.coord_range.start + u64::try_from(take_len)?;
        let rest = FilterMaskBatch {
            local_range: rest_start..batch.local_range.end,
            coord_range: rest_coord_start..batch.coord_range.end,
            mask: right,
            mask_elapsed: batch.mask_elapsed,
        };
        pending_masks.push_front(rest);
        cursor = target.end;
        masks.push(left);
    }

    Mask::concat(masks.iter())
}

struct FilterMaskBatch {
    local_range: RowRange,
    coord_range: RowRange,
    mask: Mask,
    mask_elapsed: std::time::Duration,
}

async fn publish_dataflow_masks(
    mut mask_stream: SendableArrayStream,
    published: Arc<DataflowMaskDemand>,
    parent_demand: RowDemand,
    row_range: RowRange,
    session: vortex_session::VortexSession,
    debug_label: Option<String>,
    trace: bool,
) {
    let result: VortexResult<()> = async {
        let mut mask_cursor = row_range.start;
        while let Some(mask_array) = mask_stream.next().await {
            let mask_start = Instant::now();
            let mask_array = mask_array?;
            if mask_array.is_empty() {
                continue;
            }

            let mut exec_ctx = session.create_execution_ctx();
            let mask: Mask = mask_array.execute::<Mask>(&mut exec_ctx)?;
            let mask_elapsed = mask_start.elapsed();
            let input_rows = u64::try_from(mask.len())?;
            let local_range = mask_cursor..mask_cursor + input_rows;
            if local_range.end > row_range.end {
                vortex_bail!(
                    "dataflow mask publisher produced rows past requested range: {} > {}",
                    local_range.end,
                    row_range.end
                );
            }
            let coord_range = parent_demand.global_range(&local_range);
            mask_cursor = local_range.end;

            let output_mask_for_log =
                tracing::enabled!(tracing::Level::DEBUG).then(|| mask.clone());
            if let Some(mask) = output_mask_for_log.as_ref() {
                log_mask_batch(
                    "v2 filter dataflow long-lived mask published",
                    debug_label.as_deref(),
                    &local_range,
                    &coord_range,
                    mask,
                    Some(mask_elapsed),
                    None,
                );
            }
            if trace {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    row_start = local_range.start,
                    row_end = local_range.end,
                    coord_start = coord_range.start,
                    coord_end = coord_range.end,
                    true_count = mask.true_count(),
                    mask_elapsed_ms = mask_elapsed.as_secs_f64() * 1000.0,
                    "filter dataflow long-lived mask published"
                );
            }
            published.publish(coord_range, mask)?;
        }

        if mask_cursor != row_range.end {
            vortex_bail!(
                "dataflow mask publisher produced {} rows, expected {}",
                mask_cursor - row_range.start,
                row_range.end - row_range.start
            );
        }

        Ok(())
    }
    .await;

    published.finish(result);
}

struct PublishedMaskDemand {
    total_rows: u64,
    state: parking_lot::Mutex<PublishedMaskState>,
}

struct PublishedMaskState {
    version: u64,
    ranges: Vec<(RowRange, Mask)>,
}

impl PublishedMaskDemand {
    fn new(total_rows: u64) -> Self {
        Self {
            total_rows,
            state: parking_lot::Mutex::new(PublishedMaskState {
                version: 0,
                ranges: Vec::new(),
            }),
        }
    }

    fn publish(&self, range: RowRange, mask: Mask) -> VortexResult<()> {
        if range.end > self.total_rows {
            vortex_bail!(
                "published demand range {range:?} exceeds root row count {}",
                self.total_rows
            );
        }
        let expected_len = usize::try_from(range.end - range.start)?;
        if mask.len() != expected_len {
            vortex_bail!(
                "published demand mask length {} did not match range {range:?}",
                mask.len()
            );
        }
        let mut state = self.state.lock();
        state.ranges.push((range, mask));
        state.version += 1;
        Ok(())
    }
}

impl Resource for PublishedMaskDemand {
    fn version(&self) -> u64 {
        self.state.lock().version
    }

    fn ensure_ready(&self) -> BoxFuture<'_, VortexResult<()>> {
        Box::pin(async { Ok(()) })
    }
}

impl DemandSource for PublishedMaskDemand {
    fn mask_for(&self, range: RowRange) -> BoxFuture<'_, VortexResult<Mask>> {
        Box::pin(async move {
            self.ensure_ready().await?;
            if range.end > self.total_rows {
                vortex_bail!(
                    "requested demand range {range:?} exceeds root row count {}",
                    self.total_rows
                );
            }
            let len = usize::try_from(range.end - range.start)?;
            let mut bits = BitBufferMut::new_set(len);
            let state = self.state.lock();
            for (published_range, mask) in &state.ranges {
                let start = range.start.max(published_range.start);
                let end = range.end.min(published_range.end);
                if start >= end {
                    continue;
                }
                let output_start = usize::try_from(start - range.start)?;
                let mask_start = usize::try_from(start - published_range.start)?;
                let overlap_len = usize::try_from(end - start)?;
                for idx in 0..overlap_len {
                    bits.set_to(output_start + idx, mask.value(mask_start + idx));
                }
            }
            Ok(Mask::from_buffer(bits.freeze()))
        })
    }
}

struct DataflowMaskDemand {
    state: parking_lot::Mutex<DataflowMaskDemandState>,
}

struct DataflowMaskDemandState {
    demand: OrdinalDemand,
    done: bool,
    error: Option<String>,
    waiters: Vec<Waker>,
}

impl DataflowMaskDemand {
    fn new(total_rows: u64) -> Self {
        Self {
            state: parking_lot::Mutex::new(DataflowMaskDemandState {
                demand: OrdinalDemand::new(DomainId::new(0), total_rows),
                done: false,
                error: None,
                waiters: Vec::new(),
            }),
        }
    }

    fn publish(&self, range: RowRange, mask: Mask) -> VortexResult<()> {
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

    fn finish(&self, result: VortexResult<()>) {
        let waiters = {
            let mut state = self.state.lock();
            state.done = true;
            if let Err(err) = result {
                state.error = Some(err.to_string());
            }
            std::mem::take(&mut state.waiters)
        };
        for waiter in waiters {
            waiter.wake();
        }
    }

    async fn wait_until_covered(&self, range: RowRange) -> VortexResult<()> {
        poll_fn(|cx| {
            let mut state = self.state.lock();
            if let Some(error) = state.error.as_deref() {
                return Poll::Ready(Err(vortex_err!("dataflow mask publisher failed: {error}")));
            }
            match state.demand.coverage(&range) {
                Ok(coverage) if coverage.is_complete() => Poll::Ready(Ok(())),
                Ok(_) if state.done => Poll::Ready(Err(vortex_err!(
                    "dataflow mask publisher finished before covering range {range:?}"
                ))),
                Ok(_) => {
                    state.waiters.push(cx.waker().clone());
                    Poll::Pending
                }
                Err(err) => Poll::Ready(Err(err)),
            }
        })
        .await
    }

    async fn mask_for_covered(&self, range: RowRange) -> VortexResult<Mask> {
        self.wait_until_covered(range.clone()).await?;
        self.state.lock().demand.mask_for(&range)
    }
}

impl Resource for DataflowMaskDemand {
    fn version(&self) -> u64 {
        self.state.lock().demand.version()
    }

    fn ensure_ready(&self) -> BoxFuture<'_, VortexResult<()>> {
        Box::pin(async { Ok(()) })
    }
}

impl DemandSource for DataflowMaskDemand {
    fn mask_for(&self, range: RowRange) -> BoxFuture<'_, VortexResult<Mask>> {
        Box::pin(async move { self.mask_for_covered(range).await })
    }
}
