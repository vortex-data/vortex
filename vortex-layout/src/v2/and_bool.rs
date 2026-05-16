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
use std::time::Duration;
use std::time::Instant;

use async_stream::try_stream;
use futures::StreamExt;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;

use crate::mask_debug::mask_coordinate_summary;
use crate::v2::aligned::AlignedArrayStream;
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
        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        let dtype = self.output_dtype.clone();
        let children = self.children.clone();
        let conjuncts = self.conjuncts.clone();
        let mut child_streams = Vec::with_capacity(children.len());
        for child in &children {
            child_streams.push(child.execute(row_range.clone(), demand, ctx)?);
        }
        let demand = demand.clone();
        let session = ctx.session().clone();
        let min_rows = usize_var("VORTEX_V2_CONJUNCT_MIN_ROWS").unwrap_or(CONJUNCT_MIN_ROWS);
        let aligned =
            AlignedArrayStream::new(child_streams, ctx.session().handle()).with_min_rows(min_rows);
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
