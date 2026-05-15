// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ZonedPruningPlan`] — the bool-returning conjunct path for
//! `ZonedLayout`.
//!
//! When `ZonedLayout::plan` is called with a filter conjunct whose
//! pruning predicate falsifies cleanly against the available stats,
//! it builds one of these. At execute time the plan reads the zones
//! table once, evaluates the predicate to a per-zone "prune" mask,
//! and emits the conjunct's bool stream zone by zone:
//!
//! - For pruned zones, a [`ConstantArray`] of `false` of the zone's
//!   row length — no data I/O happens for that zone at all.
//! - For kept zones, the data plan is invoked over the zone's row
//!   range and its bool chunks are forwarded.
//!
//! This is the "execute-time pruning" variant — pruning short-circuits
//! the conjunct's bool stream. Cross-conjunct fanout (where one
//! pruned zone should also let the unrelated projection skip its
//! reads on those rows) waits for the `RowDemand` work — see
//! `LAYOUT_PLAN.md` § ZonedLayout::plan.

use std::ops::Range;
use std::sync::Arc;

use async_stream::try_stream;
use futures::StreamExt;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::layouts::zoned::zone_map::ZoneMap;
use crate::v2::demand::RowDemand;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Per-execute pruning over a zoned layout. See module docs.
pub struct ZonedPruningPlan {
    /// Bool-returning evaluation of the filter conjunct against the
    /// data child. Invoked over kept-zone row ranges only.
    data_plan: LayoutPlanRef,
    /// Reads the zones stats table; expected to emit a single
    /// `StructArray` chunk of `nzones` rows.
    zones_plan: LayoutPlanRef,
    /// Predicate over the zones table. `true` for a zone means the
    /// original filter cannot match any row in that zone, so the
    /// zone is prunable.
    pruning_predicate: Expression,
    zone_len: u64,
    row_count: u64,
    output_dtype: DType,
}

impl ZonedPruningPlan {
    pub fn new(
        data_plan: LayoutPlanRef,
        zones_plan: LayoutPlanRef,
        pruning_predicate: Expression,
        zone_len: u64,
        row_count: u64,
    ) -> Self {
        debug_assert!(zone_len > 0, "ZonedPruningPlan requires zone_len > 0");
        debug_assert!(
            matches!(data_plan.schema(), DType::Bool(_)),
            "ZonedPruningPlan: data_plan must produce a Bool stream"
        );
        let output_dtype = DType::Bool(Nullability::NonNullable);
        Self {
            data_plan,
            zones_plan,
            pruning_predicate,
            zone_len,
            row_count,
            output_dtype,
        }
    }
}

impl PartialEq for ZonedPruningPlan {
    fn eq(&self, other: &Self) -> bool {
        crate::v2::plan::plans_eq(&self.data_plan, &other.data_plan)
            && crate::v2::plan::plans_eq(&self.zones_plan, &other.zones_plan)
            && self.pruning_predicate == other.pruning_predicate
            && self.zone_len == other.zone_len
            && self.row_count == other.row_count
            && self.output_dtype == other.output_dtype
    }
}

impl Eq for ZonedPruningPlan {}

impl std::hash::Hash for ZonedPruningPlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::v2::plan::hash_plan(&self.data_plan, state);
        crate::v2::plan::hash_plan(&self.zones_plan, state);
        self.pruning_predicate.hash(state);
        self.zone_len.hash(state);
        self.row_count.hash(state);
        self.output_dtype.hash(state);
    }
}

impl LayoutPlan for ZonedPruningPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        // Pruning passes through the data plan's partitioning.
        self.data_plan.partition_count()
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        self.data_plan.partition_stats(partition)
    }

    fn output_ordered(&self) -> bool {
        self.data_plan.output_ordered()
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        // [data, zones]. Data must be ordered (pruning preserves
        // order); zones order is fixed by the underlying flat read.
        vec![true, true]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true, true]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        // data_plan and zones_plan aren't contiguous; expose just the
        // data plan so the typical pushdown walker sees the bool
        // stream's source. Pushdown rules that need the zones plan
        // can downcast.
        std::slice::from_ref(&self.data_plan)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != 1 {
            vortex_bail!(
                "ZonedPruningPlan::with_new_children expected 1 child (data), got {}",
                children.len()
            );
        }
        let data_plan = children
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("ZonedPruningPlan with_new_children: empty vec"))?;
        Ok(Arc::new(Self {
            data_plan,
            zones_plan: Arc::clone(&self.zones_plan),
            pruning_predicate: self.pruning_predicate.clone(),
            zone_len: self.zone_len,
            row_count: self.row_count,
            output_dtype: self.output_dtype.clone(),
        }))
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        if row_range.start > self.row_count || row_range.end > self.row_count {
            vortex_bail!(
                "ZonedPruningPlan::execute row range {row_range:?} exceeds total {}",
                self.row_count
            );
        }

        // Zones plan operates in zone-coords (one row per zone) — an
        // unrelated row space, so pass detached.
        let zones_row_count = self
            .zones_plan
            .partition_stats(0)
            .map(|s| s.row_count())
            .unwrap_or_default();
        let zones_demand = RowDemand::detached(zones_row_count);
        let zones_stream = self
            .zones_plan
            .execute(0..zones_row_count, &zones_demand, ctx)?;

        let pruning_predicate = self.pruning_predicate.clone();
        let zone_len = self.zone_len;
        let row_count = self.row_count;
        let session = ctx.session().clone();
        let dtype = self.output_dtype.clone();
        let data_plan = Arc::clone(&self.data_plan);
        let ctx_for_stream = ctx.clone();
        let demand_for_stream = demand.clone();

        // Acquire a producer guard so consumers waiting on demand
        // know we're an active publisher; the guard is owned by the
        // async block and decrements on stream end.
        let guard = demand.producer_guard();

        let stream = try_stream! {
            let _guard = guard;
            // Materialise the zones table into one ArrayRef. For a
            // typical Flat zones layout this is a single chunk.
            let zones_array = zones_stream.read_all().await?;
            let mut ctx_exec = session.create_execution_ctx();
            let zones_struct = zones_array.execute::<StructArray>(&mut ctx_exec)?;

            // Compute the per-zone prune mask. SAFETY: the zones
            // layout invariant guarantees the struct's schema matches
            // the present stats.
            let zone_map = unsafe {
                ZoneMap::new_unchecked(zones_struct, zone_len, row_count)
            };
            let prune_mask = zone_map.prune(&pruning_predicate, &session)?;

            // Publish the pruning result to RowDemand. Pruned zones
            // → demand 0 for their row range. We build one BitBuffer
            // of `row_count` bits with bits SET for kept zones, UNSET
            // for pruned zones, then publish in one call.
            let row_count_usize = usize::try_from(row_count)?;
            let mut demand_bits = BitBufferMut::new_set(row_count_usize);
            let nzones = usize::try_from(row_count.div_ceil(zone_len))?;
            for z in 0..nzones {
                if prune_mask.value(z) {
                    let zr_start = (z as u64) * zone_len;
                    let zr_end = (zr_start + zone_len).min(row_count);
                    let s = usize::try_from(zr_start)?;
                    let e = usize::try_from(zr_end)?;
                    for i in s..e {
                        demand_bits.set_to(i, false);
                    }
                }
            }
            let publish_mask = Mask::from_buffer(demand_bits.freeze());
            demand_for_stream.publish(0..row_count, &publish_mask);

            // Iterate intersecting zones, emit per-zone bool chunks.
            let zone_start_idx = row_range.start / zone_len;
            let zone_end_idx = row_range.end.div_ceil(zone_len);
            for zone_idx in zone_start_idx..zone_end_idx {
                let zone_row_start = zone_idx.saturating_mul(zone_len);
                let zone_row_end = zone_row_start.saturating_add(zone_len).min(row_count);
                let intersect_start = row_range.start.max(zone_row_start);
                let intersect_end = row_range.end.min(zone_row_end);
                if intersect_start >= intersect_end {
                    continue;
                }
                let intersect_len = usize::try_from(intersect_end - intersect_start)?;
                let zone_idx_usize = usize::try_from(zone_idx)?;

                if prune_mask.value(zone_idx_usize) {
                    // Pruned — no data I/O. Emit constant false for
                    // the zone's intersection with the requested range.
                    yield ConstantArray::new(false, intersect_len).into_array();
                } else {
                    // Kept — invoke the data plan over this zone's
                    // intersection and forward its bool chunks. The
                    // data plan shares our row coord system (it's a
                    // pass-through layout under ZonedLayout), so hand
                    // demand through unchanged.
                    let intersect = intersect_start..intersect_end;
                    let mut data_stream = data_plan.execute(intersect, &demand_for_stream, &ctx_for_stream)?;
                    while let Some(item) = data_stream.next().await {
                        yield item?;
                    }
                }
            }
        };

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }
}
