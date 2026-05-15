// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ZonedPruningPlan`] — the bool-returning conjunct path for
//! `ZonedLayout`.
//!
//! Holds an `Arc<ZoneMapResource>` (built and registered by
//! `ZonedLayout::plan`). At execute time it pulls the per-zone prune
//! mask from the resource (which has already been `ensure_ready`-
//! awaited by `ScanPlan`), then emits the conjunct's bool stream
//! zone by zone:
//!
//! - For pruned zones, a [`ConstantArray`] of `false` of the zone's
//!   row length — no data I/O happens for that zone at all.
//! - For kept zones, the data plan is invoked over the zone's row
//!   range and its bool chunks are forwarded.
//!
//! Cross-conjunct fanout (where this conjunct's pruning lets *other*
//! plan nodes skip work on the same rows) is implemented by the same
//! `ZoneMapResource` being registered as a [`DemandSource`] on
//! `ScanPlan`. Other plan nodes pull `RowDemand`, which intersects
//! every registered source.

use std::ops::Range;
use std::sync::Arc;

use async_stream::try_stream;
use futures::StreamExt;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::layouts::zoned::ZoneMapResource;
use crate::v2::demand::Resource;
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
    /// Shared resource that owns the falsified expression, the zones
    /// fetch pipeline, and the cached per-zone prune mask. Awaited
    /// by `ScanPlan::execute` before the body runs, so by the time
    /// this plan node executes the resource is guaranteed populated.
    resource: Arc<ZoneMapResource>,
    output_dtype: DType,
}

impl ZonedPruningPlan {
    pub fn new(data_plan: LayoutPlanRef, resource: Arc<ZoneMapResource>) -> Self {
        debug_assert!(
            matches!(data_plan.schema(), DType::Bool(_)),
            "ZonedPruningPlan: data_plan must produce a Bool stream"
        );
        let output_dtype = DType::Bool(Nullability::NonNullable);
        Self {
            data_plan,
            resource,
            output_dtype,
        }
    }

    /// The shared resource that backs this plan's pruning decisions.
    pub fn resource(&self) -> &Arc<ZoneMapResource> {
        &self.resource
    }
}

impl PartialEq for ZonedPruningPlan {
    fn eq(&self, other: &Self) -> bool {
        crate::v2::plan::plans_eq(&self.data_plan, &other.data_plan)
            // Resources participate by Arc identity — two plans built
            // in different `Scan::build` calls never share resource
            // Arcs, which is the right semantics for CSE.
            && Arc::ptr_eq(&self.resource, &other.resource)
            && self.output_dtype == other.output_dtype
    }
}

impl Eq for ZonedPruningPlan {}

impl std::hash::Hash for ZonedPruningPlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::v2::plan::hash_plan(&self.data_plan, state);
        (Arc::as_ptr(&self.resource) as *const () as usize).hash(state);
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
        // [data]. The zones source is owned by the resource.
        vec![true]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn children(&self) -> &[LayoutPlanRef] {
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
            resource: Arc::clone(&self.resource),
            output_dtype: self.output_dtype.clone(),
        }))
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        let row_count = self.resource.row_count();
        if row_range.start > row_count || row_range.end > row_count {
            vortex_bail!(
                "ZonedPruningPlan::execute row range {row_range:?} exceeds total {row_count}"
            );
        }

        let zone_len = self.resource.zone_len();
        let dtype = self.output_dtype.clone();
        let data_plan = Arc::clone(&self.data_plan);
        let ctx_for_stream = ctx.clone();
        let demand_for_stream = demand.clone();
        let resource = Arc::clone(&self.resource);

        let stream = try_stream! {
            // Lazy init: drive the resource's ensure_ready ourselves
            // (we directly read its per-zone mask, not via the demand
            // pull that would otherwise lazy-init).
            resource.ensure_ready().await?;
            let zone_prune = resource.zone_prune_mask()?;

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

                if zone_prune.value(zone_idx_usize) {
                    // Pruned — no data I/O. Emit constant false for
                    // the zone's intersection with the requested range.
                    yield ConstantArray::new(false, intersect_len).into_array();
                } else {
                    // Kept — invoke the data plan over this zone's
                    // intersection and forward its bool chunks. Demand
                    // is in our coord system; pass through unchanged
                    // (data plan shares the row coord system).
                    let intersect = intersect_start..intersect_end;
                    let mut data_stream =
                        data_plan.execute(intersect, &demand_for_stream, &ctx_for_stream)?;
                    while let Some(item) = data_stream.next().await {
                        yield item?;
                    }
                }
            }
        };

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }
}
