// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lowering stored layout trees into physical plans.

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::LayoutRef;
use crate::layouts::chunked::Chunked;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::dict::Dict;
use crate::layouts::dict::DictLayout;
use crate::layouts::flat::Flat;
use crate::layouts::flat::FlatLayout;
use crate::layouts::list::List;
use crate::layouts::list::ListLayout;
use crate::layouts::struct_::Struct;
use crate::layouts::struct_::StructLayout;
use crate::layouts::zoned::LegacyStats;
use crate::layouts::zoned::Zoned;
use crate::plan::physical::ConcatPlan;
use crate::plan::physical::FilterPlan;
use crate::plan::physical::LazyPlanRef;
use crate::plan::physical::LegacyStatsPlan;
use crate::plan::physical::ListPackPlan;
use crate::plan::physical::PackPlan;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanSlots;
use crate::plan::physical::SegmentScanPlan;
use crate::plan::physical::TakePlan;
use crate::plan::physical::ZonedPlan;

/// Lowers `layout` into a physical plan.
///
/// Zoned and legacy stats layouts keep their zone statistics: unoptimized they read only their
/// data, and the optimizer rewrites a statistics predicate over them into a pruning plan. Those
/// with no zones or no statistics lower to their data.
///
/// The root operator is built immediately. Each of its slots owns a hidden clone of the source
/// layout and lowers its child independently on first access.
///
/// A flat layout lowers to a [`Filter`](crate::plan::physical::Filter) over its segment scan, so
/// the scan returns only the rows it is executed with.
pub fn lower(layout: &LayoutRef) -> VortexResult<PlanRef> {
    if let Some(layout) = layout.as_opt::<Flat>() {
        return Ok(FilterPlan::new(lower_flat(layout).into_plan()).into_plan());
    }
    if let Some(layout) = layout.as_opt::<Chunked>() {
        return Ok(lower_chunked(layout)?.into_plan());
    }
    if let Some(layout) = layout.as_opt::<Struct>() {
        return Ok(lower_struct(layout)?.into_plan());
    }
    if let Some(layout) = layout.as_opt::<Dict>() {
        return Ok(lower_dict(layout)?.into_plan());
    }
    if let Some(layout) = layout.as_opt::<List>() {
        return Ok(lower_list(layout)?.into_plan());
    }
    if layout.is::<Zoned>() || layout.is::<LegacyStats>() {
        return lower_zoned(layout);
    }
    vortex_bail!(
        "No physical plan implementation for layout '{}'",
        layout.encoding_id()
    )
}

fn lower_flat(layout: &FlatLayout) -> SegmentScanPlan {
    SegmentScanPlan::new(
        layout.dtype().clone(),
        layout.row_count(),
        layout.segment_id(),
        layout.array_ctx().clone(),
        layout.array_tree().cloned(),
    )
}

fn lower_chunked(layout: &ChunkedLayout) -> VortexResult<ConcatPlan> {
    let mut row_offsets = Vec::with_capacity(layout.nchildren());
    let mut row_count = 0u64;
    for index in 0..layout.nchildren() {
        row_offsets.push(row_count);
        row_count = row_count
            .checked_add(layout.child_row_count(index))
            .ok_or_else(|| vortex_err!("Chunked row count overflow"))?;
    }
    // SAFETY: Chunked layout construction validates that every child has the parent dtype and
    // that their row counts sum to the parent; offsets were computed with checked addition above.
    Ok(unsafe {
        ConcatPlan::from_slots_unchecked(
            layout.dtype().clone(),
            layout.row_count(),
            row_offsets.into(),
            lazy_slots(&layout.to_layout()),
        )
    })
}

fn lower_struct(layout: &StructLayout) -> VortexResult<PackPlan> {
    // SAFETY: Struct layout construction validates child count and row counts. Its slots are
    // validity then one per field, which are the Pack slots, and its logical slot schema defines
    // every field dtype plus the optional non-nullable boolean validity dtype.
    Ok(unsafe {
        PackPlan::from_slots_unchecked(
            layout.struct_fields().clone(),
            layout.dtype().nullability(),
            layout.row_count(),
            lazy_slots(&layout.to_layout()),
        )
    })
}

fn lower_dict(layout: &DictLayout) -> VortexResult<TakePlan> {
    // Dict serialization stores values before codes; the plan order is deliberately codes,
    // values because that is the optimizer-facing logical shape.
    // SAFETY: Dict layout construction validates its values and codes slots. The plan reorders
    // those slots to [codes, values] while preserving the derived output dtype and row domain.
    Ok(unsafe {
        TakePlan::from_slots_unchecked(
            layout.dtype().clone(),
            layout.row_count(),
            lazy_slots_from(&layout.to_layout(), [1, 0]),
        )
    })
}

fn lower_list(layout: &ListLayout) -> VortexResult<ListPackPlan> {
    // SAFETY: List layout construction validates its element, offsets, and optional validity
    // child shapes. Its slots are `[elements, offsets, validity]`, which are the ListPack slots.
    Ok(unsafe {
        ListPackPlan::from_slots_unchecked(
            layout.dtype().clone(),
            layout.row_count(),
            lazy_slots(&layout.to_layout()),
        )
    })
}

/// One lazy plan slot per slot of `layout`, at the same position.
fn lazy_slots(layout: &LayoutRef) -> PlanSlots {
    lazy_slots_from(layout, 0..layout.nslots())
}

/// One lazy plan slot per entry of `layout_slots`: plan slot `i` lowers layout slot
/// `layout_slots[i]`.
///
/// This is where every lowering maps layout slots to plan slots. An absent layout slot is an
/// absent plan slot; presence comes from the layout's metadata, so nothing is read or lowered
/// until a plan slot is first accessed.
fn lazy_slots_from(layout: &LayoutRef, layout_slots: impl IntoIterator<Item = usize>) -> PlanSlots {
    layout_slots
        .into_iter()
        .map(|slot| {
            layout.slot_type(slot)?;
            let layout = Arc::clone(layout);
            Some(LazyPlanRef::lazy(move || {
                let child = layout
                    .slot(slot)?
                    .ok_or_else(|| vortex_err!("Layout child slot {slot} is absent"))?;
                lower(&child)
            }))
        })
        .collect()
}

fn lower_zoned(layout: &LayoutRef) -> VortexResult<PlanRef> {
    // Zoned and legacy stats layouts share a child shape: transparent data, auxiliary zones.
    let metadata = if let Some(layout) = layout.as_opt::<Zoned>() {
        layout.data()
    } else if let Some(layout) = layout.as_opt::<LegacyStats>() {
        layout.data()
    } else {
        vortex_bail!("Zoned plan requires a zoned layout")
    };
    // Without zones, or with a zone table holding no statistics, nothing can be pruned.
    let no_stats = metadata
        .stats_table_dtype()
        .as_struct_fields_opt()
        .is_none_or(|fields| fields.nfields() == 0);
    if metadata.zone_len() == 0 || no_stats {
        let data = layout
            .slot(0)?
            .ok_or_else(|| vortex_err!("Zoned layout has no data child"))?;
        return lower(&data);
    }
    let slots = lazy_slots_from(layout, [0, 1]);
    let zone_len = u64::try_from(metadata.zone_len())?;
    // A legacy table is converted to the aggregate schema when the plan is optimized.
    Ok(match metadata.legacy_stats() {
        Some(stats) => LegacyStatsPlan::from_slots(
            layout.dtype().clone(),
            layout.row_count(),
            slots,
            zone_len,
            Arc::clone(stats),
        )
        .into_plan(),
        None => ZonedPlan::from_slots(
            layout.dtype().clone(),
            layout.row_count(),
            slots,
            zone_len,
            metadata.aggregate_fns(),
        )
        .into_plan(),
    })
}
