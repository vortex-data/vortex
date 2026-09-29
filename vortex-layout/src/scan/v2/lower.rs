// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lowers a file's layout tree into the physical plan the V2 executor runs.
//!
//! A copy of the test-only [`plan::lower`](crate::plan::lower) with two modes for zoned layouts:
//! data plans lower them to their data child, and pruning plans keep them as a
//! [`ZonedPlan`](crate::plan::ZonedPlan) so pruning proofs can reach the zone statistics.

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
use crate::layouts::list::ELEMENTS_CHILD_INDEX;
use crate::layouts::list::List;
use crate::layouts::list::ListLayout;
use crate::layouts::list::OFFSETS_CHILD_INDEX;
use crate::layouts::list::VALIDITY_CHILD_INDEX;
use crate::layouts::struct_::Struct;
use crate::layouts::struct_::StructLayout;
use crate::layouts::zoned::LegacyStats;
use crate::layouts::zoned::Zoned;
use crate::plan::ConcatPlan;
use crate::plan::FilterPlan;
use crate::plan::ListPackPlan;
use crate::plan::PackPlan;
use crate::plan::PlanChildren;
use crate::plan::PlanRef;
use crate::plan::SegmentScanPlan;
use crate::plan::SharePlan;
use crate::plan::TakePlan;
use crate::plan::ZonedPlan;

/// Lowers `layout` into a physical plan.
///
/// The root operator is built immediately. Its child container owns a hidden clone of the source
/// layout and lowers each child independently on first access.
///
/// A flat layout lowers to a [`Filter`](crate::plan::Filter) over its segment scan, so the scan
/// returns only the rows it is executed with.
pub(crate) fn lower(layout: &LayoutRef) -> VortexResult<PlanRef> {
    lower_with(layout, Zones::Drop)
}

/// Lowers `layout` like [`lower`], but keeps each zoned layout as a [`ZonedPlan`] over its data
/// and zone statistics, so pruning proofs can be pushed down to the zones.
pub(crate) fn lower_with_zones(layout: &LayoutRef) -> VortexResult<PlanRef> {
    lower_with(layout, Zones::Keep)
}

/// What a zoned layout lowers to.
#[derive(Clone, Copy)]
enum Zones {
    /// Its data child: data plans do not read zone statistics.
    Drop,
    /// A [`ZonedPlan`] over its data and zones.
    Keep,
}

fn lower_with(layout: &LayoutRef, zones: Zones) -> VortexResult<PlanRef> {
    if layout.is::<Zoned>() || layout.is::<LegacyStats>() {
        return match zones {
            Zones::Drop => {
                let data = layout
                    .slot(0)?
                    .ok_or_else(|| vortex_err!("Zoned layout has no data child"))?;
                lower_with(&data, zones)
            }
            Zones::Keep => Ok(lower_zoned(layout, zones)?.into_plan()),
        };
    }
    if let Some(layout) = layout.as_opt::<Flat>() {
        return Ok(FilterPlan::new(lower_flat(layout).into_plan()).into_plan());
    }
    if let Some(layout) = layout.as_opt::<Chunked>() {
        return Ok(lower_chunked(layout, zones)?.into_plan());
    }
    if let Some(layout) = layout.as_opt::<Struct>() {
        return Ok(lower_struct(layout, zones)?.into_plan());
    }
    if let Some(layout) = layout.as_opt::<Dict>() {
        return Ok(lower_dict(layout, zones)?.into_plan());
    }
    if let Some(layout) = layout.as_opt::<List>() {
        return Ok(lower_list(layout, zones)?.into_plan());
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

fn lower_chunked(layout: &ChunkedLayout, zones: Zones) -> VortexResult<ConcatPlan> {
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
        ConcatPlan::from_children_unchecked(
            layout.dtype().clone(),
            layout.row_count(),
            row_offsets.into(),
            lazy_children(layout.to_layout(), (0..layout.nchildren()).collect(), zones),
        )
    })
}

fn lower_struct(layout: &StructLayout, zones: Zones) -> VortexResult<PackPlan> {
    // Struct layout slot 0 is validity and field i is slot i + 1. The plan puts validity last so
    // field indices are identical to their plan-child indices.
    let fields = layout.struct_fields().clone();
    let mut slots = (1..=fields.nfields()).collect::<Vec<_>>();
    if layout.dtype().is_nullable() {
        slots.push(0);
    }
    // SAFETY: Struct layout construction validates child count and row counts, and its logical
    // slot schema defines every field dtype plus the optional non-nullable boolean validity dtype.
    Ok(unsafe {
        PackPlan::from_children_unchecked(
            fields,
            layout.dtype().nullability(),
            layout.row_count(),
            lazy_children(layout.to_layout(), slots, zones),
        )
    })
}

fn lower_dict(layout: &DictLayout, zones: Zones) -> VortexResult<TakePlan> {
    // The values are shared, so an expression pushed onto them runs over values canonicalized
    // once, and the projection of the same dictionary reuses them, as the V1 dictionary reader does.
    // Dict serialization stores values before codes; the plan order is codes, values.
    let layout = layout.to_layout();
    let slot = |slot| {
        layout
            .slot(slot)?
            .ok_or_else(|| vortex_err!("Dict layout slot {slot} is absent"))
    };
    let codes = lower_with(&slot(1)?, zones)?;
    let values = SharePlan::new(lower_with(&slot(0)?, zones)?).into_plan();
    Ok(TakePlan::new(codes, values))
}

fn lower_list(layout: &ListLayout, zones: Zones) -> VortexResult<ListPackPlan> {
    let mut slots = vec![ELEMENTS_CHILD_INDEX, OFFSETS_CHILD_INDEX];
    if layout.dtype().is_nullable() {
        slots.push(VALIDITY_CHILD_INDEX);
    }
    // SAFETY: List layout construction validates its element, offsets, and optional validity
    // child shapes before this plan preserves them in the same logical order.
    Ok(unsafe {
        ListPackPlan::from_children_unchecked(
            layout.dtype().clone(),
            layout.row_count(),
            lazy_children(layout.to_layout(), slots, zones),
        )
    })
}

fn lower_zoned(layout: &LayoutRef, zones: Zones) -> VortexResult<ZonedPlan> {
    // Zoned and legacy stats layouts share a child shape: transparent data, auxiliary zones.
    let metadata = if let Some(layout) = layout.as_opt::<Zoned>() {
        layout.data()
    } else if let Some(layout) = layout.as_opt::<LegacyStats>() {
        layout.data()
    } else {
        vortex_bail!("Zoned plan requires a zoned layout")
    };
    Ok(ZonedPlan::from_children(
        layout.dtype().clone(),
        layout.row_count(),
        lazy_children(Arc::clone(layout), vec![0, 1], zones),
        u64::try_from(metadata.zone_len())?,
        metadata.aggregate_fns(),
    ))
}

fn lazy_children(layout: LayoutRef, slots: Vec<usize>, zones: Zones) -> PlanChildren {
    PlanChildren::lazy(slots.len(), move |index| {
        let slot = slots
            .get(index)
            .copied()
            .ok_or_else(|| vortex_err!("Missing plan child slot {index}"))?;
        let child = layout
            .slot(slot)?
            .ok_or_else(|| vortex_err!("Layout child slot {slot} is absent"))?;
        lower_with(&child, zones)
    })
}
