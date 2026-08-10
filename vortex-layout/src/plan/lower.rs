// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test support for constructing physical plans from stored layout trees.
//!
//! This module is only used to build physical-plan fixtures for tests. It is not a production
//! planning API.

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
use crate::plan::ConcatPlan;
use crate::plan::ListPackPlan;
use crate::plan::PackPlan;
use crate::plan::PlanChildren;
use crate::plan::PlanRef;
use crate::plan::SegmentScanPlan;
use crate::plan::TakePlan;

/// Constructs a physical-plan fixture from `layout` for tests.
///
/// The root operator is built immediately. Its child container owns a hidden clone of the source
/// layout and lowers each child independently on first access.
pub fn lower(layout: &LayoutRef) -> VortexResult<PlanRef> {
    if let Some(layout) = layout.as_opt::<Flat>() {
        return Ok(lower_flat(layout).into_plan());
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
    Ok(ConcatPlan::from_children(
        layout.dtype().clone(),
        layout.row_count(),
        row_offsets.into(),
        lazy_children(layout.to_layout(), (0..layout.nchildren()).collect()),
    ))
}

fn lower_struct(layout: &StructLayout) -> VortexResult<PackPlan> {
    // Struct layout slot 0 is validity and field i is slot i + 1. The plan puts validity last so
    // field indices are identical to their plan-child indices.
    let fields = layout.struct_fields().clone();
    let mut slots = (1..=fields.nfields()).collect::<Vec<_>>();
    if layout.dtype().is_nullable() {
        slots.push(0);
    }
    Ok(PackPlan::from_children(
        fields,
        layout.dtype().nullability(),
        layout.row_count(),
        lazy_children(layout.to_layout(), slots),
    ))
}

fn lower_dict(layout: &DictLayout) -> VortexResult<TakePlan> {
    // Dict serialization stores values before codes; the plan order is deliberately codes,
    // values because that is the optimizer-facing logical shape.
    Ok(TakePlan::from_children(
        layout.dtype().clone(),
        layout.row_count(),
        lazy_children(layout.to_layout(), vec![1, 0]),
    ))
}

fn lower_list(layout: &ListLayout) -> VortexResult<ListPackPlan> {
    let mut slots = vec![ELEMENTS_CHILD_INDEX, OFFSETS_CHILD_INDEX];
    if layout.dtype().is_nullable() {
        slots.push(VALIDITY_CHILD_INDEX);
    }
    Ok(ListPackPlan::from_children(
        layout.dtype().clone(),
        layout.row_count(),
        lazy_children(layout.to_layout(), slots),
    ))
}

fn lazy_children(layout: LayoutRef, slots: Vec<usize>) -> PlanChildren {
    PlanChildren::lazy(slots.len(), move |index| {
        let slot = slots
            .get(index)
            .copied()
            .ok_or_else(|| vortex_err!("Missing plan child slot {index}"))?;
        let child = layout
            .slot(slot)?
            .ok_or_else(|| vortex_err!("Layout child slot {slot} is absent"))?;
        lower(&child)
    })
}
