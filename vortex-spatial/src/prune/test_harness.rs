// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared zone-map fixtures for the pruning-rule tests; each rule module owns its own cases.

use std::sync::Arc;

use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_layout::layouts::zoned::zone_map::ZoneMap;

use crate::aggregate_fn::GeometryAabb;
use crate::extension::Rect;
use crate::extension::SpatialMetadata;

/// A single-column zone map holding one native-box AABB stat row (`[xmin, ymin, xmax, ymax]`)
/// per zone, with default (unreferenced) metadata to match the aggregate's return dtype.
pub(super) fn aabb_zone_map(point_dtype: &DType, boxes: &[[f64; 4]]) -> VortexResult<ZoneMap> {
    let aabb_fn = GeometryAabb.bind(EmptyOptions);
    let col = |i: usize| PrimitiveArray::from_iter(boxes.iter().map(move |b| b[i])).into_array();
    let storage = StructArray::try_new(
        ["xmin", "ymin", "xmax", "ymax"].into(),
        vec![col(0), col(1), col(2), col(3)],
        boxes.len(),
        Validity::AllValid,
    )?
    .into_array();
    let box_dtype =
        ExtDType::<Rect>::try_new(SpatialMetadata::default(), storage.dtype().clone())?.erased();
    let aabbs = ExtensionArray::try_new(box_dtype, storage)?.into_array();
    let zone_array = StructArray::from_fields(&[(aabb_fn.to_string().as_str(), aabbs)])?;
    ZoneMap::try_new(
        point_dtype.clone(),
        zone_array,
        Arc::new([aabb_fn]),
        1,
        boxes.len() as u64,
    )
}

/// A two-zone zone map with no stat columns at all, as written before the AABB stat existed.
pub(super) fn empty_zone_map(point_dtype: &DType) -> VortexResult<ZoneMap> {
    ZoneMap::try_new(
        point_dtype.clone(),
        StructArray::try_new(FieldNames::empty(), vec![], 2, Validity::NonNullable)?,
        Arc::new([]),
        1,
        2,
    )
}
