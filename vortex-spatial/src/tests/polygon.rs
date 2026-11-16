// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Polygon export after selections that can retain unused coordinate children.
//!
//! Arrow allows those children, but reconstructing a GeoArrow polygon imposes a stricter length
//! constraint. The trailing null row keeps that distinction visible during batch selections.

use geoarrow::datatypes::PolygonType;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::SESSION;
use crate::extension::Dimension;
use crate::extension::Polygon;
use crate::extension::SpatialMetadata;

/// Each present row is a closed XYM triangle with the given x origin. Null rows contain no rings.
fn polygon_column(source: &[Option<f64>], ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ring_offsets = vec![0i32];
    let mut polygon_offsets = vec![0i32];

    for &origin in source {
        if let Some(x) = origin {
            xs.extend([x, x + 1.0, x, x]);
            ys.extend([0.0, 0.0, 1.0, 0.0]);
            ring_offsets.push(i32::try_from(xs.len())?);
        }
        polygon_offsets.push(i32::try_from(ring_offsets.len() - 1)?);
    }

    let coordinate_count = xs.len();
    let coordinates = StructArray::from_fields(&[
        ("x", PrimitiveArray::from_iter(xs).into_array()),
        ("y", PrimitiveArray::from_iter(ys).into_array()),
        (
            "m",
            PrimitiveArray::from_iter(vec![0.0f64; coordinate_count]).into_array(),
        ),
    ])?
    .into_array();
    let rings = ListArray::try_new(
        coordinates,
        PrimitiveArray::from_iter(ring_offsets).into_array(),
        Validity::NonNullable,
        ctx,
    )?
    .into_array();
    let storage = ListArray::try_new(
        rings,
        PrimitiveArray::from_iter(polygon_offsets).into_array(),
        Validity::from_iter(source.iter().map(Option::is_some)),
        ctx,
    )?
    .into_array();

    Ok(ExtensionArray::try_new_from_vtable(
        Polygon,
        SpatialMetadata::default().with_dimension(Dimension::Xym),
        storage,
    )?
    .into_array())
}

#[track_caller]
fn check_selected_polygons() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let source = polygon_column(&[Some(0.0), Some(1.0), None], &mut ctx)?;
    let field = PolygonType::new(Dimension::Xym.into(), Default::default()).to_field("value", true);
    let selections = [
        ("slice", source.slice(1..3)?, vec![Some(1.0), None]),
        (
            "take",
            source.take(PrimitiveArray::from_iter([2u32, 2, 1, 1, 0, 0]).into_array())?,
            vec![None, None, Some(1.0), Some(1.0), Some(0.0), Some(0.0)],
        ),
        (
            "filter",
            source.filter(Mask::from_iter([true, false, true]))?,
            vec![Some(0.0), None],
        ),
    ];

    for (operation, selected, expected) in selections {
        let selected = selected.execute::<ExtensionArray>(&mut ctx)?.into_array();
        let exported = selected.execute_arrow(Some(&field), &mut ctx)?;
        exported.to_data().validate_full()?;
        assert_eq!(exported.data_type(), field.data_type(), "{operation} dtype");

        let imported = SESSION.arrow().from_arrow_array(exported, &field)?;
        let expected = polygon_column(&expected, &mut ctx)?;
        assert_arrays_eq!(imported, expected, &mut ctx);
    }

    Ok(())
}

#[test]
fn selected_polygons_export_with_unused_coordinate_children() -> VortexResult<()> {
    check_selected_polygons()
}
