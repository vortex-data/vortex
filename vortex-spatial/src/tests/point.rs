// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.st.point` extension type (`geoarrow.point`).

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Float64Array;
use arrow_array::StructArray as ArrowStructArray;
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::PointType;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;

use super::SESSION;
use crate::test_harness::nullable_point_column;

/// A `geoarrow.point` Arrow field with separated (struct) XY coordinates.
fn point_field(name: &str, nullable: bool, crs: Option<&str>) -> Field {
    let crs = crs
        .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
        .unwrap_or_default();
    let metadata = Arc::new(Metadata::new(crs, None));
    PointType::new(GeoArrowDimension::XY, metadata).to_field(name, nullable)
}

/// A field with interleaved (`FixedSizeList`) coordinates fails to import.
#[test]
fn import_interleaved_field_fails() {
    let point_type = PointType::new(GeoArrowDimension::XY, Default::default())
        .with_coord_type(CoordType::Interleaved);
    let field = point_type.to_field("loc", false);
    assert!(SESSION.arrow().from_arrow_field(&field).is_err());
}

/// A nullable `geoarrow.point` column whose coordinate fields Arrow marks nullable, over the given
/// coordinates, where `points` is the validity of the points.
fn points_with_nullable_coordinates(
    xs: Vec<Option<f64>>,
    ys: Vec<Option<f64>>,
    points: Option<NullBuffer>,
) -> (ArrowArrayRef, Field) {
    let coordinates = Fields::from(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
    ]);
    let field = point_field("loc", true, Some("EPSG:4326"))
        .with_data_type(DataType::Struct(coordinates.clone()));
    let array = ArrowStructArray::new(
        coordinates,
        vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
        ],
        points,
    );
    (Arc::new(array), field)
}

/// Coordinates that Arrow marks nullable import as the non-nullable coordinates of a point, and
/// the null coordinates of a null point are dropped.
#[test]
fn imports_nullable_coordinates() -> VortexResult<()> {
    let (arrow, field) = points_with_nullable_coordinates(
        vec![Some(1.0), None, Some(3.0)],
        vec![Some(2.0), None, Some(4.0)],
        Some(NullBuffer::from(vec![true, false, true])),
    );

    let imported = SESSION.arrow().from_arrow_array(arrow, &field)?;
    let expected = nullable_point_column(vec![Some((1.0, 2.0)), None, Some((3.0, 4.0))])?;
    assert_arrays_eq!(imported, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// A null coordinate of a valid point is outside the domain of `Point`, so the import fails.
#[test]
fn null_coordinate_of_valid_point_fails() {
    let (arrow, field) = points_with_nullable_coordinates(vec![Some(1.0)], vec![None], None);
    let err = SESSION
        .arrow()
        .from_arrow_array(arrow, &field)
        .expect_err("a null coordinate of a valid point must fail the import");
    assert!(
        err.to_string()
            .contains("Failed to import Arrow storage as vortex.st.point"),
        "unexpected error: {err}"
    );
}
