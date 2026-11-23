// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.st.linestring` extension type (`geoarrow.linestring`).

use std::sync::Arc;

use arrow_array::Array as _;
use arrow_array::Float64Array;
use arrow_array::LargeListArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_buffer::OffsetBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::LineStringType;
use geoarrow::datatypes::Metadata;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;

use super::SESSION;
use crate::test_harness::linestring_column;

/// A `geoarrow.linestring` Arrow field with separated (struct) XY coordinates.
fn linestring_field(name: &str, nullable: bool, crs: Option<&str>) -> Field {
    let crs = crs
        .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
        .unwrap_or_default();
    let metadata = Arc::new(Metadata::new(crs, None));
    LineStringType::new(GeoArrowDimension::XY, metadata).to_field(name, nullable)
}

/// A field with interleaved (`FixedSizeList`) coordinates fails to import.
#[test]
fn import_interleaved_field_fails() {
    let linestring_type = LineStringType::new(GeoArrowDimension::XY, Default::default())
        .with_coord_type(CoordType::Interleaved);
    let field = linestring_type.to_field("geom", false);
    assert!(SESSION.arrow().from_arrow_field(&field).is_err());
}

/// A field in a layout that GeoArrow rejects fails to import instead of losing its CRS.
#[test]
fn import_list_view_field_fails() {
    let field = linestring_field("geom", false, Some("EPSG:4326"));
    let DataType::List(vertices) = field.data_type().clone() else {
        panic!("expected List, got {}", field.data_type());
    };
    let field = field.with_data_type(DataType::ListView(vertices));

    let err = SESSION
        .arrow()
        .from_arrow_field(&field)
        .expect_err("a list view field must fail to import");
    assert!(
        err.to_string().contains("as geoarrow.linestring"),
        "unexpected error: {err}"
    );
}

/// GeoArrow also accepts the `LargeList` layout, and it imports like the `List` layout: coordinates
/// that Arrow marks nullable import as the non-nullable coordinates of a line string.
#[test]
fn imports_large_list() -> VortexResult<()> {
    let coordinates = Fields::from(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
    ]);
    let vertices = ArrowStructArray::new(
        coordinates.clone(),
        vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ],
        None,
    );
    let lines = LargeListArray::new(
        Arc::new(Field::new("vertices", DataType::Struct(coordinates), false)),
        OffsetBuffer::new(vec![0, 2].into()),
        Arc::new(vertices),
        None,
    );
    let field = linestring_field("geom", false, Some("EPSG:4326"))
        .with_data_type(lines.data_type().clone());

    let imported = SESSION.arrow().from_arrow_array(Arc::new(lines), &field)?;
    let expected = linestring_column(vec![vec![(0.0, 0.0), (1.0, 1.0)]])?;
    assert_arrays_eq!(imported, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}
