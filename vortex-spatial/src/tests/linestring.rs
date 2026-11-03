// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.st.linestring` extension type (`geoarrow.linestring`).

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType as _;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::LineStringType;
use geoarrow::datatypes::Metadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;

use super::SESSION;
use crate::extension::LineString;

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


