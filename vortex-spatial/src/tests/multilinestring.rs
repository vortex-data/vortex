// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.st.multilinestring` extension type (`geoarrow.multilinestring`).

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType as _;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::MultiLineStringType;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;

use super::SESSION;
use crate::extension::MultiLineString;

/// A `geoarrow.multilinestring` Arrow field with separated (struct) XY coordinates.
fn multilinestring_field(name: &str, nullable: bool, crs: Option<&str>) -> Field {
    let crs = crs
        .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
        .unwrap_or_default();
    let metadata = Arc::new(Metadata::new(crs, None));
    MultiLineStringType::new(GeoArrowDimension::XY, metadata).to_field(name, nullable)
}

/// An imported `geoarrow.multilinestring` field maps to the MultiLineString extension dtype (not
/// Polygon, despite the identical `List<List<Struct<x, y>>>` storage), recovering CRS and shape.
#[test]
fn import_field_recovers_extension() -> VortexResult<()> {
    let field = multilinestring_field("geom", true, Some("EPSG:4326"));
    let dtype = SESSION.arrow().from_arrow_field(&field)?;

    let DType::Extension(ext) = &dtype else {
        panic!("expected Extension dtype, got {dtype}");
    };
    assert!(ext.is::<MultiLineString>());
    assert_eq!(
        ext.metadata::<MultiLineString>().crs.as_deref(),
        Some("EPSG:4326")
    );

    // Storage peels two List layers (multilinestring → line strings) to the coordinate struct.
    let DType::List(lines, nullability) = ext.storage_dtype() else {
        panic!("expected List storage, got {}", ext.storage_dtype());
    };
    assert_eq!(*nullability, Nullability::Nullable);
    let DType::List(coords, _) = lines.as_ref() else {
        panic!("expected List of line strings");
    };
    let DType::Struct(fields, _) = coords.as_ref() else {
        panic!("expected coordinate Struct");
    };
    let names: Vec<&str> = fields.names().iter().map(|n| n.as_ref()).collect();
    assert_eq!(names, vec!["x", "y"]);
    Ok(())
}

/// A field with interleaved (`FixedSizeList`) coordinates fails to import.
#[test]
fn import_interleaved_field_fails() {
    let multilinestring_type = MultiLineStringType::new(GeoArrowDimension::XY, Default::default())
        .with_coord_type(CoordType::Interleaved);
    let field = multilinestring_type.to_field("geom", false);
    assert!(SESSION.arrow().from_arrow_field(&field).is_err());
}

/// A field imported to the MultiLineString dtype and exported back carries the
/// `geoarrow.multilinestring` extension over its `List` storage.
#[test]
fn export_field_carries_extension() -> VortexResult<()> {
    let imported = SESSION.arrow().from_arrow_field(&multilinestring_field(
        "geom",
        false,
        Some("EPSG:4326"),
    ))?;
    let field = SESSION.arrow().to_arrow_field("geom", &imported)?;

    assert_eq!(field.extension_type_name(), Some(MultiLineStringType::NAME));
    assert!(
        matches!(field.data_type(), DataType::List(_)),
        "expected List storage, got {}",
        field.data_type()
    );
    Ok(())
}
