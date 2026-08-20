// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.st.multipoint` extension type (`geoarrow.multipoint`).

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType as _;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::MultiPointType;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;

use super::SESSION;
use crate::extension::MultiPoint;

/// A `geoarrow.multipoint` Arrow field with separated (struct) XY coordinates.
fn multipoint_field(name: &str, nullable: bool, crs: Option<&str>) -> Field {
    let crs = crs
        .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
        .unwrap_or_default();
    let metadata = Arc::new(Metadata::new(crs, None));
    MultiPointType::new(GeoArrowDimension::XY, metadata).to_field(name, nullable)
}

/// An imported `geoarrow.multipoint` field maps to the MultiPoint extension dtype (not LineString,
/// despite the identical `List<Struct<x, y>>` storage), recovering the CRS and nullability.
#[test]
fn import_field_recovers_extension() -> VortexResult<()> {
    let field = multipoint_field("geom", true, Some("EPSG:4326"));
    let dtype = SESSION.arrow().from_arrow_field(&field)?;

    let DType::Extension(ext) = &dtype else {
        panic!("expected Extension dtype, got {dtype}");
    };
    assert!(ext.is::<MultiPoint>());
    assert_eq!(
        ext.metadata::<MultiPoint>().crs.as_deref(),
        Some("EPSG:4326")
    );

    let DType::List(coords, nullability) = ext.storage_dtype() else {
        panic!("expected List storage, got {}", ext.storage_dtype());
    };
    assert_eq!(*nullability, Nullability::Nullable);
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
    let multipoint_type = MultiPointType::new(GeoArrowDimension::XY, Default::default())
        .with_coord_type(CoordType::Interleaved);
    let field = multipoint_type.to_field("geom", false);
    assert!(SESSION.arrow().from_arrow_field(&field).is_err());
}

/// A field imported to the MultiPoint dtype and exported back carries the `geoarrow.multipoint`
/// extension over its `List` storage.
#[test]
fn export_field_carries_extension() -> VortexResult<()> {
    let imported =
        SESSION
            .arrow()
            .from_arrow_field(&multipoint_field("geom", false, Some("EPSG:4326")))?;
    let field = SESSION.arrow().to_arrow_field("geom", &imported)?;

    assert_eq!(field.extension_type_name(), Some(MultiPointType::NAME));
    assert!(
        matches!(field.data_type(), DataType::List(_)),
        "expected List storage, got {}",
        field.data_type()
    );
    Ok(())
}
