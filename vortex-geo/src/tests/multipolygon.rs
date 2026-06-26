// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.geo.multipolygon` extension type (`geoarrow.multipolygon`).

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType as _;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::MultiPolygonType;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;

use super::SESSION;
use crate::extension::MultiPolygon;

/// A `geoarrow.multipolygon` Arrow field with separated (struct) XY coordinates.
fn multipolygon_field(name: &str, nullable: bool, crs: Option<&str>) -> Field {
    let crs = crs
        .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
        .unwrap_or_default();
    let metadata = Arc::new(Metadata::new(crs, None));
    MultiPolygonType::new(GeoArrowDimension::XY, metadata).to_field(name, nullable)
}

/// An imported `geoarrow.multipolygon` field maps to the MultiPolygon extension dtype, recovering the
/// CRS, the `List<List<List<Struct<x, y>>>>` storage, and nullability.
#[test]
fn import_field_recovers_extension() -> VortexResult<()> {
    let field = multipolygon_field("geom", true, Some("EPSG:4326"));
    let dtype = SESSION.arrow().from_arrow_field(&field)?;

    let DType::Extension(ext) = &dtype else {
        panic!("expected Extension dtype, got {dtype}");
    };
    assert!(ext.is::<MultiPolygon>());
    assert_eq!(
        ext.metadata::<MultiPolygon>().crs.as_deref(),
        Some("EPSG:4326")
    );

    // Storage peels three List layers (multipolygon → polygons → rings) to the coordinate struct.
    let DType::List(polygons, nullability) = ext.storage_dtype() else {
        panic!("expected List storage, got {}", ext.storage_dtype());
    };
    assert_eq!(*nullability, Nullability::Nullable);
    let DType::List(rings, _) = polygons.as_ref() else {
        panic!("expected List of polygons");
    };
    let DType::List(coords, _) = rings.as_ref() else {
        panic!("expected List of rings");
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
    let multipolygon_type = MultiPolygonType::new(GeoArrowDimension::XY, Default::default())
        .with_coord_type(CoordType::Interleaved);
    let field = multipolygon_type.to_field("geom", false);
    assert!(SESSION.arrow().from_arrow_field(&field).is_err());
}

/// A field imported to the MultiPolygon dtype and exported back carries the `geoarrow.multipolygon`
/// extension over its `List` storage.
#[test]
fn export_field_carries_extension() -> VortexResult<()> {
    let imported =
        SESSION
            .arrow()
            .from_arrow_field(&multipolygon_field("geom", false, Some("EPSG:4326")))?;
    let field = SESSION.arrow().to_arrow_field("geom", &imported)?;

    assert_eq!(field.extension_type_name(), Some(MultiPolygonType::NAME));
    assert!(
        matches!(field.data_type(), DataType::List(_)),
        "expected List storage, got {}",
        field.data_type()
    );
    Ok(())
}
