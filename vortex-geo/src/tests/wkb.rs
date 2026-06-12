// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.geo.wkb` extension type (`geoarrow.wkb`).

use std::sync::Arc;

use arrow_array::Array as _;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::BinaryArray;
use arrow_array::BinaryViewArray;
use arrow_array::LargeBinaryArray;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType as _;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Coord;
use geo_types::Geometry;
use geo_types::LineString;
use geo_types::Polygon;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::WkbType;
use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use wkb::writer::WriteOptions;

use super::SESSION;
use crate::extension::GeoMetadata;
use crate::extension::WellKnownBinary;

/// The polygon geometry encoded by these tests.
fn test_polygon() -> Geometry {
    Geometry::Polygon(Polygon::new(
        LineString::new(vec![
            Coord::zero(),
            Coord { x: 100.0, y: 0.0 },
            Coord { x: 100.0, y: 100.0 },
            Coord { x: 0.0, y: 100.0 },
            Coord::zero(),
        ]),
        vec![],
    ))
}

/// A WKB column (CRS `EPSG:4326`) holding [`test_polygon`] three times, along with the
/// polygon's WKB bytes.
fn wkb_extension_array() -> VortexResult<(Vec<u8>, vortex_array::ArrayRef)> {
    let mut buf = Vec::new();
    // We should always prefer to write little-endian, which is the default option.
    wkb::writer::write_geometry(&mut buf, &test_polygon(), &WriteOptions::default())
        .map_err(|e| vortex_err!("writing WKB failed: {e}"))?;

    let mut builder = VarBinBuilder::<i32>::with_capacity(3);
    builder.append_value(&buf);
    builder.append_value(&buf);
    builder.append_value(&buf);

    let dtype = ExtDType::<WellKnownBinary>::try_new(
        GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        },
        DType::Binary(Nullability::NonNullable),
    )?;
    let storage = builder.finish(DType::Binary(Nullability::NonNullable));
    let array = ExtensionArray::new(dtype.erased(), storage.into_array()).into_array();
    Ok((buf, array))
}

/// A `geoarrow.wkb` Arrow field over the given binary data type.
fn wkb_field(name: &str, data_type: DataType, nullable: bool, crs: Option<&str>) -> Field {
    let crs = crs
        .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
        .unwrap_or_default();
    let wkb_type = WkbType::new(Arc::new(Metadata::new(crs, None)));
    Field::new(name, data_type, nullable).with_extension_type(wkb_type)
}

fn assert_imported_wkb_dtype(dtype: &DType, expected_crs: Option<&str>, nullable: bool) {
    let DType::Extension(ext) = dtype else {
        panic!("expected Extension dtype, got {dtype}");
    };
    assert!(ext.is::<WellKnownBinary>());
    assert_eq!(ext.storage_dtype(), &DType::Binary(nullable.into()));
    let geo = ext.metadata::<WellKnownBinary>();
    assert_eq!(geo.crs.as_deref(), expected_crs);
}

/// WKB scalars unpack back to the geometry they encode.
#[test]
fn scalar_unpacks_to_geometry() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (_, array) = wkb_extension_array()?;

    let dtype = ExtDType::<WellKnownBinary>::try_new(
        GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        },
        DType::Binary(Nullability::NonNullable),
    )?;

    for idx in 0..3 {
        let geom = array.execute_scalar(idx, &mut ctx)?;
        let wkb = WellKnownBinary::unpack_native(&dtype, geom.value().unwrap())?;
        assert_eq!(wkb.to_geometry(), test_polygon());
    }
    Ok(())
}

/// The exported Arrow field carries the `geoarrow.wkb` extension over Vortex's canonical
/// binary mapping (`BinaryView`).
#[test]
fn export_field_carries_extension() -> VortexResult<()> {
    let (_, array) = wkb_extension_array()?;
    let field = SESSION.arrow().to_arrow_field("geom", array.dtype())?;
    assert_eq!(field.extension_type_name(), Some(WkbType::NAME));
    assert_eq!(field.data_type(), &DataType::BinaryView);
    Ok(())
}

/// Export materializes the WKB bytes into the requested binary-family Arrow array.
#[rstest]
#[case::binary(DataType::Binary)]
#[case::large_binary(DataType::LargeBinary)]
#[case::binary_view(DataType::BinaryView)]
fn exports_to_binary_family(#[case] data_type: DataType) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (wkb_bytes, array) = wkb_extension_array()?;

    let field = Field::new("geom", data_type.clone(), false)
        .with_extension_type(WkbType::new(Default::default()));
    let exported = SESSION
        .arrow()
        .execute_arrow(array, Some(&field), &mut ctx)?;

    assert_eq!(exported.data_type(), &data_type);
    let values: Vec<&[u8]> = match &data_type {
        DataType::Binary => exported.as_binary::<i32>().iter().flatten().collect(),
        DataType::LargeBinary => exported.as_binary::<i64>().iter().flatten().collect(),
        DataType::BinaryView => exported.as_binary_view().iter().flatten().collect(),
        _ => unreachable!("cases cover only the binary family"),
    };
    assert_eq!(values.len(), 3);
    for value in values {
        assert_eq!(value, wkb_bytes.as_slice());
    }
    Ok(())
}

/// An imported `geoarrow.wkb` field maps to the WKB extension dtype, recovering the CRS.
#[test]
fn import_field_recovers_extension() -> VortexResult<()> {
    let field = wkb_field("geom", DataType::Binary, false, Some("EPSG:4326"));
    let dtype = SESSION.arrow().from_arrow_field(&field)?;
    assert_imported_wkb_dtype(&dtype, Some("EPSG:4326"), false);
    Ok(())
}

/// A `geoarrow.wkb` field without a CRS imports as an unreferenced geometry.
#[test]
fn import_field_without_crs() -> VortexResult<()> {
    let field = wkb_field("geom", DataType::BinaryView, true, None);
    let dtype = SESSION.arrow().from_arrow_field(&field)?;
    assert_imported_wkb_dtype(&dtype, None, true);
    Ok(())
}

/// Import wraps the binary-family Arrow array's WKB values unchanged.
#[test]
fn imports_from_binary() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (wkb_bytes, _) = wkb_extension_array()?;
    let arrow: ArrowArrayRef = Arc::new(BinaryArray::from_iter_values([
        wkb_bytes.as_slice(),
        wkb_bytes.as_slice(),
        wkb_bytes.as_slice(),
    ]));
    let field = wkb_field("geom", DataType::Binary, false, Some("EPSG:4326"));

    let imported = SESSION.arrow().from_arrow_array(arrow, &field)?;
    assert_imported_wkb_dtype(imported.dtype(), Some("EPSG:4326"), false);

    for idx in 0..3 {
        let geom = imported.execute_scalar(idx, &mut ctx)?;
        assert_eq!(geom.value().unwrap().as_binary().as_slice(), wkb_bytes);
    }
    Ok(())
}

/// Import wraps the binary-family Arrow array's WKB values unchanged.
#[test]
fn imports_from_large_binary() -> VortexResult<()> {
    let (wkb_bytes, _) = wkb_extension_array()?;
    let arrow: ArrowArrayRef = Arc::new(LargeBinaryArray::from_iter_values([
        wkb_bytes.as_slice(),
        wkb_bytes.as_slice(),
    ]));
    let field = wkb_field("geom", DataType::LargeBinary, false, Some("EPSG:4326"));

    let imported = SESSION.arrow().from_arrow_array(arrow, &field)?;
    assert_imported_wkb_dtype(imported.dtype(), Some("EPSG:4326"), false);
    assert_eq!(imported.len(), 2);
    Ok(())
}

/// Import wraps the binary-family Arrow array's WKB values unchanged.
#[test]
fn imports_from_binary_view() -> VortexResult<()> {
    let (wkb_bytes, _) = wkb_extension_array()?;
    let arrow: ArrowArrayRef = Arc::new(BinaryViewArray::from_iter_values([
        wkb_bytes.as_slice(),
        wkb_bytes.as_slice(),
        wkb_bytes.as_slice(),
        wkb_bytes.as_slice(),
    ]));
    let field = wkb_field("geom", DataType::BinaryView, false, None);

    let imported = SESSION.arrow().from_arrow_array(arrow, &field)?;
    assert_imported_wkb_dtype(imported.dtype(), None, false);
    assert_eq!(imported.len(), 4);
    Ok(())
}

/// A WKB column exported to Arrow and imported back is unchanged, byte for byte.
#[test]
fn roundtrips_through_arrow() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (wkb_bytes, original) = wkb_extension_array()?;

    let field = Field::new("geom", DataType::BinaryView, false)
        .with_extension_type(WkbType::new(Default::default()));
    let exported = SESSION
        .arrow()
        .execute_arrow(original, Some(&field), &mut ctx)?;
    let arrow_field = wkb_field("geom", DataType::BinaryView, false, Some("EPSG:4326"));
    let reimported = SESSION.arrow().from_arrow_array(exported, &arrow_field)?;

    assert_imported_wkb_dtype(reimported.dtype(), Some("EPSG:4326"), false);
    for idx in 0..3 {
        let geom = reimported.execute_scalar(idx, &mut ctx)?;
        assert_eq!(geom.value().unwrap().as_binary().as_slice(), wkb_bytes);
    }
    Ok(())
}
