// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.geo.point` extension type (`geoarrow.point`).

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Float64Array;
use arrow_array::StructArray as ArrowStructArray;
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::extension::ExtensionType as _;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::PointType;
use vortex_array::VortexSessionExecute;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::SESSION;
use crate::extension::Point;
use crate::extension::coordinate::Coordinate;
use crate::test_harness::coordinate_from_scalar;
use crate::test_harness::point_column;

/// A `geoarrow.point` Arrow field with separated (struct) XY coordinates.
fn point_field(name: &str, nullable: bool, crs: Option<&str>) -> Field {
    let crs = crs
        .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
        .unwrap_or_default();
    let metadata = Arc::new(Metadata::new(crs, None));
    PointType::new(GeoArrowDimension::XY, metadata).to_field(name, nullable)
}

/// An Arrow `Struct<x, y>` point array with non-nullable `Float64` children.
fn arrow_point_struct(xs: Vec<f64>, ys: Vec<f64>) -> ArrowStructArray {
    let fields: Fields = vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]
    .into();
    ArrowStructArray::new(
        fields,
        vec![
            Arc::new(Float64Array::from(xs)) as ArrowArrayRef,
            Arc::new(Float64Array::from(ys)),
        ],
        None,
    )
}

/// The exported Arrow field carries the `geoarrow.point` extension over the separated
/// `Struct<x, y>` coordinate layout.
#[test]
fn export_field_carries_extension() -> VortexResult<()> {
    let array = point_column(vec![1.0], vec![2.0])?;
    let field = SESSION.arrow().to_arrow_field("loc", array.dtype())?;

    assert_eq!(field.extension_type_name(), Some(PointType::NAME));
    let DataType::Struct(fields) = field.data_type() else {
        panic!("expected Struct, got {}", field.data_type());
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name(), "x");
    assert_eq!(fields[0].data_type(), &DataType::Float64);
    assert_eq!(fields[1].name(), "y");
    assert_eq!(fields[1].data_type(), &DataType::Float64);
    Ok(())
}

/// Export materializes the point column as an Arrow struct with the original ordinates.
#[test]
fn exports_to_struct() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = point_column(vec![1.0, 3.0], vec![2.0, 4.0])?;

    let target = point_field("loc", false, Some("EPSG:4326"));
    let exported = SESSION
        .arrow()
        .execute_arrow(array, Some(&target), &mut ctx)?;

    let points = exported.as_struct();
    let ordinates = |name: &str| -> VortexResult<Vec<f64>> {
        Ok(points
            .column_by_name(name)
            .ok_or_else(|| vortex_err!("missing {name} column"))?
            .as_primitive::<Float64Type>()
            .values()
            .to_vec())
    };
    assert_eq!(ordinates("x")?, vec![1.0, 3.0]);
    assert_eq!(ordinates("y")?, vec![2.0, 4.0]);
    Ok(())
}

/// An imported `geoarrow.point` field maps to the Point extension dtype, recovering the
/// CRS, coordinate field names, and nullability.
#[test]
fn import_field_recovers_extension() -> VortexResult<()> {
    let field = point_field("loc", true, Some("EPSG:4326"));
    let dtype = SESSION.arrow().from_arrow_field(&field)?;

    let DType::Extension(ext) = &dtype else {
        panic!("expected Extension dtype, got {dtype}");
    };
    assert!(ext.is::<Point>());
    assert_eq!(ext.metadata::<Point>().crs.as_deref(), Some("EPSG:4326"));

    let DType::Struct(fields, nullability) = ext.storage_dtype() else {
        panic!("expected Struct storage, got {}", ext.storage_dtype());
    };
    assert_eq!(*nullability, Nullability::Nullable);
    let names: Vec<&str> = fields.names().iter().map(|n| n.as_ref()).collect();
    assert_eq!(names, vec!["x", "y"]);
    Ok(())
}

/// A field with interleaved (`FixedSizeList`) coordinates fails to import.
#[test]
fn import_interleaved_field_fails() {
    let point_type = PointType::new(GeoArrowDimension::XY, Default::default())
        .with_coord_type(CoordType::Interleaved);
    let field = point_type.to_field("loc", false);
    assert!(SESSION.arrow().from_arrow_field(&field).is_err());
}

/// Import wraps the Arrow struct's coordinate buffers into a Point column.
#[test]
fn imports_from_struct() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let arrow: ArrowArrayRef =
        Arc::new(arrow_point_struct(vec![1.0, -111.7610], vec![2.0, 34.8697]));
    let field = point_field("loc", false, Some("EPSG:4326"));

    let imported = SESSION.arrow().from_arrow_array(arrow, &field)?;
    assert!(
        imported
            .dtype()
            .as_extension_opt()
            .map(|ext| ext.is::<Point>())
            .unwrap_or(false)
    );

    assert_eq!(
        coordinate_from_scalar(&imported.execute_scalar(0, &mut ctx)?)?,
        Coordinate::xy(1.0, 2.0)
    );
    assert_eq!(
        coordinate_from_scalar(&imported.execute_scalar(1, &mut ctx)?)?,
        Coordinate::xy(-111.7610, 34.8697)
    );
    Ok(())
}

/// A point column exported to Arrow and imported back is unchanged, including the CRS.
#[test]
fn roundtrips_through_arrow() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let original = point_column(vec![0.0, 3.0], vec![4.0, 0.0])?;

    let target = point_field("loc", false, Some("EPSG:4326"));
    let exported = SESSION
        .arrow()
        .execute_arrow(original, Some(&target), &mut ctx)?;
    let reimported = SESSION.arrow().from_arrow_array(exported, &target)?;

    let ext = reimported
        .dtype()
        .as_extension_opt()
        .ok_or_else(|| vortex_err!("expected Extension dtype"))?;
    assert_eq!(ext.metadata::<Point>().crs.as_deref(), Some("EPSG:4326"));

    assert_eq!(
        coordinate_from_scalar(&reimported.execute_scalar(0, &mut ctx)?)?,
        Coordinate::xy(0.0, 4.0)
    );
    assert_eq!(
        coordinate_from_scalar(&reimported.execute_scalar(1, &mut ctx)?)?,
        Coordinate::xy(3.0, 0.0)
    );
    Ok(())
}
