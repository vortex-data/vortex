// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.geo.box` extension type (`geoarrow.box`).

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType as _;
use geoarrow::datatypes::BoxType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::Metadata;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::SESSION;
use crate::extension::Rect;
use crate::test_harness::rect_column;

/// A `geoarrow.box` Arrow field of the given dimension.
fn box_field(name: &str, dim: GeoArrowDimension, nullable: bool, crs: Option<&str>) -> Field {
    let crs = crs
        .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
        .unwrap_or_default();
    let metadata = Arc::new(Metadata::new(crs, None));
    BoxType::new(dim, metadata).to_field(name, nullable)
}

/// The exported Arrow field carries the `geoarrow.box` extension over the `Struct<xmin, ymin,
/// xmax, ymax>` layout.
#[test]
fn export_field_carries_extension() -> VortexResult<()> {
    let array = rect_column(vec![(0.0, 0.0, 1.0, 1.0)])?;
    let field = SESSION.arrow().to_arrow_field("bbox", array.dtype())?;

    assert_eq!(field.extension_type_name(), Some(BoxType::NAME));
    let DataType::Struct(fields) = field.data_type() else {
        panic!("expected Struct, got {}", field.data_type());
    };
    let names: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["xmin", "ymin", "xmax", "ymax"]);
    Ok(())
}

/// An imported `geoarrow.box` field maps to the Rect extension dtype, recovering the CRS,
/// box field names, and nullability.
#[test]
fn import_field_recovers_extension() -> VortexResult<()> {
    let field = box_field("bbox", GeoArrowDimension::XY, true, Some("EPSG:4326"));
    let dtype = SESSION.arrow().from_arrow_field(&field)?;

    let DType::Extension(ext) = &dtype else {
        panic!("expected Extension dtype, got {dtype}");
    };
    assert!(ext.is::<Rect>());
    assert_eq!(ext.metadata::<Rect>().crs.as_deref(), Some("EPSG:4326"));

    let DType::Struct(fields, nullability) = ext.storage_dtype() else {
        panic!("expected Struct storage, got {}", ext.storage_dtype());
    };
    assert_eq!(*nullability, Nullability::Nullable);
    let names: Vec<&str> = fields.names().iter().map(|n| n.as_ref()).collect();
    assert_eq!(names, vec!["xmin", "ymin", "xmax", "ymax"]);
    Ok(())
}

/// A `Rect` column exported to Arrow and imported back is unchanged, including the CRS and the
/// box corners.
#[test]
fn roundtrips_through_arrow() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let original = rect_column(vec![(0.0, 1.0, 2.0, 3.0), (-5.0, -5.0, 5.0, 5.0)])?;

    let target = box_field("bbox", GeoArrowDimension::XY, false, Some("EPSG:4326"));
    let exported = SESSION
        .arrow()
        .execute_arrow(original, Some(&target), &mut ctx)?;
    let reimported = SESSION.arrow().from_arrow_array(exported, &target)?;

    let ext = reimported
        .dtype()
        .as_extension_opt()
        .ok_or_else(|| vortex_err!("expected Extension dtype"))?;
    assert!(ext.is::<Rect>());
    assert_eq!(ext.metadata::<Rect>().crs.as_deref(), Some("EPSG:4326"));

    let mut corner = |row: usize, name: &str| -> VortexResult<f64> {
        let scalar = reimported.execute_scalar(row, &mut ctx)?;
        let storage = scalar
            .as_extension_opt()
            .ok_or_else(|| vortex_err!("expected extension scalar"))?
            .to_storage_scalar();
        f64::try_from(
            &storage
                .as_struct()
                .field(name)
                .ok_or_else(|| vortex_err!("missing {name}"))?,
        )
    };
    assert_eq!(corner(0, "xmin")?, 0.0);
    assert_eq!(corner(0, "ymax")?, 3.0);
    assert_eq!(corner(1, "xmin")?, -5.0);
    assert_eq!(corner(1, "ymax")?, 5.0);
    Ok(())
}

/// The existing geo scalar functions run on a `Rect` operand via the shared `geometries()` decode,
/// producing the same results as the equivalent polygon: a box `(0,0)-(10,10)` against interior
/// point `(5,5)` and exterior point `(20,20)`.
#[test]
fn scalar_functions_run_on_rect() -> VortexResult<()> {
    use vortex_array::Canonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;

    use crate::scalar_fn::contains::GeoContains;
    use crate::scalar_fn::distance::GeoDistance;
    use crate::scalar_fn::intersects::GeoIntersects;
    use crate::test_harness::point_column;

    let mut ctx = SESSION.create_execution_ctx();
    let bbox = rect_column(vec![(0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)])?;
    let points = point_column(vec![5.0, 20.0], vec![5.0, 20.0])?;

    // Distance: 0 to the interior point, >0 to the exterior point.
    let distance =
        GeoDistance.try_new_array(bbox.len(), EmptyOptions, [bbox.clone(), points.clone()])?;
    let distance = distance.execute::<Canonical>(&mut ctx)?.into_primitive();
    let distances = distance.as_slice::<f64>();
    assert_eq!(distances[0], 0.0);
    assert!(distances[1] > 0.0);

    // Intersects / Contains: true for the interior point, false for the exterior one.
    let intersects =
        GeoIntersects.try_new_array(bbox.len(), EmptyOptions, [bbox.clone(), points.clone()])?;
    assert_arrays_eq!(intersects, BoolArray::from_iter([true, false]), &mut ctx);

    let contains = GeoContains.try_new_array(bbox.len(), EmptyOptions, [bbox, points])?;
    assert_arrays_eq!(contains, BoolArray::from_iter([true, false]), &mut ctx);
    Ok(())
}
