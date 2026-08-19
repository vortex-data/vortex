// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the mixed `vortex.st.geometry` extension (`geoarrow.geometry`).

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::UnionArray as ArrowUnionArray;
use arrow_schema::Field;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry as GeoGeometry;
use geo_types::GeometryCollection;
use geo_types::LineString;
use geo_types::MultiLineString;
use geo_types::MultiPoint;
use geo_types::MultiPolygon;
use geo_types::Point;
use geo_types::Polygon;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::array::GeometryArray as GeoArrowGeometryArray;
use geoarrow::array::GeometryBuilder;
use geoarrow::array::IntoArrow;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::GeometryType as GeoArrowGeometryType;
use geoarrow::datatypes::Metadata;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray as SparseUnionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::SESSION;
use crate::aggregate_fn::GeometryAabb;
use crate::dense_union::DenseUnion;
use crate::extension::Geometry;
use crate::extension::decode_geometries;
use crate::scalar_fn::envelope::SpatialEnvelope;
use crate::test_harness::nullable_rect_column;

fn polygon(points: &[(f64, f64)]) -> Polygon<f64> {
    Polygon::new(LineString::from(points.to_vec()), vec![])
}

fn supported_geometries() -> Vec<Option<GeoGeometry<f64>>> {
    let line = LineString::from(vec![(3.0, -4.0), (8.0, 7.0)]);
    let polygon = polygon(&[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 0.0)]);
    vec![
        Some(GeoGeometry::Point(Point::new(1.0, 2.0))),
        Some(GeoGeometry::LineString(line.clone())),
        Some(GeoGeometry::Polygon(polygon.clone())),
        Some(GeoGeometry::MultiPoint(MultiPoint::new(vec![
            Point::new(-2.0, 5.0),
            Point::new(4.0, 9.0),
        ]))),
        Some(GeoGeometry::MultiLineString(MultiLineString::new(vec![
            line,
        ]))),
        Some(GeoGeometry::MultiPolygon(MultiPolygon::new(vec![polygon]))),
        None,
    ]
}

fn geoarrow_geometry_type() -> GeoArrowGeometryType {
    let crs = Crs::from_unknown_crs_type("EPSG:4326".to_string());
    GeoArrowGeometryType::new(Arc::new(Metadata::new(crs, None)))
        .with_coord_type(CoordType::Separated)
}

fn arrow_fixture(geometries: &[Option<GeoGeometry<f64>>]) -> VortexResult<(ArrowArrayRef, Field)> {
    let geometry_type = geoarrow_geometry_type();
    let array = GeometryBuilder::from_nullable_geometries(geometries, geometry_type.clone())
        .map_err(|e| vortex_err!("failed to build GeoArrow geometry array: {e}"))?
        .finish();
    let field = geometry_type.to_field("geom", true);
    Ok((Arc::new(array.into_arrow()), field))
}

fn decode_arrow(
    array: &ArrowArrayRef,
    field: &Field,
) -> VortexResult<Vec<Option<GeoGeometry<f64>>>> {
    let geometries = GeoArrowGeometryArray::try_from((array.as_ref(), field))
        .map_err(|e| vortex_err!("failed to decode exported GeoArrow geometry array: {e}"))?;
    geometries
        .iter()
        .map(|geometry| match geometry {
            None => Ok(None),
            Some(Ok(geometry)) => Ok(Some(geometry.to_geometry())),
            Some(Err(e)) => Err(vortex_err!("failed to access GeoArrow geometry: {e}")),
        })
        .collect()
}

/// The [`supported_geometries`] fixture imported as a `vortex.st.geometry` column, alongside the
/// GeoArrow field it came from and the geometries it should read back as.
struct Fixture {
    imported: ArrayRef,
    field: Field,
    expected: Vec<Option<GeoGeometry<f64>>>,
}

fn imported_fixture() -> VortexResult<Fixture> {
    let expected = supported_geometries();
    let (arrow, field) = arrow_fixture(&expected)?;
    let imported = SESSION.arrow().from_arrow_array(arrow, &field)?;
    Ok(Fixture {
        imported,
        field,
        expected,
    })
}

fn aabb(result: &Scalar) -> VortexResult<(f64, f64, f64, f64)> {
    let storage = result.as_extension().to_storage_scalar();
    let fields = storage.as_struct();
    let read = |name: &str| -> VortexResult<f64> {
        f64::try_from(
            &fields
                .field(name)
                .ok_or_else(|| vortex_err!("AABB result is missing {name}"))?,
        )
    };
    Ok((read("xmin")?, read("ymin")?, read("xmax")?, read("ymax")?))
}

#[test]
fn imports_as_geometry_over_logical_union() -> VortexResult<()> {
    let field = geoarrow_geometry_type().to_field("geom", true);
    let dtype = SESSION.arrow().from_arrow_field(&field)?;

    let DType::Extension(ext) = &dtype else {
        return Err(vortex_err!(
            "expected Geometry extension dtype, got {dtype}"
        ));
    };
    assert!(ext.is::<Geometry>());
    assert_eq!(ext.metadata::<Geometry>().crs.as_deref(), Some("EPSG:4326"));

    let DType::Union(variants, nullability) = ext.storage_dtype() else {
        return Err(vortex_err!(
            "expected logical Union storage, got {}",
            ext.storage_dtype()
        ));
    };
    assert!(nullability.is_nullable());
    assert_eq!(variants.len(), 24);
    assert!(variants.type_ids().iter().all(|type_id| type_id % 10 != 7));
    Ok(())
}

#[test]
fn infers_canonical_geoarrow_field() -> VortexResult<()> {
    let expected_type = geoarrow_geometry_type();
    let dtype = SESSION
        .arrow()
        .from_arrow_field(&expected_type.to_field("input", true))?;
    let field = SESSION.arrow().to_arrow_field("geom", &dtype)?;

    assert_eq!(field.name(), "geom");
    assert!(field.is_nullable());
    assert_eq!(field.data_type(), &expected_type.data_type());
    let actual_type = field
        .try_extension_type::<GeoArrowGeometryType>()
        .map_err(|e| vortex_err!("failed to read inferred GeoArrow GeometryType: {e}"))?;
    assert_eq!(actual_type, expected_type);
    Ok(())
}

#[test]
fn roundtrips_supported_kinds_and_nulls() -> VortexResult<()> {
    let Fixture {
        imported,
        field,
        expected,
    } = imported_fixture()?;

    let mut ctx = SESSION.create_execution_ctx();
    let extension = imported.clone().execute::<ExtensionArray>(&mut ctx)?;
    assert!(extension.storage_array().is::<DenseUnion>());
    assert_eq!(
        imported
            .validity()?
            .execute_mask(imported.len(), &mut ctx)?
            .iter()
            .collect::<Vec<_>>(),
        vec![true, true, true, true, true, true, false]
    );

    let exported = SESSION
        .arrow()
        .execute_arrow(imported, Some(&field), &mut ctx)?;
    assert_eq!(decode_arrow(&exported, &field)?, expected);
    Ok(())
}

#[test]
fn preserves_non_first_type_id_for_nulls() -> VortexResult<()> {
    let expected = vec![
        Some(GeoGeometry::LineString(LineString::from(vec![
            (0.0, 1.0),
            (2.0, 3.0),
        ]))),
        None,
    ];
    let (arrow, field) = arrow_fixture(&expected)?;
    let input = arrow
        .as_any()
        .downcast_ref::<ArrowUnionArray>()
        .ok_or_else(|| vortex_err!("GeoArrow fixture must use a UnionArray"))?;
    assert_eq!(input.type_ids().as_ref(), &[2, 2]);

    let imported = SESSION.arrow().from_arrow_array(arrow, &field)?;
    let mut ctx = SESSION.create_execution_ctx();
    let exported = SESSION
        .arrow()
        .execute_arrow(imported, Some(&field), &mut ctx)?;
    let output = exported
        .as_any()
        .downcast_ref::<ArrowUnionArray>()
        .ok_or_else(|| vortex_err!("exported GeoArrow Geometry must use a UnionArray"))?;
    assert_eq!(output.type_ids().as_ref(), &[2, 2]);
    assert_eq!(decode_arrow(&exported, &field)?, expected);
    Ok(())
}

#[test]
fn slice_preserves_dense_union_storage() -> VortexResult<()> {
    let Fixture {
        imported,
        field,
        expected,
    } = imported_fixture()?;
    let sliced = imported.slice(1..5)?;

    let mut ctx = SESSION.create_execution_ctx();
    let extension = sliced.clone().execute::<ExtensionArray>(&mut ctx)?;
    assert!(extension.storage_array().is::<DenseUnion>());

    let exported = SESSION
        .arrow()
        .execute_arrow(sliced, Some(&field), &mut ctx)?;
    assert_eq!(decode_arrow(&exported, &field)?, expected[1..5]);
    Ok(())
}

#[test]
fn take_compacts_dense_union_for_arrow() -> VortexResult<()> {
    let geometries = vec![
        Some(GeoGeometry::Point(Point::new(1.0, 2.0))),
        Some(GeoGeometry::Point(Point::new(3.0, 4.0))),
        Some(GeoGeometry::Point(Point::new(5.0, 6.0))),
    ];
    let (arrow, field) = arrow_fixture(&geometries)?;
    let imported = SESSION.arrow().from_arrow_array(arrow, &field)?;
    let taken = imported.take(PrimitiveArray::from_iter([2u32, 0, 2]).into_array())?;

    let mut ctx = SESSION.create_execution_ctx();
    let exported = SESSION
        .arrow()
        .execute_arrow(taken, Some(&field), &mut ctx)?;
    let union = exported
        .as_any()
        .downcast_ref::<ArrowUnionArray>()
        .ok_or_else(|| vortex_err!("exported GeoArrow Geometry must use a UnionArray"))?;

    assert_eq!(union.offsets().unwrap().as_ref(), &[0, 1, 2]);
    assert_eq!(union.child(1).len(), 3);
    assert_eq!(
        decode_arrow(&exported, &field)?,
        vec![
            geometries[2].clone(),
            geometries[0].clone(),
            geometries[2].clone()
        ]
    );
    Ok(())
}

#[test]
fn exports_canonical_sparse_union() -> VortexResult<()> {
    let Fixture {
        imported,
        field,
        expected,
    } = imported_fixture()?;
    let mut ctx = SESSION.create_execution_ctx();

    let extension = imported.execute::<ExtensionArray>(&mut ctx)?;
    let sparse = extension
        .storage_array()
        .clone()
        .execute::<SparseUnionArray>(&mut ctx)?;
    let sparse_geometry =
        ExtensionArray::try_new(extension.ext_dtype().clone(), sparse.into_array())?.into_array();
    let exported = SESSION
        .arrow()
        .execute_arrow(sparse_geometry, Some(&field), &mut ctx)?;
    assert_eq!(decode_arrow(&exported, &field)?, expected);
    Ok(())
}

#[test]
fn exports_constant_nulls() -> VortexResult<()> {
    let Fixture {
        imported, field, ..
    } = imported_fixture()?;
    let mut ctx = SESSION.create_execution_ctx();
    let nulls = ConstantArray::new(Scalar::null(imported.dtype().clone()), 2).into_array();
    let exported = SESSION
        .arrow()
        .execute_arrow(nulls, Some(&field), &mut ctx)?;
    assert_eq!(decode_arrow(&exported, &field)?, vec![None, None]);
    Ok(())
}

#[test]
fn decodes_supported_variants() -> VortexResult<()> {
    let Fixture {
        imported, expected, ..
    } = imported_fixture()?;
    let mut ctx = SESSION.create_execution_ctx();

    let non_null = imported.slice(0..6)?;
    let expected_non_null = expected[..6]
        .iter()
        .filter_map(Clone::clone)
        .collect::<Vec<_>>();
    assert_eq!(decode_geometries(&non_null, &mut ctx)?, expected_non_null);
    Ok(())
}

#[test]
fn computes_aabb_across_variants() -> VortexResult<()> {
    let Fixture { imported, .. } = imported_fixture()?;
    let mut ctx = SESSION.create_execution_ctx();
    let mut accumulator =
        Accumulator::try_new(GeometryAabb, EmptyOptions, imported.dtype().clone())?;
    accumulator.accumulate(&imported, &mut ctx)?;
    assert_eq!(aabb(&accumulator.finish()?)?, (-2.0, -4.0, 8.0, 9.0));
    Ok(())
}

#[test]
fn computes_envelopes_across_variants() -> VortexResult<()> {
    let Fixture { imported, .. } = imported_fixture()?;
    let mut ctx = SESSION.create_execution_ctx();
    let envelopes = SpatialEnvelope::try_new_array(imported)?
        .into_array()
        .execute::<ExtensionArray>(&mut ctx)?;
    let expected_envelopes = nullable_rect_column(vec![
        Some((1.0, 2.0, 1.0, 2.0)),
        Some((3.0, -4.0, 8.0, 7.0)),
        Some((0.0, 0.0, 4.0, 3.0)),
        Some((-2.0, 5.0, 4.0, 9.0)),
        Some((3.0, -4.0, 8.0, 7.0)),
        Some((0.0, 0.0, 4.0, 3.0)),
        None,
    ])?;
    assert_arrays_eq!(envelopes, expected_envelopes, &mut ctx);
    Ok(())
}

#[test]
fn rejects_selected_geometry_collection() -> VortexResult<()> {
    let geometries = vec![Some(GeoGeometry::GeometryCollection(
        GeometryCollection::new_from(vec![
            GeoGeometry::Point(Point::new(1.0, 2.0)),
            GeoGeometry::Point(Point::new(3.0, 4.0)),
        ]),
    ))];
    let (arrow, field) = arrow_fixture(&geometries)?;
    let Err(error) = SESSION.arrow().from_arrow_array(arrow, &field) else {
        return Err(vortex_err!("selected GeometryCollection must be rejected"));
    };
    assert!(error.to_string().contains("GeometryCollection"));
    Ok(())
}
