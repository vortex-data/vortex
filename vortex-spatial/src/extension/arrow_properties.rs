// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generated geometry families and metadata for Arrow transport checks.
//!
//! Each case samples one family so shrinking can reduce the geometry as well as its storage.

mod tests {
    use std::sync::Arc;

    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_schema::Field;
    use geo_traits::to_geo::ToGeoGeometry;
    use geoarrow::datatypes::BoxType;
    use geoarrow::datatypes::Crs;
    use geoarrow::datatypes::LineStringType;
    use geoarrow::datatypes::Metadata;
    use geoarrow::datatypes::MultiLineStringType;
    use geoarrow::datatypes::MultiPointType;
    use geoarrow::datatypes::MultiPolygonType;
    use geoarrow::datatypes::PointType;
    use geoarrow::datatypes::PolygonType;
    use geoarrow::datatypes::WkbType;
    use hegel::TestCase;
    use hegel::generators as gs;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::builders::builder_with_capacity_in;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtDTypeRef;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_arrow::test_harness::StorageCase;
    use vortex_arrow::test_harness::check_arrow;
    use vortex_arrow::test_harness::check_generated;
    use vortex_buffer::BufferAllocatorRef;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::extension::Dimension;
    use crate::extension::LineString;
    use crate::extension::MultiLineString;
    use crate::extension::MultiPoint;
    use crate::extension::MultiPolygon;
    use crate::extension::Point;
    use crate::extension::Polygon;
    use crate::extension::Rect;
    use crate::extension::SpatialMetadata;
    use crate::extension::WellKnownBinary;

    #[derive(Clone, Copy, Debug)]
    enum GeometryKind {
        Point,
        Rect,
        LineString,
        MultiPoint,
        Polygon,
        MultiLineString,
        MultiPolygon,
        Wkb,
    }

    impl GeometryKind {
        fn dtype(
            self,
            metadata: SpatialMetadata,
            dimension: Dimension,
            nullability: Nullability,
        ) -> VortexResult<ExtDTypeRef> {
            let metadata = if matches!(self, Self::Wkb) {
                metadata
            } else {
                metadata.with_dimension(dimension)
            };

            Ok(match self {
                Self::Point => ExtDType::try_with_vtable(Point, metadata, nullability)?.erased(),
                Self::Rect => ExtDType::try_with_vtable(Rect, metadata, nullability)?.erased(),
                Self::LineString => {
                    ExtDType::try_with_vtable(LineString, metadata, nullability)?.erased()
                }
                Self::MultiPoint => {
                    ExtDType::try_with_vtable(MultiPoint, metadata, nullability)?.erased()
                }
                Self::Polygon => {
                    ExtDType::try_with_vtable(Polygon, metadata, nullability)?.erased()
                }
                Self::MultiLineString => {
                    ExtDType::try_with_vtable(MultiLineString, metadata, nullability)?.erased()
                }
                Self::MultiPolygon => {
                    ExtDType::try_with_vtable(MultiPolygon, metadata, nullability)?.erased()
                }
                Self::Wkb => {
                    ExtDType::try_with_vtable(WellKnownBinary, metadata, nullability)?.erased()
                }
            })
        }
    }

    #[track_caller]
    fn check_spatial_round_trip(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullability = tc.draw(gs::booleans()).into();
        let dimension = tc.draw(gs::sampled_from(&[
            Dimension::Xy,
            Dimension::Xyz,
            Dimension::Xym,
            Dimension::Xyzm,
        ]));
        let crs = tc.draw(gs::sampled_from(&[
            None,
            Some(""),
            Some("EPSG:4326"),
            Some("非ASCII"),
        ]));
        let metadata = SpatialMetadata::new(crs.map(Into::into));

        let kind = tc.draw(gs::sampled_from(&[
            GeometryKind::Point,
            GeometryKind::Rect,
            GeometryKind::LineString,
            GeometryKind::MultiPoint,
            GeometryKind::Polygon,
            GeometryKind::MultiLineString,
            GeometryKind::MultiPolygon,
            GeometryKind::Wkb,
        ]));
        let dtype = kind.dtype(metadata, dimension, nullability)?;
        let arrow_metadata = Arc::new(Metadata::new(
            crs.map(|s| Crs::from_unknown_crs_type(s.to_string()))
                .unwrap_or_default(),
            None,
        ));
        let dim = dimension.into();
        let field = match kind {
            GeometryKind::Point => {
                PointType::new(dim, arrow_metadata).to_field("value", nullability.is_nullable())
            }
            GeometryKind::Rect => {
                BoxType::new(dim, arrow_metadata).to_field("value", nullability.is_nullable())
            }
            GeometryKind::LineString => LineStringType::new(dim, arrow_metadata)
                .to_field("value", nullability.is_nullable()),
            GeometryKind::MultiPoint => MultiPointType::new(dim, arrow_metadata)
                .to_field("value", nullability.is_nullable()),
            GeometryKind::Polygon => {
                PolygonType::new(dim, arrow_metadata).to_field("value", nullability.is_nullable())
            }
            GeometryKind::MultiLineString => MultiLineStringType::new(dim, arrow_metadata)
                .to_field("value", nullability.is_nullable()),
            GeometryKind::MultiPolygon => MultiPolygonType::new(dim, arrow_metadata)
                .to_field("value", nullability.is_nullable()),
            GeometryKind::Wkb => {
                let mut field = Field::new(
                    "value",
                    arrow_schema::DataType::BinaryView,
                    nullability.is_nullable(),
                );
                field.try_with_extension_type(WkbType::new(arrow_metadata))?;
                field
            }
        };

        if matches!(kind, GeometryKind::Wkb) {
            let WkbCase {
                array: storage,
                bytes: source,
                ..
            } = draw_wkb_storage(tc, nullability, len)?;
            let array = ExtensionArray::try_new(dtype, storage)?.into_array();
            let bytes = source.iter().map(|row| row.as_deref());
            let alternate_layouts: [ArrowArrayRef; 2] = [
                Arc::new(arrow_array::BinaryArray::from_iter(bytes.clone())),
                Arc::new(arrow_array::LargeBinaryArray::from_iter(bytes.clone())),
            ];
            for expected in alternate_layouts {
                let target = field.clone().with_data_type(expected.data_type().clone());
                check_arrow(array.clone(), &target, expected, &mut ctx)?;
            }
            let expected = Arc::new(arrow_array::BinaryViewArray::from_iter(bytes));
            check_generated(tc, array, &field, expected, &mut ctx)
        } else {
            let source = (0..len)
                .map(|_| {
                    if nullability.is_nullable() && tc.draw(gs::booleans()) {
                        None
                    } else {
                        Some(draw_geometry(tc, dtype.storage_dtype(), kind))
                    }
                })
                .collect::<Vec<_>>();
            let storage =
                build_geometry(dtype.storage_dtype(), field.data_type(), &source, &mut ctx)?;
            let array = ExtensionArray::try_new(dtype, storage.array)?.into_array();
            check_generated(tc, array, &field, storage.arrow, &mut ctx)
        }
    }

    #[derive(Clone)]
    enum GeometryValue {
        Coordinates(Vec<f64>),
        List(Vec<Self>),
    }

    fn draw_geometry(tc: &TestCase, dtype: &DType, kind: GeometryKind) -> GeometryValue {
        match dtype {
            DType::Struct(fields, _) => {
                let mut values = (0..fields.names().len())
                    .map(|_| f64::from(tc.draw(gs::integers::<i32>())) / 8.0)
                    .collect::<Vec<_>>();
                if matches!(kind, GeometryKind::Rect) {
                    let axes = values.len() / 2;
                    for axis in 0..axes {
                        let (min, max) = (
                            values[axis].min(values[axes + axis]),
                            values[axis].max(values[axes + axis]),
                        );
                        values[axis] = min;
                        values[axes + axis] = max;
                    }
                }
                GeometryValue::Coordinates(values)
            }
            DType::List(child, _) => {
                let ring = matches!(kind, GeometryKind::Polygon | GeometryKind::MultiPolygon)
                    && matches!(child.as_ref(), DType::Struct(..));
                let len = if ring {
                    3
                } else {
                    tc.draw(gs::integers::<usize>().max_value(4))
                };
                let mut values = (0..len)
                    .map(|_| draw_geometry(tc, child, kind))
                    .collect::<Vec<_>>();
                if ring {
                    values.push(values[0].clone());
                }
                GeometryValue::List(values)
            }
            _ => unreachable!("native geometry storage contains lists and coordinate structs"),
        }
    }

    /// Builds both representations from source geometry values, using the independent Arrow field.
    /// Coordinate arities and list nesting must agree with the two storage dtypes.
    #[track_caller]
    fn build_geometry(
        dtype: &DType,
        arrow_dtype: &arrow_schema::DataType,
        source: &[Option<GeometryValue>],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<StorageCase> {
        assert!(
            dtype.is_nullable() || source.iter().all(Option::is_some),
            "non-nullable geometry storage requires non-null source rows"
        );
        let validity = if dtype.is_nullable() {
            Validity::from_iter(source.iter().map(Option::is_some))
        } else {
            Validity::NonNullable
        };
        let nulls = dtype.is_nullable().then(|| {
            arrow_buffer::NullBuffer::from(source.iter().map(Option::is_some).collect::<Vec<_>>())
        });
        match dtype {
            DType::Struct(fields, _) => {
                for value in source.iter().flatten() {
                    let GeometryValue::Coordinates(values) = value else {
                        unreachable!("struct storage requires coordinate source rows")
                    };
                    assert_eq!(values.len(), fields.names().len());
                }

                let mut arrays = Vec::new();
                let mut arrow_arrays = Vec::new();
                let mut arrow_fields = Vec::new();
                for (index, name) in fields.names().iter().enumerate() {
                    let values = source
                        .iter()
                        .map(|v| match v {
                            Some(GeometryValue::Coordinates(v)) => v[index],
                            None => 0.0,
                            _ => unreachable!("struct storage requires coordinate source rows"),
                        })
                        .collect::<Vec<_>>();
                    arrays.push(PrimitiveArray::from_iter(values.clone()).into_array());
                    arrow_arrays
                        .push(Arc::new(arrow_array::Float64Array::from(values)) as ArrowArrayRef);
                    arrow_fields.push(Field::new(
                        name.as_ref(),
                        arrow_schema::DataType::Float64,
                        false,
                    ));
                }

                Ok(StorageCase {
                    array: StructArray::try_new(
                        fields.names().clone(),
                        arrays,
                        source.len(),
                        validity,
                    )?
                    .into_array(),
                    arrow: Arc::new(arrow_array::StructArray::try_new_with_length(
                        arrow_fields.into(),
                        arrow_arrays,
                        nulls,
                        source.len(),
                    )?),
                })
            }
            DType::List(child, _) => {
                let mut offsets = vec![0i32];
                let mut children = Vec::new();
                for value in source {
                    match value {
                        Some(GeometryValue::List(values)) => {
                            children.extend(values.iter().cloned().map(Some))
                        }
                        None => {}
                        Some(GeometryValue::Coordinates(_)) => {
                            unreachable!("list storage requires list source rows")
                        }
                    }
                    offsets.push(i32::try_from(children.len())?);
                }
                let arrow_schema::DataType::List(field) = arrow_dtype else {
                    unreachable!("GeoArrow list storage")
                };
                let child = build_geometry(child, field.data_type(), &children, ctx)?;
                let field = Arc::clone(field);
                Ok(StorageCase {
                    array: ListArray::try_new(
                        child.array,
                        PrimitiveArray::from_iter(offsets.clone()).into_array(),
                        validity,
                        ctx,
                    )?
                    .into_array(),
                    arrow: Arc::new(arrow_array::ListArray::try_new(
                        field,
                        arrow_buffer::OffsetBuffer::new(offsets.into()),
                        child.arrow,
                        nulls,
                    )?),
                })
            }
            _ => unreachable!("geometry storage has only list and struct levels"),
        }
    }

    struct WkbCase {
        array: ArrayRef,
        bytes: Vec<Option<Vec<u8>>>,
        geometries: Vec<Option<geo_types::Geometry>>,
    }

    fn draw_wkb_storage(
        tc: &TestCase,
        nullability: Nullability,
        len: usize,
    ) -> VortexResult<WkbCase> {
        let mut builder = builder_with_capacity_in(
            &DType::Binary(nullability),
            len,
            BufferAllocatorRef::static_ref(),
        );
        let mut bytes = Vec::with_capacity(len);
        let mut geometries = Vec::with_capacity(len);
        for _ in 0..len {
            if nullability.is_nullable() && tc.draw(gs::booleans()) {
                builder.append_nulls(1);
                bytes.push(None);
                geometries.push(None);
            } else {
                let x = f64::from(tc.draw(gs::integers::<i32>())) / 8.0;
                let y = f64::from(tc.draw(gs::integers::<i32>())) / 8.0;
                let geometry = if tc.draw(gs::booleans()) {
                    geo_types::Geometry::Point(geo_types::Point::new(x, y))
                } else {
                    geo_types::Geometry::Polygon(geo_types::Polygon::new(
                        geo_types::LineString::from(vec![
                            (x, y),
                            (x + 1.0, y),
                            (x + 1.0, y + 1.0),
                            (x, y),
                        ]),
                        vec![],
                    ))
                };
                let mut encoded = Vec::new();
                wkb::writer::write_geometry(
                    &mut encoded,
                    &geometry,
                    &wkb::writer::WriteOptions::default(),
                )
                .map_err(|e| vortex_error::vortex_err!("writing source WKB: {e}"))?;
                builder.append_scalar(&Scalar::binary(encoded.clone(), nullability))?;
                bytes.push(Some(encoded));
                geometries.push(Some(geometry));
            }
        }

        Ok(WkbCase {
            array: builder.finish(),
            bytes,
            geometries,
        })
    }

    #[track_caller]
    fn check_native_points(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullable: Nullability = tc.draw(gs::booleans()).into();
        let dimension = tc.draw(gs::sampled_from(&[
            Dimension::Xy,
            Dimension::Xyz,
            Dimension::Xym,
            Dimension::Xyzm,
        ]));
        let dtype = ExtDType::try_with_vtable(
            Point,
            SpatialMetadata::default().with_dimension(dimension),
            nullable,
        )?;
        let source = (0..len)
            .map(|_| {
                if nullable.is_nullable() && tc.draw(gs::booleans()) {
                    None
                } else {
                    Some(draw_geometry(
                        tc,
                        dtype.storage_dtype(),
                        GeometryKind::Point,
                    ))
                }
            })
            .collect::<Vec<_>>();
        let field = PointType::new(dimension.into(), Default::default())
            .to_field("point", nullable.is_nullable());
        let storage = build_geometry(dtype.storage_dtype(), field.data_type(), &source, &mut ctx)?;
        for (i, source) in source.iter().enumerate() {
            let scalar = storage.array.execute_scalar(i, &mut ctx)?;
            let actual = scalar
                .value()
                .map(|v| Point::unpack_native(&dtype, v))
                .transpose()?;
            match (source, actual) {
                (None, None) => {}
                (Some(GeometryValue::Coordinates(source)), Some(actual)) => {
                    assert_eq!(actual.x, source[0]);
                    assert_eq!(actual.y, source[1]);
                    assert_eq!(
                        actual.z,
                        match dimension {
                            Dimension::Xyz | Dimension::Xyzm => Some(source[2]),
                            _ => None,
                        }
                    );
                    assert_eq!(
                        actual.m,
                        match dimension {
                            Dimension::Xym => Some(source[2]),
                            Dimension::Xyzm => Some(source[3]),
                            _ => None,
                        }
                    );
                }
                _ => panic!("native point nullability must match the source"),
            }
        }

        Ok(())
    }

    #[track_caller]
    fn check_native_wkb(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullable: Nullability = tc.draw(gs::booleans()).into();
        let source = draw_wkb_storage(tc, nullable, len)?;
        let dtype =
            ExtDType::try_with_vtable(WellKnownBinary, SpatialMetadata::default(), nullable)?;
        for (i, expected) in source.geometries.iter().enumerate() {
            let scalar = source.array.execute_scalar(i, &mut ctx)?;
            let actual = scalar
                .value()
                .map(|v| WellKnownBinary::unpack_native(&dtype, v))
                .transpose()?;
            assert_eq!(actual.map(|g| g.to_geometry()), *expected);
        }

        Ok(())
    }

    /// The trailing null makes filtering leave unused coordinate children in valid Arrow lists.
    /// Reconstructing a GeoArrow polygon rejects that layout even though Arrow accepts it.
    #[track_caller]
    fn check_filtered_polygons(base: i32) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let dtype = ExtDType::try_with_vtable(
            Polygon,
            SpatialMetadata::default().with_dimension(Dimension::Xym),
            Nullability::Nullable,
        )?;
        let field =
            PolygonType::new(Dimension::Xym.into(), Default::default()).to_field("value", true);
        let mut source = (0..2)
            .map(|row| {
                let x = f64::from(base) + f64::from(row);
                Some(GeometryValue::List(vec![GeometryValue::List(vec![
                    GeometryValue::Coordinates(vec![x, 0.0, 0.0]),
                    GeometryValue::Coordinates(vec![x + 1.0, 0.0, 0.0]),
                    GeometryValue::Coordinates(vec![x, 1.0, 0.0]),
                    GeometryValue::Coordinates(vec![x, 0.0, 0.0]),
                ])]))
            })
            .collect::<Vec<_>>();
        source.push(None);
        let storage = build_geometry(dtype.storage_dtype(), field.data_type(), &source, &mut ctx)?;
        let array = ExtensionArray::try_new(dtype.erased(), storage.array)?.into_array();
        check_arrow(array, &field, storage.arrow, &mut ctx)
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    #[hegel::explicit_test_case(base = 0i32)]
    fn filtered_polygons(tc: TestCase) {
        let base = tc.draw(gs::integers::<i32>());
        check_filtered_polygons(base)
            .vortex_expect("filtered polygons must export with unused coordinate children");
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn native_points(tc: TestCase) {
        check_native_points(&tc)
            .vortex_expect("native point decoding must preserve source coordinates");
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn native_wkb(tc: TestCase) {
        check_native_wkb(&tc).vortex_expect("native WKB decoding must preserve source geometries");
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn spatial_arrow_transport(tc: TestCase) {
        check_spatial_round_trip(&tc)
            .vortex_expect("spatial Arrow transport must preserve dtype and stored values");
    }
}
