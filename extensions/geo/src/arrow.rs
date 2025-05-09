//! Conversions between GeoVortex and GeoArrow arrays and extension types, using
//! the `geoarrow` crate.

#[allow(clippy::disallowed_types)]
use std::collections::HashMap;
use std::sync::Arc;

use arcref::ArcRef;
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::extension::{
    EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY, ExtensionType,
};
use arrow_schema::{DataType, Field};
use geoarrow::ArrayBase;
use geoarrow::array::{CoordBuffer, LineStringArray, PointArray, SeparatedCoordBuffer};
use geoarrow_schema::{Crs, LineStringType, PointType, PolygonType, WkbType};
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrow::ArrowArray;
use vortex_array::arrow::compute::{ToArrowArgs, ToArrowKernelRef};
use vortex_array::compute::{InvocationArgs, Kernel, Output, cast};
use vortex_array::{Array, ToCanonical, register_kernel};
use vortex_dtype::arrow::{ArrowTypeConversion, ArrowTypeConversionRef};
use vortex_dtype::{DType, ExtDType, PType, register_extension_type};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::array::GeometryArray;
use crate::{Dimension, GeoMetadata, GeometryType, OwnedGeoMetadata, OwnedGeometryType};

/// Kernel to convert into GeoArrow memory format.
#[derive(Debug)]
pub struct ToGeoArrow;

impl Kernel for ToGeoArrow {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let ToArrowArgs { array, .. } = ToArrowArgs::try_from(args)?;
        match array.as_any().downcast_ref::<ExtensionArray>() {
            None => Ok(None),
            Some(ext_array) => {
                if let Ok(geometry_array) = GeometryArray::try_from(ext_array) {
                    // based on the particular geometry, encode into GeoArrow type.
                    let array = match geometry_array {
                        GeometryArray::Point(point_array, metadata) => {
                            to_arrow_point(point_array, &metadata)?
                        }
                        GeometryArray::LineString(line_string, metadata) => {
                            to_arrow_linestring(line_string, &metadata)?
                        }
                        GeometryArray::Polygon(..) => todo!(),
                        GeometryArray::WKB(..) => todo!(),
                    };

                    Ok(Some(Output::Array(
                        ArrowArray::new(array, ext_array.dtype().nullability()).into_array(),
                    )))
                } else {
                    Ok(None)
                }
            }
        }
    }
}
register_kernel!(ToArrowKernelRef(ArcRef::new_ref(&ToGeoArrow)));

fn to_arrow_point(
    point_array: &ExtensionArray,
    metadata: &GeoMetadata,
) -> VortexResult<arrow_array::ArrayRef> {
    let coords = coordinate_buffer(point_array.storage(), metadata)?;
    let nulls = point_array.validity_mask()?.to_null_buffer();
    let crs = match metadata.crs {
        None => Crs::default(),
        Some(wkt) => Crs::from_wkt2_2019(wkt.to_string()),
    };

    Ok(PointArray::new(
        coords,
        nulls,
        Arc::new(geoarrow_schema::Metadata::new(crs, None)),
    )
    .into_array_ref())
}

fn to_arrow_linestring(
    line_string: &ExtensionArray,
    metadata: &GeoMetadata,
) -> VortexResult<arrow_array::ArrayRef> {
    let list = line_string.to_list()?;
    let coords = coordinate_buffer(list.elements().as_ref(), metadata)?;
    let offsets = cast(&list.offsets().to_primitive()?, &PType::I32.into())?
        .to_primitive()?
        .into_buffer::<i32>()
        .into_arrow_scalar_buffer();
    let offsets = OffsetBuffer::new(offsets);
    let nulls = list.validity_mask()?.to_null_buffer();

    let crs = match metadata.crs {
        None => Crs::default(),
        Some(wkt) => Crs::from_wkt2_2019(wkt.to_string()),
    };

    Ok(LineStringArray::new(
        coords,
        offsets,
        nulls,
        Arc::new(geoarrow_schema::Metadata::new(crs, None)),
    )
    .into_array_ref())
}

/// ZST that holds the VTable for building a `DType` from an Arrow [`Field`] type.
#[derive(Debug)]
pub struct GeoArrowConversion;

// Conversion between Vortex geospatial extension types and GeoArrow extension types.
impl ArrowTypeConversion for GeoArrowConversion {
    fn to_vortex(&self, field: &Field) -> VortexResult<Option<DType>> {
        // Validate that the field is one of the supported geospatial
        // extension types.
        let Some(ext_type) = field.extension_type_name() else {
            return Ok(None);
        };

        if ext_type != PointType::NAME
            && ext_type != LineStringType::NAME
            && ext_type != PolygonType::NAME
            && ext_type != WkbType::NAME
        {
            return Ok(None);
        }

        macro_rules! dim_from_fields {
            ($fields:expr) => {{
                match $fields.len() {
                    2 => {
                        // XY is only supported 2-item format
                        Dimension::XY
                    }
                    3 => {
                        // Can be either XYZ or XYM
                        if $fields.iter().any(|x| x.name().to_lowercase() == "m") {
                            Dimension::XYM
                        } else {
                            Dimension::XYZ
                        }
                    }
                    4 => Dimension::XYZM,
                    _ => vortex_bail!("Unsupported field layout for Point geometry: {:?}", $fields),
                }
            }};
        }

        let ext_dtype: ExtDType = match ext_type {
            x if x == PointType::NAME => {
                let geoarrow_meta = geoarrow_schema::Metadata::try_from(field)?;
                // We only accept Separated format.
                let DataType::Struct(fields) = field.data_type() else {
                    vortex_bail!("Only Separated format is supported for Point geometry")
                };

                OwnedGeometryType::Point(OwnedGeoMetadata {
                    dimension: dim_from_fields!(fields),
                    crs: geoarrow_meta.crs().crs_value().map(|v| v.to_string()),
                })
                .into_ext_dtype(field.is_nullable().into())
            }
            x if x == LineStringType::NAME => {
                let geoarrow_meta = geoarrow_schema::Metadata::try_from(field)?;
                let DataType::List(coordinates_field) = field.data_type() else {
                    vortex_bail!(
                        "LineString geometry must be List<Struct>, was {:?}",
                        field.data_type()
                    )
                };
                let DataType::Struct(fields) = coordinates_field.data_type() else {
                    vortex_bail!(
                        "LineString geometry must be List<Struct>, was {:?}",
                        field.data_type()
                    )
                };
                OwnedGeometryType::LineString(OwnedGeoMetadata {
                    dimension: dim_from_fields!(fields),
                    crs: geoarrow_meta.crs().crs_value().map(|v| v.to_string()),
                })
                .into_ext_dtype(field.is_nullable().into())
            }
            x if x == PolygonType::NAME => {
                let geoarrow_meta = geoarrow_schema::Metadata::try_from(field)?;
                let DataType::List(ring_field) = field.data_type() else {
                    vortex_bail!(
                        "LineString geometry must be List<Struct>, was {:?}",
                        field.data_type()
                    )
                };
                let DataType::List(coordinates_field) = ring_field.data_type() else {
                    vortex_bail!(
                        "Polygon geometry must be List<List<Struct>>, was {:?}",
                        field.data_type()
                    )
                };
                let DataType::Struct(fields) = coordinates_field.data_type() else {
                    vortex_bail!(
                        "LineString geometry must be List<Struct>, was {:?}",
                        field.data_type()
                    )
                };
                OwnedGeometryType::Polygon(OwnedGeoMetadata {
                    dimension: dim_from_fields!(fields),
                    crs: geoarrow_meta.crs().crs_value().map(|v| v.to_string()),
                })
                .into_ext_dtype(field.is_nullable().into())
            }
            x if x == WkbType::NAME => {
                let geoarrow_meta = geoarrow_schema::Metadata::try_from(field)?;
                if field.data_type() != &DataType::Binary {
                    vortex_bail!(
                        "WKB geometry Arrow type must be Binary, was {:?}",
                        field.data_type()
                    )
                }
                OwnedGeometryType::WKB(OwnedGeoMetadata {
                    dimension: Dimension::default(),
                    crs: geoarrow_meta.crs().crs_value().map(|v| v.to_string()),
                })
                .into_ext_dtype(field.is_nullable().into())
            }
            _ => vortex_bail!("extension type {} not supported", ext_type),
        };

        Ok(Some(DType::Extension(Arc::new(ext_dtype))))
    }

    #[allow(clippy::disallowed_types)]
    fn arrow_metadata(
        &self,
        vortex_extension_type: &ExtDType,
    ) -> VortexResult<Option<HashMap<String, String>>> {
        if let Ok(geometry) = GeometryType::try_from(vortex_extension_type) {
            let mut extension_metadata = HashMap::new();
            let ext_type_name = match geometry {
                GeometryType::Point(_) => PointType::NAME.to_string(),
                GeometryType::LineString(_) => LineStringType::NAME.to_string(),
                GeometryType::Polygon(_) => PolygonType::NAME.to_string(),
                GeometryType::WKB(_) => WkbType::NAME.to_string(),
            };
            extension_metadata.insert(EXTENSION_TYPE_NAME_KEY.to_string(), ext_type_name);

            let extension_metadata = match geometry {
                GeometryType::Point(meta)
                | GeometryType::LineString(meta)
                | GeometryType::Polygon(meta)
                | GeometryType::WKB(meta) => {
                    if let Some(wkt) = meta.crs {
                        let crs = Crs::from_wkt2_2019(wkt.to_string());
                        let metadata_json =
                            serde_json::to_string(&geoarrow_schema::Metadata::new(crs, None))
                                .map_err(|e| {
                                    vortex_err!("Encoding geoarrow metadata failed: {}", e)
                                })
                                .vortex_expect("failed to serialize geoarrow metadata");
                        extension_metadata
                            .insert(EXTENSION_TYPE_METADATA_KEY.to_string(), metadata_json);
                        extension_metadata
                    } else {
                        extension_metadata
                    }
                }
            };

            Ok(Some(extension_metadata))
        } else {
            Ok(None)
        }
    }
}

register_extension_type!(ArrowTypeConversionRef::new(ArcRef::new_ref(
    &GeoArrowConversion
)));

/// Unpack the geoarrow CoordBuffer. Errors if the dimensions specified in the metadata do not
/// match the actual encoding.
fn coordinate_buffer(array: &dyn Array, meta: &GeoMetadata) -> VortexResult<CoordBuffer> {
    let children = array.children();
    match (children.len(), meta.dimension) {
        (2, Dimension::XY) => {
            let xs = children[0]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            let ys = children[1]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            Ok(CoordBuffer::Separated(SeparatedCoordBuffer::new(
                [
                    xs,
                    ys,
                    ScalarBuffer::from(Vec::<f64>::new()),
                    ScalarBuffer::from(Vec::<f64>::new()),
                ],
                geoarrow_schema::Dimension::XY,
            )))
        }
        (3, Dimension::XYZ) => {
            let xs = children[0]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            let ys = children[1]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            let zs = children[2]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            Ok(CoordBuffer::Separated(SeparatedCoordBuffer::new(
                [xs, ys, zs, ScalarBuffer::from(Vec::<f64>::new())],
                geoarrow_schema::Dimension::XYZ,
            )))
        }
        (3, Dimension::XYM) => {
            let xs = children[0]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            let ys = children[1]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            let ms = children[1]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            Ok(CoordBuffer::Separated(SeparatedCoordBuffer::new(
                [xs, ys, ms, ScalarBuffer::from(Vec::<f64>::new())],
                geoarrow_schema::Dimension::XYM,
            )))
        }
        (4, Dimension::XYZM) => {
            let xs = children[0]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            let ys = children[1]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            let zs = children[2]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            let ms = children[3]
                .to_primitive()?
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer();
            Ok(CoordBuffer::Separated(SeparatedCoordBuffer::new(
                [xs, ys, zs, ms],
                geoarrow_schema::Dimension::XYM,
            )))
        }
        _ => {
            vortex_bail!(
                "child count {} invalid for expected Dimension {:?}",
                children.len(),
                meta.dimension
            )
        }
    }
}

#[cfg(test)]
mod tests {
    //! Test that round trip through GeoArrow works as expected.
    use std::sync::Arc;

    use arrow_array::{ArrayRef, StructArray};
    use arrow_schema::Fields;
    use geoarrow::ArrayBase;
    use geoarrow::array::PointBuilder;
    use geoarrow_schema::{CoordType, Dimension};
    use vortex_array::arrow::{FromArrowArray, IntoArrowArray};
    use vortex_dtype::arrow::FromArrowType;
    use vortex_dtype::{DType, ExtDType};

    use crate::{OwnedGeoMetadata, OwnedGeometryType};

    #[test]
    fn test_geo_simple() {
        // Make a square
        let mut points =
            PointBuilder::new_with_options(Dimension::XY, CoordType::Separated, Default::default());
        points.push_coord(Some(&(0.0f64, 0.0f64)));
        let points = points.finish();
        let field_type = points.extension_field();
        let dtype = DType::from_arrow(field_type.as_ref());

        let owned_type: ExtDType = OwnedGeometryType::Point(OwnedGeoMetadata {
            dimension: crate::Dimension::XY,
            crs: None,
        })
        .into_ext_dtype(true.into());
        assert_eq!(dtype, DType::Extension(Arc::new(owned_type)));

        // round trip back to Arrow type.
        assert_eq!(&dtype.to_arrow().unwrap(), field_type.data_type());
    }

    #[test]
    fn test_arrow_extension_type() {
        let mut points =
            PointBuilder::new_with_options(Dimension::XY, CoordType::Separated, Default::default());
        points.push_coord(Some(&(0.0f64, 0.0f64)));
        let points = points.finish();

        let struct_array: ArrayRef = Arc::new(StructArray::new(
            Fields::from(vec![points.extension_field()]),
            vec![points.to_array_ref()],
            None,
        ));

        let imported = vortex_array::ArrayRef::from_arrow(struct_array.clone(), false);
        let exported = imported.into_arrow_preferred().unwrap();
        assert_eq!(exported.data_type(), struct_array.data_type());
    }
}
