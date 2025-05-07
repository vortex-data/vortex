#[allow(clippy::disallowed_types)]
use std::collections::HashMap;
use std::sync::Arc;

use arrow_buffer::ScalarBuffer;
use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
use arrow_schema::{DataType, Field};
use geoarrow::ArrayBase;
use geoarrow::array::{CoordBuffer, PointArray, SeparatedCoordBuffer};
use geoarrow_schema::Crs;
use vortex::arrays::ExtensionArray;
use vortex::arrow::ArrowArray;
use vortex::arrow::compute::{ToArrowArgs, ToArrowKernelRef};
use vortex::compute::{InvocationArgs, Kernel, Output};
use vortex::dtype::arrow::{ArrowMetadata, ArrowMetadataRef, ArrowToDType, ArrowToDTypeRef};
use vortex::dtype::{DType, ExtDType, register_extension_type};
use vortex::error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::{Array, ToCanonical, register_kernel};
use vortex_arcref::ArcRef;

use crate::array::GeometryArray;
use crate::{Dimension, GeoMetadata, GeometryType, OwnedGeometryType};

/// Kernel that allows converting a Vortex extension array with geometry type into a GeoArrow
/// array layout.
#[derive(Debug)]
pub struct ToGeoArrow;

impl Kernel for ToGeoArrow {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let ToArrowArgs { array, .. } = ToArrowArgs::try_from(args)?;
        match array.as_any().downcast_ref::<ExtensionArray>() {
            None => Ok(None),
            Some(ext_array) => {
                let _ = GeometryArray::try_from(ext_array)?;
                if let Ok(geometry_array) = GeometryArray::try_from(ext_array) {
                    // based on the particular geometry, encode into GeoArrow type.
                    let array = match geometry_array {
                        GeometryArray::Point(point_array, metadata) => {
                            let coords = coordinate_buffer(point_array, &metadata)?;
                            let nulls = point_array.validity_mask()?.to_null_buffer();
                            let crs = match metadata.crs {
                                None => Crs::default(),
                                Some(wkt) => Crs::from_wkt2_2019(wkt.to_string()),
                            };
                            PointArray::new(
                                coords,
                                nulls,
                                Arc::new(geoarrow_schema::Metadata::new(crs, None)),
                            )
                            .into_array_ref()
                        }
                        GeometryArray::LineString(..) => todo!(),
                        GeometryArray::Polygon(..) => todo!(),
                        GeometryArray::WKB(..) => todo!(),
                    };

                    Ok(Some(Output::Array(
                        ArrowArray::new(array, ext_array.dtype().nullability()).into_array(),
                    )))
                } else {
                    println!("GeometryArray::try_from failed");
                    Ok(None)
                }
            }
        }
    }
}
register_kernel!(ToArrowKernelRef(ArcRef::new_ref(&ToGeoArrow)));

/// ZST that holds the VTable for building a `DType` from an Arrow [`Field`] type.
#[derive(Debug)]
pub struct GeoArrowConversion;

static GEOARROW_POINT: &str = "geoarrow.point";
static GEOARROW_LINESTRING: &str = "geoarrow.linestring";
static GEOARROW_POLYGON: &str = "geoarrow.polygon";
static GEOARROW_WKB: &str = "geoarrow.wkb";

// Conversion between Vortex geospatial extension types and GeoArrow extension types.
impl ArrowToDType for GeoArrowConversion {
    fn can_convert(&self, field: &Field) -> bool {
        // The field needs to support all of our types
        if let Some(ext_type) = field.extension_type_name() {
            if ext_type == GEOARROW_POINT
                || ext_type == GEOARROW_LINESTRING
                || ext_type == GEOARROW_POLYGON
                || ext_type == GEOARROW_WKB
            {
                return true;
            }
        }
        false
    }

    fn to_vortex(&self, field: &Field) -> VortexResult<DType> {
        let Some(ext_type) = field.extension_type_name() else {
            vortex_bail!("field does not have an extension type")
        };

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
            x if x == GEOARROW_POINT => {
                let geoarrow_meta = geoarrow_schema::Metadata::try_from(field)?;
                // We only accept Separated format.
                let DataType::Struct(fields) = field.data_type() else {
                    vortex_bail!("Only Separated format is supported for Point geometry")
                };

                OwnedGeometryType::Point(
                    dim_from_fields!(fields),
                    geoarrow_meta.crs().crs_value().map(|v| v.to_string()),
                )
                .into_ext_dtype(field.is_nullable().into())
            }
            x if x == GEOARROW_LINESTRING => {
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
                OwnedGeometryType::LineString(
                    dim_from_fields!(fields),
                    geoarrow_meta.crs().crs_value().map(|v| v.to_string()),
                )
                .into_ext_dtype(field.is_nullable().into())
            }
            x if x == GEOARROW_POLYGON => {
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
                OwnedGeometryType::Polygon(
                    dim_from_fields!(fields),
                    geoarrow_meta.crs().crs_value().map(|v| v.to_string()),
                )
                .into_ext_dtype(field.is_nullable().into())
            }
            x if x == GEOARROW_WKB => {
                let geoarrow_meta = geoarrow_schema::Metadata::try_from(field)?;
                if field.data_type() != &DataType::Binary {
                    vortex_bail!(
                        "WKB geometry Arrow type must be Binary, was {:?}",
                        field.data_type()
                    )
                }
                OwnedGeometryType::WKB(
                    Dimension::default(),
                    geoarrow_meta.crs().crs_value().map(|v| v.to_string()),
                )
                .into_ext_dtype(field.is_nullable().into())
            }
            _ => vortex_bail!("extension type {} not supported", ext_type),
        };

        Ok(DType::Extension(Arc::new(ext_dtype)))
    }
}

impl ArrowMetadata for GeoArrowConversion {
    #[allow(clippy::disallowed_types)]
    fn arrow_metadata(&self, vortex_extension_type: &ExtDType) -> Option<HashMap<String, String>> {
        if let Ok(geometry) = GeometryType::try_from(vortex_extension_type) {
            let mut extension_metadata = HashMap::new();
            let ext_type_name = match geometry {
                GeometryType::Point(_) => GEOARROW_POINT.to_string(),
                GeometryType::LineString(_) => GEOARROW_LINESTRING.to_string(),
                GeometryType::Polygon(_) => GEOARROW_POLYGON.to_string(),
                GeometryType::WKB(_) => GEOARROW_WKB.to_string(),
            };
            extension_metadata.insert(EXTENSION_TYPE_NAME_KEY.to_string(), ext_type_name);

            match geometry {
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
                    }
                }
            };

            Some(extension_metadata)
        } else {
            None
        }
    }
}

register_extension_type!(ArrowToDTypeRef(ArcRef::new_ref(&GeoArrowConversion)));
register_extension_type!(ArrowMetadataRef(ArcRef::new_ref(&GeoArrowConversion)));

/// Unpack the geoarrow CoordBuffer. Errors if the dimensions specified in the metadata do not
/// match the actual encoding.
fn coordinate_buffer(array: &ExtensionArray, meta: &GeoMetadata) -> VortexResult<CoordBuffer> {
    let children = array.storage().children();
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
    use vortex::arrow::{FromArrowArray, IntoArrowArray};
    use vortex::dtype::arrow::FromArrowType;
    use vortex::dtype::{DType, ExtDType};

    use crate::OwnedGeometryType;

    #[test]
    fn test_geo_simple() {
        // Make a square
        let mut points =
            PointBuilder::new_with_options(Dimension::XY, CoordType::Separated, Default::default());
        points.push_coord(Some(&(0.0f64, 0.0f64)));
        let points = points.finish();
        let field_type = points.extension_field();
        let dtype = DType::from_arrow(field_type.as_ref());

        let owned_type: ExtDType =
            OwnedGeometryType::Point(crate::Dimension::XY, None).into_ext_dtype(false.into());
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

        let imported = vortex::ArrayRef::from_arrow(struct_array.clone(), false);
        let exported = imported.into_arrow_preferred().unwrap();
        assert_eq!(exported.data_type(), struct_array.data_type());
    }
}
