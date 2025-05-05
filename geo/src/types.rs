use std::convert::Into;
use std::iter;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::__private::DataType;
use arrow_buffer::ScalarBuffer;
use geoarrow::ArrayBase;
use geoarrow::array::{CoordBuffer, PointArray, SeparatedCoordBuffer};
use geoarrow_schema::Dimension;
use vortex::arcref::ArcRef;
use vortex::arrays::ExtensionArray;
use vortex::arrow::ArrowArray;
use vortex::arrow::compute::{ToArrowArgs, ToArrowKernelRef};
use vortex::compute::{InvocationArgs, Kernel, Output};
use vortex::dtype::arrow::DTypeConversion;
use vortex::dtype::{ExtDType, ExtID};
use vortex::error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::{Array, ArrayRef, ArrayVisitor, ToCanonical, register_kernel};

pub static WKB_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.wkb"));
/// Point is an N-dimensional point. It is stored using one value per variant here.
/// The actual point is based on something like a Struct of the components.
pub static POINT_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.point"));

pub static LINESTRING_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.linestring"));

/// Polygon consists of a pair of "rings", which are lists of points defining the
/// exterior and interior boundaries of the polygon.
pub static POLYGON_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.polygon"));

pub enum GeometryArray {
    Point(ExtensionArray),
    LineString(ExtensionArray),
    Polygon(ExtensionArray),
    WKB(ExtensionArray),
}

// Based on the individual Arrow types here...I think?
// How do we plug this into PyVortex if we don't have support in core?
// We need a separated "geo" plugin for these types.

impl TryFrom<&dyn Array> for GeometryArray {
    type Error = VortexError;

    fn try_from(value: &dyn Array) -> VortexResult<Self> {
        let extension = value
            .as_any()
            .downcast_ref::<ExtensionArray>()
            .ok_or_else(|| vortex_err!("ExtensionArray expected for GeometryArray"))?;

        GeometryArray::try_from(extension)
    }
}

impl TryFrom<ArrayRef> for GeometryArray {
    type Error = VortexError;

    fn try_from(value: ArrayRef) -> Result<Self, Self::Error> {
        GeometryArray::try_from(value.as_ref())
    }
}

impl TryFrom<&ExtensionArray> for GeometryArray {
    type Error = VortexError;

    fn try_from(extension: &ExtensionArray) -> Result<Self, Self::Error> {
        match extension.id().as_ref() {
            x if x == POINT_ID.as_ref() => Ok(GeometryArray::Point(extension.clone())),
            x if x == LINESTRING_ID.as_ref() => Ok(GeometryArray::LineString(extension.clone())),
            x if x == POLYGON_ID.as_ref() => Ok(GeometryArray::Polygon(extension.clone())),
            x if x == WKB_ID.as_ref() => Ok(GeometryArray::WKB(extension.clone())),
            _ => Err(vortex_err!("Unsupported geometry type {}", extension.id())),
        }
    }
}

/// Kernel that allows converting a Vortex extension array with geometry type into a GeoArrow
/// compatible encoding.
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
                    match geometry_array {
                        GeometryArray::Point(ext) => Ok(Some(Output::Array(
                            ArrowArray::new(make_point_array(&ext)?, array.dtype().nullability())
                                .into_array(),
                        ))),
                        GeometryArray::LineString(_) => todo!(),
                        GeometryArray::Polygon(_) => todo!(),
                        GeometryArray::WKB(_) => todo!(),
                    }
                } else {
                    Ok(None)
                }
            }
        }
    }
}
register_kernel!(ToArrowKernelRef(ArcRef::new_ref(&ToGeoArrow)));

#[derive(Debug)]
pub struct GeoArrowConversion;

impl DTypeConversion for GeoArrowConversion {
    fn can_convert_arrow(&self, data_type: &DataType) -> bool {
        // Only if the DType includes a field type
        todo!()
    }

    fn can_convert_vortex(&self, ext_dtype: &ExtDType) -> bool {
        // This converter only matches the geometry types defined by this crate
        ext_dtype.id().as_ref() == POINT_ID.as_ref()
            || ext_dtype.id().as_ref() == LINESTRING_ID.as_ref()
            || ext_dtype.id().as_ref() == POLYGON_ID.as_ref()
            || ext_dtype.id().as_ref() == WKB_ID.as_ref()
    }

    fn to_arrow(&self, dtype: &ExtDType) -> VortexResult<DataType> {
        // Convert into a DataType. This actually requires us to return a full Field type instead
        todo!()
    }

    fn to_vortex(&self, data_type: DataType) -> VortexResult<Box<Self>> {
        todo!()
    }
}

fn make_point_array(point_array: &ExtensionArray) -> VortexResult<arrow_array::ArrayRef> {
    // Based on the number of child arrays, we can convert all of these directly into Arrow types.
    let dimensions = point_array.storage().children();
    let dim = match dimensions.len() {
        2 => Dimension::XY,
        3 => Dimension::XYZ,
        n => {
            vortex_bail!("Expected 2 or 3 dimensions, got {}", n)
        }
    };

    let buffers: Vec<ScalarBuffer<f64>> = dimensions
        .into_iter()
        // Take the first 3 dimensions
        .take(3)
        .map(|x| {
            x.to_primitive()
                .vortex_expect("to_primitive")
                .into_buffer::<f64>()
                .into_arrow_scalar_buffer()
        })
        // Pad the rest with empty ScalarBuffer, per the expectation of the PointArray constructor
        .chain(iter::repeat(ScalarBuffer::from(Vec::<f64>::new())))
        .take(4)
        .collect::<Vec<_>>();

    let nulls = point_array.validity_mask()?.to_null_buffer();

    Ok(PointArray::new(
        CoordBuffer::Separated(SeparatedCoordBuffer::new(
            [
                buffers[0].clone(),
                buffers[1].clone(),
                buffers[2].clone(),
                buffers[3].clone(),
            ],
            dim,
        )),
        nulls,
        // TODO(aduffy): include CRS information from metadata
        Arc::default(),
    )
    .into_array_ref())
}

pub type GeoArrowRef = Arc<dyn ArrayBase>;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex::Array;
    use vortex::arrays::{ExtensionArray, StructArray};
    use vortex::arrow::compute::to_arrow_preferred;
    use vortex::builders::{ArrayBuilder, PrimitiveBuilder};
    use vortex::dtype::{ExtDType, Nullability};
    use vortex::validity::Validity;

    use crate::types::POINT_ID;

    #[test]
    fn test_convert_to_arrow() {
        let mut xs = PrimitiveBuilder::<f64>::new(Nullability::NonNullable);
        xs.append_value(0.0);
        xs.append_value(1.0);
        xs.append_value(1.0);
        xs.append_value(0.0);
        xs.append_value(0.0);
        let xs = xs.finish();

        let mut ys = PrimitiveBuilder::<f64>::new(Nullability::NonNullable);
        ys.append_value(0.0);
        ys.append_value(0.0);
        ys.append_value(1.0);
        ys.append_value(1.0);
        ys.append_value(0.0);
        let ys = ys.finish();

        let storage = StructArray::try_new(
            ["x".into(), "y".into()].into(),
            vec![xs, ys],
            5,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let ext = ExtensionArray::new(
            Arc::new(ExtDType::new(
                POINT_ID.clone(),
                Arc::new(storage.dtype().clone()),
                None,
            )),
            storage,
        );

        // We need to preserve the field information as well for access to the schema type.
        let arrow_points = to_arrow_preferred(&ext).unwrap();

        assert_eq!(arrow_points.len(), 5);
    }
}
