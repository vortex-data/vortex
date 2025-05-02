use std::convert::Into;
use std::iter;
use std::sync::{Arc, LazyLock};

use arrow_buffer::ScalarBuffer;
use geoarrow::ArrayBase;
use geoarrow::array::{CoordBuffer, PointArray, SeparatedCoordBuffer};
use geoarrow_schema::Dimension;
use vortex::arrays::ExtensionArray;
use vortex::arrow::compute::ToArrowArgs;
use vortex::compute::{InvocationArgs, Kernel, Output};
use vortex::dtype::{DType, ExtDType, ExtID, Nullability, StructDType};
use vortex::error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::{Array, ArrayRef, ArrayVisitor, register_kernel};

pub static WKB_ID: &'static str = "geovortex.wkb";
pub static POINT_ID: &'static str = "geovortex.point";
pub static LINESTRING_ID: &'static str = "geovortex.linestring";
pub static POLYGON_ID: &'static str = "geovortex.polygon";

pub static WKB: LazyLock<ExtID> = LazyLock::new(|| ExtID::new(WKB_ID.into()));
/// Point is an N-dimensional point. It is stored using one value per variant here.
/// The actual point is based on something like a Struct of the components.
pub static POINT: LazyLock<ExtID> = LazyLock::new(|| ExtID::new(POINT_ID.into()));

pub static LINESTRING: LazyLock<ExtID> = LazyLock::new(|| ExtID::new(LINESTRING_ID.into()));

/// Polygon consists of a pair of "rings", which are lists of points defining the
/// exterior and interior boundaries of the polygon.
pub static POLYGON: LazyLock<ExtID> = LazyLock::new(|| ExtID::new(POLYGON_ID.into()));

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

    fn try_from(value: ArrayRef) -> VortexResult<Self> {
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
        match extension.id() {
            POINT => Ok(GeometryArray::Point(extension.clone())),
            LINESTRING => Ok(GeometryArray::LineString(extension.clone())),
            POLYGON => Ok(GeometryArray::Polygon(extension.clone())),
            WKB => Ok(GeometryArray::WKB(extension.clone())),
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
                        GeometryArray::Point(ext) => make_point_array(ext),
                        GeometryArray::LineString(_) => {}
                        GeometryArray::Polygon(_) => {}
                        GeometryArray::WKB(_) => {}
                    }
                } else {
                    Ok(None)
                }
            }
        }
    }
}

fn make_point_array(point_array: &ExtensionArray) -> VortexResult<arrow_array::ArrayRef> {
    // Based on the number of child arrays, we can convert all of these directly into Arrow types.
    let mut dimensions = point_array.storage().children();
    let dim = match dimensions.len() {
        2 => Dimension::XY,
        3 => Dimension::XYZ,
        n => {
            vortex_bail!("Expected 2 or 3 dimensions, got {}", n)
        }
    };

    let buffers: [ScalarBuffer<f64>; 4] = dimensions
        .into_iter()
        // Take the first 3 dimensions
        .take(3)
        // Pad the rest with empty ScalarBuffer, per the expectation of the PointArray constructor
        .chain(iter::repeat(ScalarBuffer::from(Vec::<f64>::new())))
        .take(4)
        .into();

    let nulls = point_array.validity_mask()?.to_null_buffer();

    Ok(PointArray::new(
        CoordBuffer::Separated(SeparatedCoordBuffer::new(buffers, dim)),
        nulls,
        // TODO(aduffy): include CRS information from metadata
        Arc::default(),
    )
    .into_array_ref())
}

// Register the ToArrow kernel that can handle the geospatial data types.
register_kernel!(ToGeoArrow);

fn make_point_dtype<const N: usize>(
    dims: [&ArrayRef; N],
    nullability: Nullability,
) -> Arc<ExtDType> {
    let point_struct = StructDType::from_iter(iter::zip(["x", "y", "z"], dims.iter()));

    Arc::new(ExtDType::new(
        POINT.clone(),
        Arc::new(DType::Struct(Arc::new(point_struct), nullability)),
        None,
    ))
}

pub type GeoArrowRef = Arc<dyn ArrayBase>;

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use geoarrow::array::NativeArrayDyn;
    use vortex::Array;
    use vortex::arrays::{ExtensionArray, StructArray};
    use vortex::arrow::compute::{to_arrow, to_arrow_preferred};
    use vortex::builders::{ArrayBuilder, PrimitiveBuilder};
    use vortex::dtype::{ExtDType, FieldNames, Nullability};
    use vortex::validity::Validity;

    use crate::types::{POINT, make_point_dtype};

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
                POINT.clone(),
                Arc::new(storage.dtype().clone()),
                None,
            )),
            storage,
        );

        // We need to preserve the field information as well for access to the schema type.
        let arrow_points = to_arrow_preferred(&ext).unwrap();

        NativeArrayDyn::from_arrow_array(arrow_points)
        geoarrow::array::from_arrow_array()

        assert_eq!(arrow_points.len(), 5);
    }
}
