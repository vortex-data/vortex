//! `ExtensionArray` wrapper for arrays that hold geospatial data types.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::arrays::ExtensionArray;
use vortex_array::variants::ExtensionArrayTrait;
use vortex_dtype::{DType, PType, StructDType};
use vortex_error::{VortexError, VortexResult, vortex_assert, vortex_bail};

use crate::{Dimension, GeoMetadata, GeometryType, OwnedGeoMetadata, OwnedGeometryType};

/// Holder for what is known to be one of the blessed extension array types.
pub enum GeometryArray<'a> {
    Point(&'a ExtensionArray, GeoMetadata<'a>),
    LineString(&'a ExtensionArray, GeoMetadata<'a>),
    Polygon(&'a ExtensionArray, GeoMetadata<'a>),
    #[allow(clippy::upper_case_acronyms)]
    WKB(&'a ExtensionArray, GeoMetadata<'a>),
}

impl<'a> TryFrom<&'a ExtensionArray> for GeometryArray<'a> {
    type Error = VortexError;

    fn try_from(value: &'a ExtensionArray) -> VortexResult<Self> {
        let geometry_type = GeometryType::try_from(value.ext_dtype().as_ref())?;
        Ok(match geometry_type {
            GeometryType::Point(meta) => Self::Point(value, meta),
            GeometryType::Polygon(meta) => Self::Polygon(value, meta),
            GeometryType::WKB(meta) => Self::WKB(value, meta),
            GeometryType::LineString(meta) => Self::LineString(value, meta),
        })
    }
}

#[derive(Debug, Clone)]
pub struct PointArray {
    inner: ExtensionArray,
    metadata: OwnedGeoMetadata,
}

impl PointArray {
    /// Wrap an existing array as a `geovortex.point` extension array.
    ///
    /// ## Error checking
    ///
    /// The provided `points` storage array must be a struct-typed array with f64 columns named based
    /// on their dimensions.
    pub fn try_new(points: ArrayRef, metadata: OwnedGeoMetadata) -> VortexResult<Self> {
        let DType::Struct(schema, _) = points.dtype() else {
            vortex_bail!("points must be Struct typed, was {}", points.dtype())
        };

        validate_coord_schema(schema, metadata.dimension)?;
        let point_type =
            OwnedGeometryType::Point(metadata.clone()).into_ext_dtype(points.dtype().nullability());
        let inner = ExtensionArray::new(Arc::new(point_type), points);
        Ok(Self { inner, metadata })
    }

    /// Deconstruct the `PointsArray` wrapper into the storage array and the parsed extension metadata.
    pub fn into_parts(self) -> (ExtensionArray, OwnedGeoMetadata) {
        (self.inner, self.metadata)
    }
}

pub fn validate_coord_schema(schema: &StructDType, dimensions: Dimension) -> VortexResult<()> {
    match dimensions {
        Dimension::XY => {
            vortex_assert!(schema.nfields() == 2);
            vortex_assert!(schema.field_name(0)?.as_ref().eq("x"));
            vortex_assert!(schema.field_name(1)?.as_ref().eq("y"));
            schema
                .fields()
                .all(|field| field.eq_ignore_nullability(PType::F64.into()));
        }
        Dimension::XYZ => {
            vortex_assert!(schema.nfields() == 3);
            vortex_assert!(schema.field_name(0)?.as_ref().eq("x"));
            vortex_assert!(schema.field_name(1)?.as_ref().eq("y"));
            vortex_assert!(schema.field_name(2)?.as_ref().eq("z"));
            schema
                .fields()
                .all(|field| field.eq_ignore_nullability(PType::F64.into()));
        }
        Dimension::XYM => {
            vortex_assert!(schema.nfields() == 3);
            vortex_assert!(schema.field_name(0)?.as_ref().eq("x"));
            vortex_assert!(schema.field_name(1)?.as_ref().eq("y"));
            vortex_assert!(schema.field_name(2)?.as_ref().eq("m"));
            schema
                .fields()
                .all(|field| field.eq_ignore_nullability(PType::F64.into()));
        }
        Dimension::XYZM => {
            vortex_assert!(schema.nfields() == 3);
            vortex_assert!(schema.field_name(0)?.as_ref().eq("x"));
            vortex_assert!(schema.field_name(1)?.as_ref().eq("y"));
            vortex_assert!(schema.field_name(2)?.as_ref().eq("z"));
            vortex_assert!(schema.field_name(3)?.as_ref().eq("m"));
            schema
                .fields()
                .all(|field| field.eq_ignore_nullability(PType::F64.into()));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_array::Array;
    use vortex_array::arrays::{PrimitiveArray, StructArray};

    use super::PointArray;
    use crate::{OwnedGeoMetadata, POINT_ID};

    #[test]
    fn test_points() {
        let values = StructArray::from_fields(&[
            (
                "x",
                PrimitiveArray::from_iter([0f64, 0f64, 0f64]).into_array(),
            ),
            (
                "y",
                PrimitiveArray::from_iter([1f64, 2f64, 3f64]).into_array(),
            ),
            (
                "z",
                PrimitiveArray::from_iter([4f64, 5f64, 6f64]).into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        let points_array = PointArray::try_new(
            values,
            OwnedGeoMetadata {
                crs: None,
                dimension: crate::Dimension::XYZ,
            },
        )
        .unwrap();

        let (ext, meta) = points_array.into_parts();
        assert_eq!(
            meta,
            OwnedGeoMetadata {
                crs: None,
                dimension: crate::Dimension::XYZ,
            }
        );
        assert_eq!(ext.id(), &*POINT_ID);
    }

    #[test]
    fn test_polygon() {
        // Create two lists of points: one defining an interior and exterior.
    }
}
