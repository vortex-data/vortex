use std::convert::Into;
use std::sync::Arc;

use vortex::dtype::{DType, ExtDType, ExtMetadata, Nullability, PType, StructDType};
use vortex::error::{VortexError, VortexResult, vortex_bail, vortex_err};

use crate::{LINESTRING_ID, POINT_ID, POLYGON_ID, WKB_ID};

/// Dimensions in the coordinate buffers for data.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Dimension {
    /// Two dimensional coordinates. This is the default if there is no metadata provided for
    /// the geometry.
    #[default]
    XY = 1,
    /// Three-dimensional geometry. Commonly the Z coordinate will be height above the ellipsoid,
    /// or height above sea level, determined by the CRS.
    XYZ = 2,
    /// Two-dimensional with an additional non-spatial measure (e.g. time).
    XYM = 3,
    /// Three-dimensional with an additional non-spatial measure (e.g. time).
    XYZM = 4,
}

/// Zero-allocation container for geometry metadata
pub struct GeoMetadata<'a> {
    pub dimension: Dimension,
    pub crs: Option<&'a str>,
}

/// Owned version of a [`GeoMetadata`].
#[derive(Debug, Clone)]
pub struct OwnedGeoMetadata {
    pub dimension: Dimension,
    pub crs: Option<String>,
}

// impl<'a> ToOwned for GeoMetadata<'a> {
//     type Owned = OwnedGeoMetadata;
//
//     fn to_owned(&self) -> Self::Owned {
//         OwnedGeoMetadata {
//             dimension: self.dimension,
//             crs: self.crs.map(|x| x.to_owned()),
//         }
//     }
// }

// TODO(aduffy): add more geometry types like MultiPolygon.
/// Zero-copy view of an `ExtDType` as one of the GeoVortex builtin geometry types.
///
/// The [owned][ToOwned] version of this is an [`OwnedGeometryType`].
pub enum GeometryType<'a> {
    Point(GeoMetadata<'a>),
    LineString(GeoMetadata<'a>),
    Polygon(GeoMetadata<'a>),
    WKB(GeoMetadata<'a>),
}

// impl ToOwned for GeometryType<'_> {
//     type Owned = OwnedGeometryType;
//
//     fn to_owned(&self) -> Self::Owned {
//         match self {
//             GeometryType::Point(GeoMetadata { dimension, crs }) => {
//                 OwnedGeometryType::Point(*dimension, crs.map(|x| x.to_owned()))
//             }
//             GeometryType::LineString(GeoMetadata { dimension, crs }) => {
//                 OwnedGeometryType::LineString(*dimension, crs.map(|x| x.to_owned()))
//             }
//             GeometryType::Polygon(GeoMetadata { dimension, crs }) => {
//                 OwnedGeometryType::Polygon(*dimension, crs.map(|x| x.to_owned()))
//             }
//             GeometryType::WKB(GeoMetadata { dimension, crs }) => {
//                 OwnedGeometryType::WKB(*dimension, crs.map(|x| x.to_owned()))
//             }
//         }
//     }
// }

/// An owned geometry type.
pub enum OwnedGeometryType {
    Point(Dimension, Option<String>),
    LineString(Dimension, Option<String>),
    Polygon(Dimension, Option<String>),
    WKB(Dimension, Option<String>),
}

impl OwnedGeometryType {
    /// Serialize the metadata the way it will be stored in [`ExtMetadata`].
    pub fn metadata(&self) -> Vec<u8> {
        match self {
            OwnedGeometryType::Point(dimension, crs)
            | OwnedGeometryType::LineString(dimension, crs)
            | OwnedGeometryType::Polygon(dimension, crs)
            | OwnedGeometryType::WKB(dimension, crs) => {
                let mut bytes = vec![*dimension as u8];
                if let Some(crs) = crs {
                    bytes.extend(crs.as_bytes());
                }
                bytes
            }
        }
    }
}

// We need to provide nullability info.
impl From<OwnedGeometryType> for ExtDType {
    fn from(value: OwnedGeometryType) -> Self {
        let ext_meta = value.metadata();

        match value {
            OwnedGeometryType::Point(dimension, ..) => ExtDType::new(
                POINT_ID.clone(),
                Arc::new(DType::Struct(
                    Arc::new(point_dtype(dimension)),
                    false.into(),
                )),
                Some(ExtMetadata::new(ext_meta.into())),
            ),
            OwnedGeometryType::LineString(dimension, ..) => {
                let storage_dtype = DType::List(
                    Arc::new(DType::Struct(
                        Arc::new(point_dtype(dimension)),
                        false.into(),
                    )),
                    false.into(),
                );
                ExtDType::new(
                    LINESTRING_ID.clone(),
                    Arc::new(storage_dtype),
                    Some(ExtMetadata::new(ext_meta.into())),
                )
            }
            OwnedGeometryType::Polygon(dimension, ..) => {
                let storage_dtype = DType::List(
                    Arc::new(DType::List(
                        Arc::new(DType::Struct(
                            Arc::new(point_dtype(dimension)),
                            false.into(),
                        )),
                        false.into(),
                    )),
                    false.into(),
                );
                ExtDType::new(
                    POLYGON_ID.clone(),
                    Arc::new(storage_dtype),
                    Some(ExtMetadata::new(ext_meta.into())),
                )
            }
            OwnedGeometryType::WKB(..) => ExtDType::new(
                WKB_ID.clone(),
                Arc::new(DType::Binary(false.into())),
                Some(ExtMetadata::new(ext_meta.into())),
            ),
        }
    }
}

fn point_dtype(dimension: Dimension) -> StructDType {
    match dimension {
        Dimension::XY => StructDType::from_iter([
            ("x", DType::Primitive(PType::F64, Nullability::NonNullable)),
            ("y", DType::Primitive(PType::F64, Nullability::NonNullable)),
        ]),
        Dimension::XYZ => StructDType::from_iter([
            ("x", DType::Primitive(PType::F64, Nullability::NonNullable)),
            ("y", DType::Primitive(PType::F64, Nullability::NonNullable)),
            ("z", DType::Primitive(PType::F64, Nullability::NonNullable)),
        ]),
        Dimension::XYM => StructDType::from_iter([
            ("x", DType::Primitive(PType::F64, Nullability::NonNullable)),
            ("y", DType::Primitive(PType::F64, Nullability::NonNullable)),
            ("m", DType::Primitive(PType::F64, Nullability::NonNullable)),
        ]),
        Dimension::XYZM => StructDType::from_iter([
            ("x", DType::Primitive(PType::F64, Nullability::NonNullable)),
            ("y", DType::Primitive(PType::F64, Nullability::NonNullable)),
            ("z", DType::Primitive(PType::F64, Nullability::NonNullable)),
            ("m", DType::Primitive(PType::F64, Nullability::NonNullable)),
        ]),
    }
}

impl<'a> TryFrom<&'a ExtMetadata> for GeoMetadata<'a> {
    type Error = VortexError;

    fn try_from(metadata: &'a ExtMetadata) -> VortexResult<Self> {
        let bytes = metadata.as_ref();
        if bytes.is_empty() {
            vortex_bail!("If metadata is provided must not be empty");
        } else {
            let dimension = match bytes[0] {
                x if x == Dimension::XY as u8 => Dimension::XY,
                x if x == Dimension::XYZ as u8 => Dimension::XYZ,
                x if x == Dimension::XYM as u8 => Dimension::XYM,
                x if x == Dimension::XYZM as u8 => Dimension::XYZM,
                _ => vortex_bail!("Invalid dimension: {:?}", bytes),
            };
            let crs = validate_crs(&bytes[1..])?;

            Ok(Self {
                dimension,
                crs: Some(crs),
            })
        }
    }
}

// Validate and reinterpret-cast an `ExtDType` to a `GeometryType` in-place.
impl<'a> TryFrom<&'a ExtDType> for GeometryType<'a> {
    type Error = VortexError;

    fn try_from(ext_dtype: &'a ExtDType) -> VortexResult<Self> {
        // Metadata must be provided.
        let Some(metadata) = ext_dtype.metadata() else {
            vortex_bail!("Metadata must be provided for geometry types");
        };
        match ext_dtype.id().as_ref() {
            x if x == POINT_ID.as_ref() => {
                Ok(GeometryType::Point(GeoMetadata::try_from(metadata)?))
            }
            x if x == LINESTRING_ID.as_ref() => {
                Ok(GeometryType::LineString(GeoMetadata::try_from(metadata)?))
            }
            x if x == POLYGON_ID.as_ref() => {
                Ok(GeometryType::Polygon(GeoMetadata::try_from(metadata)?))
            }
            x if x == WKB_ID.as_ref() => Ok(GeometryType::WKB(GeoMetadata::try_from(metadata)?)),
            _ => Err(vortex_err!("Unsupported geometry type {}", ext_dtype.id())),
        }
    }
}

fn validate_crs(bytes: &[u8]) -> VortexResult<&str> {
    if bytes.is_empty() {
        vortex_bail!("WKT CRS must not be empty");
    }

    match std::str::from_utf8(bytes) {
        // TODO(aduffy): validate that the UTF-8 string is also valid WKT
        Ok(s) => Ok(s),
        Err(_) => vortex_bail!("Invalid CRS string: {:?}", bytes),
    }
}
