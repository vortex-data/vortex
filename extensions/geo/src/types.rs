use std::convert::Into;
use std::sync::Arc;

use arrow_schema::Field;
use geoarrow_schema::{CoordType, Crs, LineStringType, PointType, PolygonType, WkbType};
use vortex_dtype::{DType, ExtDType, ExtMetadata, Nullability, PType, StructDType};
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedGeoMetadata {
    pub dimension: Dimension,
    pub crs: Option<String>,
}

impl From<OwnedGeoMetadata> for geoarrow_schema::Metadata {
    fn from(value: OwnedGeoMetadata) -> Self {
        let crs = value
            .crs
            .as_ref()
            .map(|crs| match serde_json::from_str(crs) {
                Ok(value) => Crs::from_projjson(value),
                Err(_) => Crs::from_unknown_crs_type(crs.to_string()),
            })
            .unwrap_or_default();

        Self::new(crs, None)
    }
}

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

impl GeometryType<'_> {
    pub fn to_owned(&self) -> OwnedGeometryType {
        macro_rules! owned_meta {
            ($meta:expr) => {
                OwnedGeoMetadata {
                    dimension: $meta.dimension,
                    crs: $meta.crs.map(String::from),
                }
            };
        }

        match self {
            GeometryType::Point(meta) => OwnedGeometryType::Point(owned_meta!(meta)),
            GeometryType::LineString(meta) => OwnedGeometryType::LineString(owned_meta!(meta)),
            GeometryType::Polygon(meta) => OwnedGeometryType::Polygon(owned_meta!(meta)),
            GeometryType::WKB(meta) => OwnedGeometryType::WKB(owned_meta!(meta)),
        }
    }
}

/// An owned geometry type.
pub enum OwnedGeometryType {
    Point(OwnedGeoMetadata),
    LineString(OwnedGeoMetadata),
    Polygon(OwnedGeoMetadata),
    WKB(OwnedGeoMetadata),
}

impl OwnedGeometryType {
    /// Serialize the metadata the way it will be stored in [`ExtMetadata`].
    pub fn metadata(&self) -> Vec<u8> {
        match self {
            OwnedGeometryType::Point(OwnedGeoMetadata { dimension, crs })
            | OwnedGeometryType::LineString(OwnedGeoMetadata { dimension, crs })
            | OwnedGeometryType::Polygon(OwnedGeoMetadata { dimension, crs })
            | OwnedGeometryType::WKB(OwnedGeoMetadata { dimension, crs }) => {
                let mut bytes = vec![*dimension as u8];
                if let Some(crs) = crs {
                    bytes.extend(crs.as_bytes());
                }
                bytes
            }
        }
    }
}

impl OwnedGeometryType {
    pub fn into_ext_dtype(self, nullability: Nullability) -> ExtDType {
        let ext_meta = self.metadata();

        match self {
            OwnedGeometryType::Point(OwnedGeoMetadata { dimension, .. }) => ExtDType::new(
                POINT_ID.clone(),
                Arc::new(DType::Struct(Arc::new(point_dtype(dimension)), nullability)),
                Some(ExtMetadata::new(ext_meta.into())),
            ),
            OwnedGeometryType::LineString(OwnedGeoMetadata { dimension, .. }) => {
                let storage_dtype = DType::List(
                    Arc::new(DType::Struct(
                        Arc::new(point_dtype(dimension)),
                        false.into(),
                    )),
                    nullability,
                );
                ExtDType::new(
                    LINESTRING_ID.clone(),
                    Arc::new(storage_dtype),
                    Some(ExtMetadata::new(ext_meta.into())),
                )
            }
            OwnedGeometryType::Polygon(OwnedGeoMetadata { dimension, .. }) => {
                let storage_dtype = DType::List(
                    Arc::new(DType::List(
                        Arc::new(DType::Struct(
                            Arc::new(point_dtype(dimension)),
                            false.into(),
                        )),
                        false.into(),
                    )),
                    nullability,
                );
                ExtDType::new(
                    POLYGON_ID.clone(),
                    Arc::new(storage_dtype),
                    Some(ExtMetadata::new(ext_meta.into())),
                )
            }
            OwnedGeometryType::WKB(..) => ExtDType::new(
                WKB_ID.clone(),
                Arc::new(DType::Binary(nullability)),
                Some(ExtMetadata::new(ext_meta.into())),
            ),
        }
    }

    pub fn into_arrow_field(self, nullability: Nullability) -> Field {
        match self {
            OwnedGeometryType::Point(meta) => PointType::new(
                CoordType::Separated,
                meta.dimension.into(),
                Arc::new(meta.into()),
            )
            .to_field("point_type", nullability.into()),
            OwnedGeometryType::LineString(meta) => LineStringType::new(
                CoordType::Separated,
                meta.dimension.into(),
                Arc::new(meta.into()),
            )
            .to_field("line_string_type", nullability.into()),
            OwnedGeometryType::Polygon(meta) => PolygonType::new(
                CoordType::Separated,
                meta.dimension.into(),
                Arc::new(meta.into()),
            )
            .to_field("polygon_type", nullability.into()),
            OwnedGeometryType::WKB(meta) => {
                WkbType::new(Arc::new(meta.into())).to_field("wkb_type", nullability.into(), false)
            }
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
            let crs = match bytes.len() {
                2.. => Some(validate_crs(&bytes[1..])?),
                _ => None,
            };

            Ok(Self { dimension, crs })
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
