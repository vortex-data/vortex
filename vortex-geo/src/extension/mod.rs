// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod geometry;
mod wkb;

use std::fmt::Display;

pub use geometry::*;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
pub use wkb::*;

/// Extension metadata that is common to all the geospatial extension types.
///
/// This carries the coordinate reference system (CRS) and, for native [`Geometry`] columns, the
/// [`GeometryKind`] held by the column. We may wish to add a field for edges interpretation in
/// the future similar to the GeoArrow standard.
#[derive(Clone, PartialEq, Eq, Hash, prost::Message)]
pub struct GeoMetadata {
    /// The coordinate reference system, or `None` for unreferenced geometry.
    #[prost(optional, string, tag = "1")]
    pub crs: Option<String>,

    /// The [`GeometryKind`] held by a native [`Geometry`] column.
    #[prost(enumeration = "GeometryKind", tag = "2")]
    pub geometry_type: i32,
}

impl GeoMetadata {
    /// The decoded [`GeometryKind`] of this metadata.
    pub fn kind(&self) -> VortexResult<GeometryKind> {
        GeometryKind::try_from(self.geometry_type)
            .map_err(|_| vortex_err!("unknown geometry kind {}", self.geometry_type))
    }
}

impl Display for GeoMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Geometry(")?;
        match GeometryKind::try_from(self.geometry_type) {
            Ok(GeometryKind::Unspecified) => {}
            Ok(kind) => write!(f, "{kind}, ")?,
            Err(_) => write!(f, "kind={}, ", self.geometry_type)?,
        }
        match self.crs.as_ref() {
            Some(crs) => write!(f, "crs={crs})"),
            None => write!(f, "unreferenced)"),
        }
    }
}

/// The kind of geometry held in a native [`Geometry`] column, matching the GeoArrow native
/// geometry types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, prost::Enumeration)]
#[repr(i32)]
pub enum GeometryKind {
    /// The wire default when no kind was set (e.g. WKB); rejected by [`Geometry`].
    Unspecified = 0,
    /// A single location.
    Point = 1,
    /// A sequence of locations connected into a line.
    LineString = 2,
    /// One outer ring with zero or more interior rings (holes).
    Polygon = 3,
    /// A collection of points.
    MultiPoint = 4,
    /// A collection of linestrings.
    MultiLineString = 5,
    /// A collection of polygons.
    MultiPolygon = 6,
}

impl Display for GeometryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            GeometryKind::Unspecified => "unspecified",
            GeometryKind::Point => "point",
            GeometryKind::LineString => "linestring",
            GeometryKind::Polygon => "polygon",
            GeometryKind::MultiPoint => "multipoint",
            GeometryKind::MultiLineString => "multilinestring",
            GeometryKind::MultiPolygon => "multipolygon",
        };
        write!(f, "{name}")
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use crate::extension::GeoMetadata;
    use crate::extension::GeometryKind;

    #[test]
    fn test_metadata() {
        let meta = GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
            ..Default::default()
        };

        assert_eq!(meta.to_string(), "Geometry(crs=EPSG:4326)");
        // round trip
        let bytes = meta.encode_to_vec();
        let decoded = GeoMetadata::decode(bytes.as_slice()).unwrap();
        assert_eq!(decoded, meta);
    }

    #[test]
    fn test_metadata_with_kind() {
        let meta = GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
            geometry_type: GeometryKind::Point as i32,
        };

        assert_eq!(meta.to_string(), "Geometry(point, crs=EPSG:4326)");
        let bytes = meta.encode_to_vec();
        let decoded = GeoMetadata::decode(bytes.as_slice()).unwrap();
        assert_eq!(decoded, meta);
        assert_eq!(decoded.kind().unwrap(), GeometryKind::Point);
    }
}
