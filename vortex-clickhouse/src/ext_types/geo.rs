// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Geo Extension Type for ClickHouse Point/LineString/Ring/Polygon/MultiLineString/MultiPolygon.
//!
//! ClickHouse supports geographic types that are stored as nested Tuple/Array structures:
//! - Point = Tuple(Float64, Float64)
//! - Ring = Array(Point)
//! - LineString = Array(Point)
//! - Polygon = Array(Ring)
//! - MultiLineString = Array(LineString)
//! - MultiPolygon = Array(Polygon)
//!
//! In Vortex, these are stored as WKB-encoded binary strings. The C++ side handles
//! conversion between ClickHouse GEO columns and WKB binary format.
//!
//! This extension type preserves the GEO type name through the Vortex file format
//! so the read side can reconstruct the correct ClickHouse GEO type.

use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use vortex::dtype::extension::{ExtDType, ExtDTypeVTable, ExtID};
use vortex::dtype::{DType, Nullability};
use vortex::error::{VortexResult, vortex_bail};

/// The extension type ID for ClickHouse Geo types.
pub const GEO_EXT_ID: &str = "clickhouse.geo";

/// The concrete GEO types supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GeoType {
    Point = 0,
    LineString = 1,
    Ring = 2,
    Polygon = 3,
    MultiLineString = 4,
    MultiPolygon = 5,
}

impl GeoType {
    /// Returns the ClickHouse type name.
    pub const fn clickhouse_type_name(&self) -> &'static str {
        match self {
            GeoType::Point => "Point",
            GeoType::LineString => "LineString",
            GeoType::Ring => "Ring",
            GeoType::Polygon => "Polygon",
            GeoType::MultiLineString => "MultiLineString",
            GeoType::MultiPolygon => "MultiPolygon",
        }
    }

    /// Parse from ClickHouse type name.
    pub fn from_clickhouse_type(name: &str) -> Option<Self> {
        match name {
            "Point" => Some(GeoType::Point),
            "LineString" => Some(GeoType::LineString),
            "Ring" => Some(GeoType::Ring),
            "Polygon" => Some(GeoType::Polygon),
            "MultiLineString" => Some(GeoType::MultiLineString),
            "MultiPolygon" => Some(GeoType::MultiPolygon),
            _ => None,
        }
    }
}

impl TryFrom<u8> for GeoType {
    type Error = vortex::error::VortexError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GeoType::Point),
            1 => Ok(GeoType::LineString),
            2 => Ok(GeoType::Ring),
            3 => Ok(GeoType::Polygon),
            4 => Ok(GeoType::MultiLineString),
            5 => Ok(GeoType::MultiPolygon),
            _ => vortex_bail!("Invalid GeoType tag: {}", value),
        }
    }
}

impl Display for GeoType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.clickhouse_type_name())
    }
}

/// Metadata for Geo extension type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GeoMetadata {
    /// The specific GEO type.
    pub geo_type: GeoType,
}

impl GeoMetadata {
    /// Create new Geo metadata.
    pub fn new(geo_type: GeoType) -> Self {
        Self { geo_type }
    }
}

impl Display for GeoMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.geo_type)
    }
}

/// The Geo extension type VTable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Geo;

impl Geo {
    /// Create a new Geo extension dtype.
    pub fn new(geo_type: GeoType, nullability: Nullability) -> ExtDType<Self> {
        let metadata = GeoMetadata::new(geo_type);
        // Storage: Binary (WKB-encoded binary data)
        let storage_dtype = DType::Binary(Nullability::NonNullable);
        ExtDType::try_with_vtable(Self, metadata, storage_dtype.with_nullability(nullability))
            .expect("Geo storage dtype is always valid")
    }

    /// Create a Geo DType (type-erased).
    pub fn dtype(geo_type: GeoType, nullability: Nullability) -> DType {
        DType::Extension(Self::new(geo_type, nullability).erased())
    }

    /// Check if a DType is a Geo extension type.
    pub fn is_geo(dtype: &DType) -> bool {
        if let DType::Extension(ext) = dtype {
            ext.id().as_ref() == GEO_EXT_ID
        } else {
            false
        }
    }

    /// Try to extract GeoType from a DType.
    pub fn try_get_type(dtype: &DType) -> Option<GeoType> {
        if let DType::Extension(ext) = dtype {
            if ext.id().as_ref() == GEO_EXT_ID {
                ext.metadata_opt::<Geo>().map(|m| m.geo_type)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl ExtDTypeVTable for Geo {
    type Metadata = GeoMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref(GEO_EXT_ID)
    }

    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![metadata.geo_type as u8])
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        if data.is_empty() {
            vortex_bail!("Geo metadata is empty");
        }
        let geo_type = GeoType::try_from(data[0])?;
        Ok(GeoMetadata::new(geo_type))
    }

    fn validate_dtype(
        &self,
        _metadata: &Self::Metadata,
        storage_dtype: &DType,
    ) -> VortexResult<()> {
        // Storage should be Binary (WKB-encoded binary data)
        match storage_dtype {
            DType::Binary(_) => Ok(()),
            _ => vortex_bail!(
                "Geo extension requires Binary storage, got {:?}",
                storage_dtype
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_type_roundtrip() {
        for geo_type in [
            GeoType::Point,
            GeoType::LineString,
            GeoType::Ring,
            GeoType::Polygon,
            GeoType::MultiLineString,
            GeoType::MultiPolygon,
        ] {
            let tag = geo_type as u8;
            let roundtrip = GeoType::try_from(tag).unwrap();
            assert_eq!(geo_type, roundtrip);
        }
    }

    #[test]
    fn test_geo_dtype_creation() {
        let dtype = Geo::dtype(GeoType::Point, Nullability::Nullable);
        assert!(Geo::is_geo(&dtype));

        if let DType::Extension(ext) = &dtype {
            assert_eq!(ext.id().as_ref(), GEO_EXT_ID);
        } else {
            panic!("Expected Extension dtype");
        }
    }

    #[test]
    fn test_clickhouse_type_names() {
        assert_eq!(GeoType::Point.clickhouse_type_name(), "Point");
        assert_eq!(GeoType::LineString.clickhouse_type_name(), "LineString");
        assert_eq!(GeoType::Ring.clickhouse_type_name(), "Ring");
        assert_eq!(GeoType::Polygon.clickhouse_type_name(), "Polygon");
        assert_eq!(
            GeoType::MultiLineString.clickhouse_type_name(),
            "MultiLineString"
        );
        assert_eq!(GeoType::MultiPolygon.clickhouse_type_name(), "MultiPolygon");
    }

    #[test]
    fn test_from_clickhouse_type() {
        assert_eq!(GeoType::from_clickhouse_type("Point"), Some(GeoType::Point));
        assert_eq!(
            GeoType::from_clickhouse_type("Polygon"),
            Some(GeoType::Polygon)
        );
        assert_eq!(
            GeoType::from_clickhouse_type("MultiPolygon"),
            Some(GeoType::MultiPolygon)
        );
        assert_eq!(GeoType::from_clickhouse_type("String"), None);
    }

    #[test]
    fn test_geo_try_get_type() {
        let dtype = Geo::dtype(GeoType::Polygon, Nullability::NonNullable);
        assert_eq!(Geo::try_get_type(&dtype), Some(GeoType::Polygon));

        let non_geo = DType::Utf8(Nullability::NonNullable);
        assert_eq!(Geo::try_get_type(&non_geo), None);
    }
}
