// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::ops::Deref;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use wkb::reader::GeometryType;

use crate::extension::GeoMetadata;

/// A typed handle to an [`ExtensionArray`] that contains WKB-encoded data.
///
/// You can construct this safely using `WellKnownBinaryData::try_from(ExtensionArray)`.
#[derive(Debug, Clone)]
pub struct WellKnownBinaryData {
    ext: ExtensionArray,
}

impl WellKnownBinaryData {
    /// A reference to the array that holds the Well-Known Binary scalar values.
    pub fn wkb_values(&self) -> &ArrayRef {
        self.ext.storage_array()
    }

    /// A reference to the [geospatial metadata][GeoMetadata].
    pub fn geo_metadata(&self) -> &GeoMetadata {
        self.ext
            .dtype()
            .as_extension()
            .metadata::<WellKnownBinary>()
    }
}

impl TryFrom<ExtensionArray> for WellKnownBinaryData {
    type Error = VortexError;

    fn try_from(ext: ExtensionArray) -> Result<Self, Self::Error> {
        if !ext.ext_dtype().is::<WellKnownBinary>() {
            vortex_bail!("array extension dtype {} is not a WKB", ext.ext_dtype());
        }

        Ok(Self { ext })
    }
}

/// An [extension type][ExtVTable] for OGC Well-known Binary (WKB) data format.
///
/// This is one of the most common formats for sharing of geometry data between analytic systems,
/// used by DuckDB, PostGIS and GeoParquet.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct WellKnownBinary;

/// A reference to a value of well-known binary encoded geometry data.
///
/// Interpreting this value is dependent on the [geospatial metadata][GeoMetadata] in the extension
/// schema for the array where this scalar is taken from.
pub struct Wkb<'a>(wkb::reader::Wkb<'a>);

impl<'a> Wkb<'a> {
    /// Attempt to decode a well-known binary value from a byte slice.
    ///
    /// This will not cause any data allocations or copies, but it will perform a one-pass
    /// validation on the structure of the WKB.
    pub fn try_from_bytes(bytes: &'a [u8]) -> VortexResult<Self> {
        wkb::reader::Wkb::try_new(bytes)
            .map_err(|e| vortex_err!("failed parsing WKB: {e}"))
            .map(Wkb)
    }
}

impl<'a> Display for Wkb<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let geometry_kind = match self.0.geometry_type() {
            GeometryType::Point => "point",
            GeometryType::LineString => "linestring",
            GeometryType::Polygon => "polygon",
            GeometryType::MultiPoint => "multipoint",
            GeometryType::MultiLineString => "multilinestring",
            GeometryType::MultiPolygon => "multipolygon",
            GeometryType::GeometryCollection => "geometrycollection",
            _ => "unknown",
        };
        // TODO(aduffy): make this more useful
        write!(f, "WKB({geometry_kind})")
    }
}

impl<'a> Deref for Wkb<'a> {
    type Target = wkb::reader::Wkb<'a>;

    fn deref(&self) -> &wkb::reader::Wkb<'a> {
        &self.0
    }
}

impl ExtVTable for WellKnownBinary {
    type Metadata = GeoMetadata;

    type NativeValue<'a> = Wkb<'a>;

    fn id(&self) -> ExtId {
        ExtId::new_static("vortex.geo.wkb")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(GeoMetadata::decode(metadata)?)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(
            ext_dtype.storage_dtype().is_binary(),
            "geo.wkb must have binary storage type, was {}",
            ext_dtype.storage_dtype()
        );

        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Wkb::try_from_bytes(storage_value.as_binary().as_slice())
    }
}
