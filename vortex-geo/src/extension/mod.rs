// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod wkb;

use std::fmt::Display;

pub use wkb::*;

/// Extension metadata that is common to all the geospatial extension types.
///
/// Currently, this is just the coordinate reference system (CRS).
/// We may wish to add a second field for edges interpretation in the future similar to
/// the GeoArrow standard.
#[derive(Clone, PartialEq, Eq, Hash, prost::Message)]
pub struct GeoMetadata {
    #[prost(optional, string, tag = "1")]
    pub crs: Option<String>,
}

impl Display for GeoMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.crs.as_ref() {
            Some(crs) => write!(f, "Geometry(crs={crs})"),
            None => write!(f, "Geometry(unreferenced)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use crate::extension::GeoMetadata;

    #[test]
    fn test_metadata() {
        let meta = GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        };

        assert_eq!(meta.to_string(), "Geometry(crs=EPSG:4326)");
        // round trip
        let bytes = meta.encode_to_vec();
        let decoded = GeoMetadata::decode(bytes.as_slice()).unwrap();
        assert_eq!(decoded, meta);
    }
}
