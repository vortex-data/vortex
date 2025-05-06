//! `ExtensionArray` wrapper for arrays that hold geospatial data types.

use vortex::arrays::ExtensionArray;
use vortex::error::{VortexError, VortexResult};
use vortex::variants::ExtensionArrayTrait;

use crate::{GeoMetadata, GeometryType};

/// Holder for what is known to be one of the blessed extension array types.
#[allow(unused)]
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
            GeometryType::WKB(meta) => Self::Polygon(value, meta),
            GeometryType::LineString(meta) => Self::Polygon(value, meta),
        })
    }
}
