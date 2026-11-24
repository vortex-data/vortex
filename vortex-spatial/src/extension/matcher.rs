// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Matcher for the native geometry extension types.

use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;

use crate::extension::LineString;
use crate::extension::MultiLineString;
use crate::extension::MultiPoint;
use crate::extension::MultiPolygon;
use crate::extension::Point;
use crate::extension::Polygon;
use crate::extension::Rect;
use crate::extension::SpatialMetadata;

/// Matcher for any native geometry extension type: `Point`, `LineString`, `MultiPoint`,
/// `Polygon`, `MultiLineString`, `MultiPolygon`, and `Rect`. Matches to its [`SpatialMetadata`].
pub struct AnyNativeGeometry;

impl Matcher for AnyNativeGeometry {
    type Match<'a> = &'a SpatialMetadata;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        ext_dtype
            .metadata_opt::<Point>()
            .or_else(|| ext_dtype.metadata_opt::<LineString>())
            .or_else(|| ext_dtype.metadata_opt::<MultiPoint>())
            .or_else(|| ext_dtype.metadata_opt::<Polygon>())
            .or_else(|| ext_dtype.metadata_opt::<MultiLineString>())
            .or_else(|| ext_dtype.metadata_opt::<MultiPolygon>())
            .or_else(|| ext_dtype.metadata_opt::<Rect>())
    }
}
