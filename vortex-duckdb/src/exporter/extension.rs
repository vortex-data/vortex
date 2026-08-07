// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::extension::datetime::AnyTemporal;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex_spatial::extension::LineString;
use vortex_spatial::extension::LineStringData;
use vortex_spatial::extension::MultiLineString;
use vortex_spatial::extension::MultiLineStringData;
use vortex_spatial::extension::MultiPoint;
use vortex_spatial::extension::MultiPointData;
use vortex_spatial::extension::MultiPolygon;
use vortex_spatial::extension::MultiPolygonData;
use vortex_spatial::extension::Point;
use vortex_spatial::extension::PointData;
use vortex_spatial::extension::Polygon;
use vortex_spatial::extension::PolygonData;
use vortex_spatial::extension::WellKnownBinary;
use vortex_spatial::extension::WellKnownBinaryData;

use crate::exporter::ColumnExporter;
use crate::exporter::spatial;
use crate::exporter::temporal;

pub(crate) fn new_exporter(
    ext: ExtensionArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    if ext.ext_dtype().is::<AnyTemporal>() {
        return temporal::new_exporter(TemporalArray::try_from(ext)?, ctx);
    }

    if ext.ext_dtype().is::<WellKnownBinary>() {
        return spatial::new_wkb_exporter(WellKnownBinaryData::try_from(ext)?, ctx);
    }

    if ext.ext_dtype().is::<Point>() {
        return spatial::new_point_exporter(PointData::try_from(ext)?, ctx);
    }

    if ext.ext_dtype().is::<LineString>() {
        return spatial::new_linestring_exporter(LineStringData::try_from(ext)?, ctx);
    }

    if ext.ext_dtype().is::<MultiPoint>() {
        return spatial::new_multipoint_exporter(MultiPointData::try_from(ext)?, ctx);
    }

    if ext.ext_dtype().is::<Polygon>() {
        return spatial::new_polygon_exporter(PolygonData::try_from(ext)?, ctx);
    }

    if ext.ext_dtype().is::<MultiLineString>() {
        return spatial::new_multilinestring_exporter(MultiLineStringData::try_from(ext)?, ctx);
    }

    if ext.ext_dtype().is::<MultiPolygon>() {
        return spatial::new_multipolygon_exporter(MultiPolygonData::try_from(ext)?, ctx);
    }

    vortex_bail!("no non-temporal extension exporter")
}
