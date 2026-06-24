// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::array::arrays::VarBinViewArray;
use vortex::error::VortexResult;
use vortex_geo::extension::PointData;
use vortex_geo::extension::PolygonData;
use vortex_geo::extension::WellKnownBinaryData;

use crate::exporter::ColumnExporter;
use crate::exporter::varbinview::new_exporter;

/// Create a new exporter for geospatial data stored as Well-Known Binary (WKB) format.
pub(crate) fn new_wkb_exporter(
    array: WellKnownBinaryData,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let values = array.wkb_values().clone().execute::<VarBinViewArray>(ctx)?;
    new_exporter(values, ctx)
}

/// Create an exporter for a native `Point` column. DuckDB `GEOMETRY` vectors carry WKB, so the
/// points are serialized to WKB via [`PointData::to_wkb`] (only for rows DuckDB materializes —
/// with predicate pushdown that's just the survivors).
pub(crate) fn new_point_exporter(
    point: PointData,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let values = point.to_wkb(ctx)?.execute::<VarBinViewArray>(ctx)?;
    new_exporter(values, ctx)
}

/// Create an exporter for a native `Polygon` column. DuckDB `GEOMETRY` vectors carry WKB, so the
/// polygons are serialized to WKB via [`PolygonData::to_wkb`] (only for rows DuckDB materializes).
pub(crate) fn new_polygon_exporter(
    polygon: PolygonData,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let values = polygon.to_wkb(ctx)?.execute::<VarBinViewArray>(ctx)?;
    new_exporter(values, ctx)
}
