// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::array::arrays::VarBinViewArray;
use vortex::error::VortexResult;
use vortex_spatial::extension::LineStringData;
use vortex_spatial::extension::MultiLineStringData;
use vortex_spatial::extension::MultiPointData;
use vortex_spatial::extension::MultiPolygonData;
use vortex_spatial::extension::PointData;
use vortex_spatial::extension::PolygonData;
use vortex_spatial::extension::WellKnownBinaryData;

use crate::exporter::ColumnExporter;
use crate::exporter::varbinview::new_exporter;

/// Create a new exporter for spatial data stored as Well-Known Binary (WKB) format.
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

/// Create an exporter for a native `Polygon` column. Like [`new_point_exporter`], DuckDB `GEOMETRY`
/// vectors carry WKB, so the polygons are serialized to WKB via [`PolygonData::to_wkb`].
pub(crate) fn new_polygon_exporter(
    polygon: PolygonData,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let values = polygon.to_wkb(ctx)?.execute::<VarBinViewArray>(ctx)?;
    new_exporter(values, ctx)
}

/// Create an exporter for a native `MultiPolygon` column, serialized to WKB via
/// [`MultiPolygonData::to_wkb`] (see [`new_point_exporter`]).
pub(crate) fn new_multipolygon_exporter(
    multipolygon: MultiPolygonData,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let values = multipolygon.to_wkb(ctx)?.execute::<VarBinViewArray>(ctx)?;
    new_exporter(values, ctx)
}

/// Create an exporter for a native `LineString` column, serialized to WKB via
/// [`LineStringData::to_wkb`] (see [`new_point_exporter`]).
pub(crate) fn new_linestring_exporter(
    linestring: LineStringData,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let values = linestring.to_wkb(ctx)?.execute::<VarBinViewArray>(ctx)?;
    new_exporter(values, ctx)
}

/// Create an exporter for a native `MultiPoint` column, serialized to WKB via
/// [`MultiPointData::to_wkb`] (see [`new_point_exporter`]).
pub(crate) fn new_multipoint_exporter(
    multipoint: MultiPointData,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let values = multipoint.to_wkb(ctx)?.execute::<VarBinViewArray>(ctx)?;
    new_exporter(values, ctx)
}

/// Create an exporter for a native `MultiLineString` column, serialized to WKB via
/// [`MultiLineStringData::to_wkb`] (see [`new_point_exporter`]).
pub(crate) fn new_multilinestring_exporter(
    multilinestring: MultiLineStringData,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let values = multilinestring
        .to_wkb(ctx)?
        .execute::<VarBinViewArray>(ctx)?;
    new_exporter(values, ctx)
}
