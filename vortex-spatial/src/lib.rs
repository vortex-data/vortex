// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_array::stats::session::StatsSessionExt;
use vortex_arrow::ArrowSessionExt;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::aggregate_fn::GeometryAabb;
use crate::extension::LineString;
use crate::extension::MultiLineString;
use crate::extension::MultiPoint;
use crate::extension::MultiPolygon;
use crate::extension::Point;
use crate::extension::Polygon;
use crate::extension::Rect;
use crate::extension::WellKnownBinary;
use crate::prune::SpatialDistancePrune;
use crate::prune::SpatialIntersectsPrune;
use crate::scalar_fn::area::SpatialArea;
use crate::scalar_fn::collect::SpatialCollect;
use crate::scalar_fn::contains::SpatialContains;
use crate::scalar_fn::convex_hull::SpatialConvexHull;
use crate::scalar_fn::distance::SpatialDistance;
use crate::scalar_fn::envelope::SpatialEnvelope;
use crate::scalar_fn::intersects::SpatialIntersects;
use crate::scalar_fn::length::SpatialLength;
use crate::scalar_fn::make_line::SpatialMakeLine;

pub mod aggregate_fn;
mod dense_union;
pub mod editions;
pub mod extension;
pub mod prune;
pub mod scalar_fn;
#[cfg(any(test, feature = "_test-harness"))]
pub mod test_harness;
#[cfg(test)]
mod tests;

/// Set up a session with support for spatial extension types, encodings and layouts.
pub fn initialize(session: &VortexSession) {
    dense_union::initialize(session);

    // Register the spatial extension types.
    session.dtypes().register(WellKnownBinary);
    session.arrow().register_exporter(Arc::new(WellKnownBinary));
    session.arrow().register_importer(Arc::new(WellKnownBinary));
    session.dtypes().register(Point);
    session.arrow().register_exporter(Arc::new(Point));
    session.arrow().register_importer(Arc::new(Point));
    session.dtypes().register(LineString);
    session.arrow().register_exporter(Arc::new(LineString));
    session.arrow().register_importer(Arc::new(LineString));
    session.dtypes().register(MultiPoint);
    session.arrow().register_exporter(Arc::new(MultiPoint));
    session.arrow().register_importer(Arc::new(MultiPoint));
    session.dtypes().register(Polygon);
    session.arrow().register_exporter(Arc::new(Polygon));
    session.arrow().register_importer(Arc::new(Polygon));
    session.dtypes().register(MultiLineString);
    session.arrow().register_exporter(Arc::new(MultiLineString));
    session.arrow().register_importer(Arc::new(MultiLineString));
    session.dtypes().register(MultiPolygon);
    session.arrow().register_exporter(Arc::new(MultiPolygon));
    session.arrow().register_importer(Arc::new(MultiPolygon));
    session.dtypes().register(Rect);
    session.arrow().register_exporter(Arc::new(Rect));
    session.arrow().register_importer(Arc::new(Rect));

    // Register the geometry scalar functions.
    session.scalar_fns().register(SpatialArea);
    session.scalar_fns().register(SpatialCollect);
    session.scalar_fns().register(SpatialConvexHull);
    session.scalar_fns().register(SpatialEnvelope);
    session.scalar_fns().register(SpatialContains);
    session.scalar_fns().register(SpatialDistance);
    session.scalar_fns().register(SpatialIntersects);
    session.scalar_fns().register(SpatialMakeLine);
    session.scalar_fns().register(SpatialLength);

    // The axis-aligned bounding-box (AABB) aggregate; self-declares as a per-chunk zone stat for
    // geometry columns.
    session.aggregate_fns().register(GeometryAabb);

    // Register the spatial pruning rules that use that AABB.
    session.stats().register_rewrite(SpatialDistancePrune);
    session.stats().register_rewrite(SpatialIntersectsPrune);

    // Spatial members belong to their own edition family, enabled here so the writer may emit
    // the AABB zone stat this session just registered. `initialize` is idempotent, so a
    // repeated call must not re-declare the edition.
    if session
        .editions()
        .find(&editions::SPATIAL_2026_08)
        .is_none()
    {
        session
            .register_edition(&editions::DECLARATION)
            .map_err(|error| vortex_err!("{error}"))
            .vortex_expect("spatial edition declaration is valid");
    }
    session
        .enable_edition(editions::SPATIAL_2026_08)
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("spatial edition is registered");
}
