// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![doc = include_str!(concat!("../", env!("CARGO_PKG_README")))]

use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_arrow::ArrowSessionExt;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_expr::stats::session::StatsSessionExt;
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
use crate::prune::SpatialContainsPrune;
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
pub mod editions;
pub mod extension;
pub mod prune;
pub mod scalar_fn;
#[cfg(any(test, feature = "_test-harness"))]
pub mod test_harness;
#[cfg(test)]
mod tests;

/// Set up a session with spatial extension types, Arrow interop, functions, and pruning.
pub fn initialize(session: &VortexSession) {
    let dtypes = session.dtypes();
    let arrow = session.arrow();

    dtypes.register(WellKnownBinary);
    arrow.register_extension(WellKnownBinary);
    dtypes.register(Point);
    arrow.register_extension(Point);
    dtypes.register(LineString);
    arrow.register_extension(LineString);
    dtypes.register(MultiPoint);
    arrow.register_extension(MultiPoint);
    dtypes.register(Polygon);
    arrow.register_extension(Polygon);
    dtypes.register(MultiLineString);
    arrow.register_extension(MultiLineString);
    dtypes.register(MultiPolygon);
    arrow.register_extension(MultiPolygon);
    dtypes.register(Rect);
    arrow.register_extension(Rect);

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
    session.stats().register_rewrite(SpatialContainsPrune);
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
            .editions()
            .declare_family(&editions::FAMILY)
            .vortex_expect("spatial edition family is valid");
        session
            .register_edition(&editions::DECLARATION)
            .vortex_expect("spatial edition declaration is valid");
    }
    session
        .enable_edition(editions::SPATIAL_2026_08)
        .vortex_expect("spatial edition is registered");
}
