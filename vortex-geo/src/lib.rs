// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_array::stats::session::StatsSessionExt;
use vortex_session::VortexSession;

use crate::aggregate_fn::GeometryBounds;
use crate::extension::Point;
use crate::extension::Polygon;
use crate::extension::WellKnownBinary;
use crate::prune::GeoDistanceBoundsPrune;
use crate::scalar_fn::distance::GeoDistance;

pub mod aggregate_fn;
pub mod extension;
pub mod prune;
pub mod scalar_fn;
#[cfg(test)]
mod test_harness;
#[cfg(test)]
mod tests;

/// Set up a session with support for geospatial extension types, encodings and layouts.
pub fn initialize(session: &VortexSession) {
    // Register the geospatial extension types.
    session.dtypes().register(WellKnownBinary);
    session.arrow().register_exporter(Arc::new(WellKnownBinary));
    session.arrow().register_importer(Arc::new(WellKnownBinary));
    session.dtypes().register(Point);
    session.arrow().register_exporter(Arc::new(Point));
    session.arrow().register_importer(Arc::new(Point));
    session.dtypes().register(Polygon);
    session.arrow().register_exporter(Arc::new(Polygon));
    session.arrow().register_importer(Arc::new(Polygon));

    // Register the geometry scalar functions.
    session.scalar_fns().register(GeoDistance);

    // The bounding-box aggregate; self-declares as a per-chunk zone stat for geometry columns.
    session.aggregate_fns().register(GeometryBounds);

    // Register the spatial pruning rule that uses that bounding box.
    session.stats().register_rewrite(GeoDistanceBoundsPrune);
}
