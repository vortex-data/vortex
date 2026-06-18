// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::arrow::ArrowSessionExt;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_session::VortexSession;

use crate::extension::Point;
use crate::extension::Polygon;
use crate::extension::WellKnownBinary;
use crate::scalar_fn::distance::GeoDistance;

pub mod extension;
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
}
