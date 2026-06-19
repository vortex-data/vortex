// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::arrow::ArrowSession;
use vortex_array::dtype::session::DTypeSession;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_session::VortexSessionBuilder;

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
pub fn initialize(session: &mut VortexSessionBuilder) {
    // Register the geospatial extension types.
    {
        let dtypes = session.get_mut::<DTypeSession>();
        dtypes.register(WellKnownBinary);
        dtypes.register(Point);
        dtypes.register(Polygon);
    }
    {
        let arrow = session.get_mut::<ArrowSession>();
        arrow.register_exporter(Arc::new(WellKnownBinary));
        arrow.register_importer(Arc::new(WellKnownBinary));
        arrow.register_exporter(Arc::new(Point));
        arrow.register_importer(Arc::new(Point));
        arrow.register_exporter(Arc::new(Polygon));
        arrow.register_importer(Arc::new(Polygon));
    }

    // Register the geometry scalar functions.
    session.get_mut::<ScalarFnSession>().register(GeoDistance);
}
