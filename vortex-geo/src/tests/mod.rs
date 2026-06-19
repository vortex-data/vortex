// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop tests for the geospatial extension types, exercising the session wiring set up
//! by [`crate::initialize`].

mod point;
mod wkb;

use std::sync::LazyLock;

use vortex_session::VortexSession;

/// A session with the geospatial types and functions registered.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let mut builder = vortex_array::default_session_builder();
    crate::initialize(&mut builder);
    builder.build()
});
