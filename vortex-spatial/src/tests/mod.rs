// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop tests for the spatial extension types, exercising the session wiring set up
//! by [`crate::initialize`].

mod geometry;
mod linestring;
mod multilinestring;
mod multipoint;
mod multipolygon;
mod point;
mod rect;
mod wkb;

use std::sync::LazyLock;

use vortex_session::VortexSession;

/// A session with the spatial types and functions registered.
static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::test_harness::spatial_session);
