// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop tests for the spatial extension types, exercising the session wiring set up
//! by [`crate::initialize`].

mod linestring;
mod multilinestring;
mod multipoint;
mod multipolygon;
mod point;
mod rect;
mod wkb;

use std::sync::LazyLock;

use arrow_schema::Field;
use vortex_array::dtype::DType;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

/// A session with the spatial types and functions registered.
static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::test_harness::spatial_session);

/// The Arrow [`Field`] for `dtype`, which every test here requires to exist.
fn arrow_field(name: &str, dtype: &DType) -> VortexResult<Field> {
    SESSION
        .arrow()
        .to_arrow_field(name, dtype)?
        .ok_or_else(|| vortex_err!("dtype {dtype} has no Arrow field"))
}
