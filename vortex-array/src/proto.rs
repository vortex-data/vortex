// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bindings generated from this crate's `proto` schemas.

/// Data types.
#[allow(clippy::all)]
#[allow(clippy::absolute_paths)]
#[allow(clippy::nursery)]
#[allow(missing_docs)]
pub mod dtype {
    include!(concat!(env!("OUT_DIR"), "/proto/vortex.dtype.rs"));
}

/// Scalar values.
#[allow(clippy::all)]
#[allow(clippy::absolute_paths)]
#[allow(clippy::nursery)]
#[allow(missing_docs)]
pub mod scalar {
    include!(concat!(env!("OUT_DIR"), "/proto/vortex.scalar.rs"));
}

/// Expressions.
#[allow(clippy::all)]
#[allow(clippy::absolute_paths)]
#[allow(clippy::nursery)]
#[allow(missing_docs)]
pub mod expr {
    include!(concat!(env!("OUT_DIR"), "/proto/vortex.expr.rs"));
}
