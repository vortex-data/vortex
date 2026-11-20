// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bindings generated from the `proto` schemas of this crate.
//!
//! The bindings are checked in under `generated/`, so building needs no `protoc`. Regenerate them
//! with `cargo run -p xtask -- generate-proto`.

/// Options of the JSON scalar functions.
#[allow(clippy::all)]
#[allow(clippy::absolute_paths)]
#[allow(clippy::nursery)]
#[allow(missing_docs)]
pub mod json {
    include!("proto/generated/vortex.json.rs");
}
