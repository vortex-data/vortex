// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Re-exports of third-party crates we use in the API.
//!
//! The HashMap/Set should be preferred over the standard library variants or other alternatives.
//! Currently defers to the excellent [hashbrown](https://docs.rs/hashbrown/latest/hashbrown/) crate.

/// DashMap type aliases and re-exports.
#[cfg(feature = "dashmap")]
pub mod dash_map;
/// HashMap type aliases and re-exports.
pub mod hash_map;
/// HashSet type aliases and re-exports.
pub mod hash_set;

/// The default hash builder used by HashMap and HashSet.
pub use hashbrown::DefaultHashBuilder;
