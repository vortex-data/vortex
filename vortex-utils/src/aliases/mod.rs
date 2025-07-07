// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Re-exports of third-party crates we use in the API.
//!
//! The HashMap/Set should be preferred over the standard library variants or other alternatives.
//! Currently defers to the excellent [hashbrown](https://docs.rs/hashbrown/latest/hashbrown/) crate.

pub mod hash_map;
pub mod hash_set;

pub use hashbrown::DefaultHashBuilder;
