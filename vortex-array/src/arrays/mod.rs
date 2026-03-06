// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! All the built-in encoding schemes and arrays.

#[cfg(any(test, feature = "_test-harness"))]
mod assertions;

#[cfg(any(test, feature = "_test-harness"))]
pub use assertions::format_indices;

#[cfg(test)]
mod validation_tests;

#[cfg(any(test, feature = "_test-harness"))]
pub mod dict_test;

pub mod bool;
pub mod chunked;
pub mod constant;
pub mod datetime;
pub mod decimal;
pub mod dict;
pub mod extension;
pub mod filter;
pub mod fixed_size_list;
pub mod list;
pub mod listview;
pub mod masked;
pub mod null;
pub mod primitive;
pub mod scalar_fn;
pub mod shared;
pub mod slice;
pub mod struct_;
pub mod varbin;
pub mod varbinview;

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
