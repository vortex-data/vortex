// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! All the built-in encoding schemes and arrays.

#[cfg(any(test, feature = "test-harness"))]
mod assertions;

#[cfg(any(test, feature = "test-harness"))]
pub use assertions::format_indices;

#[cfg(test)]
mod validation_tests;

#[cfg(any(test, feature = "test-harness"))]
pub mod dict_test;

mod bool;
mod chunked;
mod constant;
mod datetime;
mod decimal;
mod dict;
mod extension;
mod filter;
mod fixed_size_list;
mod list;
mod listview;
mod masked;
mod null;
mod primitive;
mod scalar_fn;
mod struct_;
mod varbin;
mod varbinview;

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
// pub mod pipeline;
// TODO(connor): Export exact types, not glob.

pub use bool::*;
pub use chunked::*;
pub use constant::*;
pub use datetime::*;
pub use decimal::*;
pub use dict::*;
pub use extension::*;
pub use filter::*;
pub use fixed_size_list::*;
pub use list::*;
pub use listview::*;
pub use masked::*;
pub use null::*;
pub use primitive::*;
pub use scalar_fn::*;
pub use struct_::*;
pub use varbin::*;
pub use varbinview::*;
