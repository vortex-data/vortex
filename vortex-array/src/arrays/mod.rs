// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! All the built-in encoding schemes and arrays.

#[cfg(any(test, feature = "test-harness"))]
mod assertions;

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
mod expr;
mod extension;
mod fixed_size_list;
mod list;
mod listview;
mod masked;
mod null;
mod primitive;
mod struct_;
mod varbin;
mod varbinview;

#[cfg(feature = "arbitrary")]
pub mod arbitrary;

// TODO(connor): Export exact types, not glob.

pub use bool::*;
pub use chunked::*;
pub use constant::*;
pub use datetime::*;
pub use decimal::*;
pub use dict::*;
pub use expr::*;
pub use extension::*;
pub use fixed_size_list::*;
pub use list::*;
pub use listview::*;
pub use masked::*;
pub use null::*;
pub use primitive::*;
pub use struct_::*;
pub use varbin::*;
pub use varbinview::*;
