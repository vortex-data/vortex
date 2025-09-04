// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! All the built-in encoding schemes and arrays.

#[cfg(test)]
mod assertions;

#[cfg(test)]
mod validation_tests;

mod bool;
mod chunked;
mod constant;
mod datetime;
mod extension;
mod fixed_size_list;
mod list;
mod null;
mod primitive;
mod struct_;
mod varbin;
mod varbinview;

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
mod decimal;

pub use bool::*;
pub use chunked::*;
pub use constant::*;
pub use datetime::*;
pub use decimal::*;
pub use extension::*;
pub use fixed_size_list::*;
pub use list::*;
pub use null::*;
pub use primitive::*;
pub use struct_::*;
pub use varbin::*;
pub use varbinview::*;
