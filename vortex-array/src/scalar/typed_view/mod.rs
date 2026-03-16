// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definitions of typed views into the [`Scalar`] type.
//!
//! Since the [`Scalar`] type is dynamically typed, it is useful to have a typed version of it when
//! we know we are working with a specific kind of [`Scalar`].
//!
//! All the types defined in this module are either typed views into [`Scalar`] or
//! easier-to-work-with value types ([`PValue`] and [`DecimalValue`]).
//!
//! Note that we do **not** have a typed scalar for `FixedSizeList`, as a singular list value has no
//! notion of a "fixed size" in isolation. We use the same [`ListScalar`] for both `FixedSizeList`
//! and `List` `DType`s.
//!
//! [`Scalar`]: crate::scalar::Scalar

mod binary;
mod bool;
mod decimal;
mod extension;
mod list;
mod primitive;
mod struct_;
mod utf8;

pub use binary::*;
pub use bool::*;
pub use decimal::*;
pub use extension::*;
pub use list::*;
pub use primitive::*;
pub use struct_::*;
pub use utf8::*;
