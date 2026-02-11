// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension scalar types and traits for working with custom (extension) scalar values.
//!
//! This module provides the infrastructure for defining and working with extension scalar types in
//! Vortex, including:
//!
//! - [`Matcher`] - For matching extension scalar types
//! - [`ExtScalarValueRef`] and [`OwnedExtScalarValue`] - References and owned values for extension
//!   scalars
//! - [`ExtScalarVTable`] and [`DynExtScalarVTable`] - Virtual table traits for extension scalar
//!   operations

mod datetime;

mod matcher;
pub use matcher::Matcher;

mod scalar_value;
pub use scalar_value::ExtScalarValue;
pub use scalar_value::ExtScalarValueRef;

mod vtable;
pub use vtable::DynExtScalarVTable;
pub use vtable::ExtScalarVTable;

#[cfg(test)]
mod tests;
