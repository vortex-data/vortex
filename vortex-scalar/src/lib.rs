// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar values and types for the Vortex system.
//!
//! This crate provides scalar types and values that can be used to represent individual data
//! elements in the Vortex array system. [`Scalar`]s are composed of a logical data type
//! ([`DType`](vortex_dtype::DType)) and a value ([`ScalarValue`]).

#![deny(missing_docs)]

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
mod arrow;
mod binary;
mod bool;
mod decimal;
mod display;
pub mod extension;
mod list;
mod null;
mod primitive;
mod proto;
mod pvalue;
mod scalar;
mod scalar_value;
mod struct_;
mod utf8;

pub use binary::*;
pub use bool::*;
pub use decimal::*;
pub use extension::ExtScalar;
pub use extension::ExtScalarRef;
pub use list::*;
pub use primitive::*;
pub use pvalue::*;
pub use scalar::*;
pub use scalar_value::*;
pub use struct_::*;
pub use utf8::*;

mod datetime;
#[cfg(test)]
mod tests;
