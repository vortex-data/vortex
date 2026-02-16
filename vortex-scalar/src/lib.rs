// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar values and types for the Vortex system.
//!
//! This crate provides scalar types and values that can be used to represent individual data
//! elements in the Vortex array system. [`Scalar`]s are composed of a logical data type
//! ([`DType`](vortex_dtype::DType)) and an optional (encoding nullablity) value ([`ScalarValue`]).
//!
//! Note that the implementations of `Scalar` are split into several different modules.

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
mod arrow;

mod cast;
mod constructor;
mod convert;
mod display;
mod downcast;
mod proto;

mod scalar;
mod scalar_value;
mod typed_view;

pub use scalar::*;
pub use scalar_value::*;
pub use typed_view::*;

#[cfg(test)]
mod tests;
mod truncation;

pub use truncation::*;
