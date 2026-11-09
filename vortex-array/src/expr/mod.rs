// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bound expression trees and their optimizer, and the statistics vocabulary arrays expose.
//!
//! [`BoundExpression`] lives here because the scalar function vtable reduces over it: keeping it
//! next to [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable) lets every reduction rule be
//! monomorphized for expression trees as well as array trees. Unbound expressions, binding,
//! constructors and the analyses built on them live in the `vortex-expr` crate.

mod bound_expression;
pub mod display;
mod optimize;
pub mod stats;

pub use bound_expression::*;
