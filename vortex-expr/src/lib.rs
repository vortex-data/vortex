// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex expression trees and the tooling built on them.
//!
//! An [`Expression`] is a tree of scalar functions over a scope; binding it to a scope's
//! [`DType`](vortex_array::dtype::DType) gives a [`BoundExpression`] whose every node carries
//! its dtype. Expressions are applied to arrays through [`ArrayExprExt`], optimized with each
//! scalar function's simplification and reduction rules, and serialized through [`proto`].
//! Scalar functions themselves and the core builtins live in `vortex_array::scalar_fn`;
//! [`ExprBuiltins`] and the free constructors in this crate build expressions over them.
//!
//! Scan planning builds on the tooling here: tree [`traversal`], annotation and labeling
//! ([`analysis`]), normal forms such as conjunct extraction ([`forms`]), partitioning and
//! substitution rewrites ([`transform`]), and statistics-based pruning ([`stats`]), including
//! the [`StatFn`] scalar function that pruning predicates read statistics through. Call
//! [`initialize`] to register it with a session.

pub mod analysis;
mod apply;
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod bound_expression;
pub mod builtins;
pub mod display;
pub(crate) mod expression;
mod exprs;
pub mod forms;
pub mod lambda;
pub mod optimize;
pub mod proto;
mod scalar_fn_ext;
pub mod scope;
pub mod stats;
#[doc(hidden)]
pub mod test_harness;
pub mod transform;
pub mod traversal;

#[cfg(test)]
mod tests;

pub use analysis::*;
pub use apply::ArrayExprExt;
pub use bound_expression::*;
pub use builtins::ExprBuiltins;
pub use expression::*;
pub use exprs::and;
pub use exprs::and_collect;
pub use exprs::binary;
pub use exprs::bound;
pub use exprs::cast;
pub use exprs::checked_add;
pub use exprs::col;
pub use exprs::eq;
pub use exprs::fill_null;
pub use exprs::get_item;
pub use exprs::gt;
pub use exprs::gt_eq;
pub use exprs::is_not_null;
pub use exprs::is_null;
pub use exprs::is_root;
pub use exprs::lit;
pub use exprs::lt;
pub use exprs::lt_eq;
pub use exprs::mask;
pub use exprs::not;
pub use exprs::not_eq;
pub use exprs::or;
pub use exprs::or_collect;
pub use exprs::pack;
pub use exprs::root;
pub use exprs::select;
pub use exprs::select_exclude;
pub use exprs::var;
pub use lambda::*;
pub use scalar_fn_ext::ScalarFnExprExt;
pub use scope::*;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_session::VortexSession;

use crate::stats::stat_fn::StatFn;

/// Register the scalar functions this crate defines with the session.
pub fn initialize(session: &VortexSession) {
    session.scalar_fns().register(StatFn);
}
