// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Analyses and rewrites over Vortex expression trees.
//!
//! The expression types themselves ([`Expression`](vortex_array::expr::Expression),
//! [`BoundExpression`](vortex_array::expr::BoundExpression)) and their optimizer live in
//! `vortex_array::expr`. This crate holds the tooling that scan planning builds on top of them:
//! tree annotation and labeling ([`analysis`]), normal forms such as conjunct extraction
//! ([`forms`]), partitioning and substitution rewrites ([`transform`]), and statistics-based
//! pruning ([`stats`]), including the [`StatFn`] scalar function that pruning predicates read
//! statistics through. Call [`initialize`] to register it with a session.

pub mod analysis;
pub mod forms;
pub mod stats;
pub mod transform;

pub use analysis::*;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_session::VortexSession;

use crate::stats::stat_fn::StatFn;

/// Register the scalar functions this crate defines with the session.
pub fn initialize(session: &VortexSession) {
    session.scalar_fns().register(StatFn);
}
