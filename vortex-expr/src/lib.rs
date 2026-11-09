// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Analyses and rewrites over Vortex expression trees.
//!
//! The expression types themselves ([`Expression`](vortex_array::expr::Expression),
//! [`BoundExpression`](vortex_array::expr::BoundExpression)) and their optimizer live in
//! `vortex_array::expr`. This crate holds the tooling that scan planning builds on top of them:
//! tree annotation and labeling ([`analysis`]), normal forms such as conjunct extraction
//! ([`forms`]), partitioning and substitution rewrites ([`transform`]), and statistics-based
//! pruning ([`stats`]).

pub mod analysis;
pub mod forms;
pub mod stats;
pub mod transform;

pub use analysis::*;
