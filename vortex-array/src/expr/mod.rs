// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array statistics as expression-level bounds.
//!
//! Expressions themselves live in the `vortex-expr` crate; this module holds the statistics
//! vocabulary ([`stats::Stat`], [`stats::Precision`]) that arrays expose and that
//! expression-level pruning consumes.

pub mod stats;
