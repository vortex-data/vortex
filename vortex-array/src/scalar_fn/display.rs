// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The expression-tree interface scalar functions format themselves through.

use std::fmt::Display;

/// Read-only expression-tree interface used by scalar functions for SQL-style formatting.
///
/// Both `vortex_expr::Expression` and `vortex_expr::BoundExpression` implement this interface,
/// allowing scalar functions to format either representation without converting between them.
pub trait ExprDisplay: Display {
    /// Return the child at `index`.
    fn display_child(&self, index: usize) -> &dyn ExprDisplay;

    /// Return the number of children in this node.
    fn display_children_count(&self) -> usize;
}
