// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::ScalarValue;

/// Let's define a dummy expression language.
pub enum Expression {
    /// References the root scope.
    Root,
    /// Holds a scalar value.
    Literal(ScalarValue),
    /// Less than comparison.
    Lt(Box<Expression>, Box<Expression>),
    /// Logical AND operation.
    And(Box<Expression>, Box<Expression>),
}
