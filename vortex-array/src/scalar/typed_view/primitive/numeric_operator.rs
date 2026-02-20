// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`NumericOperator`] enum for arithmetic operations on primitive scalars.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Binary element-wise operations on two arrays or two scalars.
pub enum NumericOperator {
    /// Binary element-wise addition of two arrays or of two scalars.
    ///
    /// Errs at runtime if the sum would overflow or underflow.
    Add,
    /// Binary element-wise subtraction of two arrays or of two scalars.
    Sub,
    /// Binary element-wise multiplication of two arrays or of two scalars.
    Mul,
    /// Binary element-wise division of two arrays or of two scalars.
    Div,
}

impl fmt::Display for NumericOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl From<NumericOperator> for crate::expr::Operator {
    fn from(op: NumericOperator) -> Self {
        match op {
            NumericOperator::Add => crate::expr::Operator::Add,
            NumericOperator::Sub => crate::expr::Operator::Sub,
            NumericOperator::Mul => crate::expr::Operator::Mul,
            NumericOperator::Div => crate::expr::Operator::Div,
        }
    }
}
