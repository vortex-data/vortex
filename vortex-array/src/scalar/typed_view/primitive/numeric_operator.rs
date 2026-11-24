// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`NumericOperator`] enum for arithmetic operations on primitive scalars.

use std::fmt;

use crate::proto::expr::numeric_opts::NumericOp;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Binary element-wise operations.
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
        let display = match self {
            NumericOperator::Add => "+",
            NumericOperator::Sub => "-",
            NumericOperator::Mul => "*",
            NumericOperator::Div => "/",
        };
        fmt::Display::fmt(display, f)
    }
}

impl From<NumericOperator> for NumericOp {
    fn from(op: NumericOperator) -> Self {
        match op {
            NumericOperator::Add => NumericOp::Add,
            NumericOperator::Sub => NumericOp::Sub,
            NumericOperator::Mul => NumericOp::Mul,
            NumericOperator::Div => NumericOp::Div,
        }
    }
}

impl From<NumericOp> for NumericOperator {
    fn from(op: NumericOp) -> Self {
        match op {
            NumericOp::Add => NumericOperator::Add,
            NumericOp::Sub => NumericOperator::Sub,
            NumericOp::Mul => NumericOperator::Mul,
            NumericOp::Div => NumericOperator::Div,
        }
    }
}
