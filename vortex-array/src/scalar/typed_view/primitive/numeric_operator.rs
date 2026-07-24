// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`NumericOperator`] enum for arithmetic operations on primitive scalars.

use std::fmt;

use vortex_error::VortexError;
use vortex_error::vortex_err;

use crate::scalar_fn::fns::operators::Operator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        fmt::Debug::fmt(self, f)
    }
}

impl From<NumericOperator> for Operator {
    fn from(op: NumericOperator) -> Self {
        match op {
            NumericOperator::Add => Operator::Add,
            NumericOperator::Sub => Operator::Sub,
            NumericOperator::Mul => Operator::Mul,
            NumericOperator::Div => Operator::Div,
        }
    }
}

impl TryFrom<Operator> for NumericOperator {
    type Error = VortexError;

    fn try_from(op: Operator) -> Result<Self, Self::Error> {
        match op {
            Operator::Add => Ok(NumericOperator::Add),
            Operator::Sub => Ok(NumericOperator::Sub),
            Operator::Mul => Ok(NumericOperator::Mul),
            Operator::Div => Ok(NumericOperator::Div),
            _ => Err(vortex_err!(InvalidArgument: "{op} is not a numeric operator")),
        }
    }
}
