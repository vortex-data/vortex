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
    /// Same as [NumericOperator::Sub] but with the parameters flipped: `right - left`.
    RSub,
    /// Binary element-wise multiplication of two arrays or of two scalars.
    Mul,
    /// Binary element-wise division of two arrays or of two scalars.
    Div,
    /// Same as [NumericOperator::Div] but with the parameters flipped: `right / left`.
    RDiv,
    // Missing from arrow-rs:
    // Min,
    // Max,
    // Pow,
}

impl fmt::Display for NumericOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl NumericOperator {
    /// Returns the operator with swapped operands (e.g., Sub becomes RSub).
    pub fn swap(self) -> Self {
        match self {
            NumericOperator::Add => NumericOperator::Add,
            NumericOperator::Sub => NumericOperator::RSub,
            NumericOperator::RSub => NumericOperator::Sub,
            NumericOperator::Mul => NumericOperator::Mul,
            NumericOperator::Div => NumericOperator::RDiv,
            NumericOperator::RDiv => NumericOperator::Div,
        }
    }
}
