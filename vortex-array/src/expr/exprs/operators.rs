// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::fmt::{Display, Formatter};

use vortex_error::{VortexError, VortexResult, vortex_bail};
use vortex_proto::expr::binary_opts::BinaryOp;

use crate::compute;

/// Equalities, inequalities, and boolean operations over possibly null values.
///
/// For most operations, if either side is null, the result is null.
///
/// The Boolean operators (And, Or) obey [Kleene (three-valued) logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics).
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Operator {
    /// Expressions are equal.
    Eq,
    /// Expressions are not equal.
    NotEq,
    /// Expression is greater than another
    Gt,
    /// Expression is greater or equal to another
    Gte,
    /// Expression is less than another
    Lt,
    /// Expression is less or equal to another
    Lte,
    /// Boolean AND (∧).
    And,
    /// Boolean OR (∨).
    Or,
    /// The sum of the arguments.
    ///
    /// Errs at runtime if the sum would overflow or underflow.
    Add,
    /// The difference between the arguments.
    ///
    /// Errs at runtime if the sum would overflow or underflow.
    ///
    /// The result is null at any index that either input is null.
    Sub,
    /// Multiple two numbers
    Mul,
    /// Divide the left side by the right side
    Div,
}

impl From<Operator> for i32 {
    fn from(value: Operator) -> Self {
        let op: BinaryOp = value.into();
        op.into()
    }
}

impl From<Operator> for BinaryOp {
    fn from(value: Operator) -> Self {
        match value {
            Operator::Eq => BinaryOp::Eq,
            Operator::NotEq => BinaryOp::NotEq,
            Operator::Gt => BinaryOp::Gt,
            Operator::Gte => BinaryOp::Gte,
            Operator::Lt => BinaryOp::Lt,
            Operator::Lte => BinaryOp::Lte,
            Operator::And => BinaryOp::And,
            Operator::Or => BinaryOp::Or,
            Operator::Add => BinaryOp::Add,
            Operator::Sub => BinaryOp::Sub,
            Operator::Mul => BinaryOp::Mul,
            Operator::Div => BinaryOp::Div,
        }
    }
}

impl TryFrom<i32> for Operator {
    type Error = VortexError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Ok(BinaryOp::try_from(value)?.into())
    }
}

impl From<BinaryOp> for Operator {
    fn from(value: BinaryOp) -> Self {
        match value {
            BinaryOp::Eq => Operator::Eq,
            BinaryOp::NotEq => Operator::NotEq,
            BinaryOp::Gt => Operator::Gt,
            BinaryOp::Gte => Operator::Gte,
            BinaryOp::Lt => Operator::Lt,
            BinaryOp::Lte => Operator::Lte,
            BinaryOp::And => Operator::And,
            BinaryOp::Or => Operator::Or,
            BinaryOp::Add => Operator::Add,
            BinaryOp::Sub => Operator::Sub,
            BinaryOp::Mul => Operator::Mul,
            BinaryOp::Div => Operator::Div,
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match &self {
            Operator::Eq => "=",
            Operator::NotEq => "!=",
            Operator::Gt => ">",
            Operator::Gte => ">=",
            Operator::Lt => "<",
            Operator::Lte => "<=",
            Operator::And => "and",
            Operator::Or => "or",
            Operator::Add => "+",
            Operator::Sub => "-",
            Operator::Mul => "*",
            Operator::Div => "/",
        };
        Display::fmt(display, f)
    }
}

impl Operator {
    pub fn inverse(self) -> Option<Self> {
        match self {
            Operator::Eq => Some(Operator::NotEq),
            Operator::NotEq => Some(Operator::Eq),
            Operator::Gt => Some(Operator::Lte),
            Operator::Gte => Some(Operator::Lt),
            Operator::Lt => Some(Operator::Gte),
            Operator::Lte => Some(Operator::Gt),
            Operator::And
            | Operator::Or
            | Operator::Add
            | Operator::Sub
            | Operator::Mul
            | Operator::Div => None,
        }
    }

    pub fn logical_inverse(self) -> Option<Self> {
        match self {
            Operator::And => Some(Operator::Or),
            Operator::Or => Some(Operator::And),
            _ => None,
        }
    }

    /// Change the sides of the operator, so that changing lhs and rhs won't change the result of the operation
    pub fn swap(self) -> Option<Self> {
        match self {
            Operator::Eq => Some(Operator::Eq),
            Operator::NotEq => Some(Operator::NotEq),
            Operator::Gt => Some(Operator::Lt),
            Operator::Gte => Some(Operator::Lte),
            Operator::Lt => Some(Operator::Gt),
            Operator::Lte => Some(Operator::Gte),
            Operator::And => Some(Operator::And),
            Operator::Or => Some(Operator::Or),
            Operator::Add => Some(Operator::Add),
            Operator::Mul => Some(Operator::Mul),
            Operator::Sub | Operator::Div => None,
        }
    }

    pub fn maybe_cmp_operator(self) -> Option<compute::Operator> {
        match self {
            Operator::Eq => Some(compute::Operator::Eq),
            Operator::NotEq => Some(compute::Operator::NotEq),
            Operator::Lt => Some(compute::Operator::Lt),
            Operator::Lte => Some(compute::Operator::Lte),
            Operator::Gt => Some(compute::Operator::Gt),
            Operator::Gte => Some(compute::Operator::Gte),
            _ => None,
        }
    }

    pub fn is_arithmetic(&self) -> bool {
        matches!(self, Self::Add | Self::Sub | Self::Mul | Self::Div)
    }
}

impl From<compute::Operator> for Operator {
    fn from(cmp_operator: compute::Operator) -> Self {
        match cmp_operator {
            compute::Operator::Eq => Operator::Eq,
            compute::Operator::NotEq => Operator::NotEq,
            compute::Operator::Gt => Operator::Gt,
            compute::Operator::Gte => Operator::Gte,
            compute::Operator::Lt => Operator::Lt,
            compute::Operator::Lte => Operator::Lte,
        }
    }
}

impl TryInto<compute::Operator> for Operator {
    type Error = VortexError;

    fn try_into(self) -> VortexResult<compute::Operator> {
        Ok(match self {
            Operator::Eq => compute::Operator::Eq,
            Operator::NotEq => compute::Operator::NotEq,
            Operator::Gt => compute::Operator::Gt,
            Operator::Gte => compute::Operator::Gte,
            Operator::Lt => compute::Operator::Lt,
            Operator::Lte => compute::Operator::Lte,
            _ => vortex_bail!("Not a compute operator: {}", self),
        })
    }
}
