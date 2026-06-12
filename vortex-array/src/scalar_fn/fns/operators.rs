// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexError;
use vortex_proto::expr::binary_opts::BinaryOp;

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
    /// BoundExpr is greater than another
    Gt,
    /// BoundExpr is greater or equal to another
    Gte,
    /// BoundExpr is less than another
    Lt,
    /// BoundExpr is less or equal to another
    Lte,
    /// Boolean AND (∧).
    // TODO(joe): rename to KleeneAnd
    And,
    /// Boolean OR (∨).
    // TODO(joe): rename to KleeneOr
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

    pub fn is_arithmetic(&self) -> bool {
        matches!(self, Self::Add | Self::Sub | Self::Mul | Self::Div)
    }

    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Eq | Self::NotEq | Self::Gt | Self::Gte | Self::Lt | Self::Lte
        )
    }
}

/// The six comparison operators, providing compile-time guarantees that only
/// comparison variants are used where comparisons are expected.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompareOperator {
    /// Expressions are equal.
    Eq,
    /// Expressions are not equal.
    NotEq,
    /// BoundExpr is greater than another.
    Gt,
    /// BoundExpr is greater or equal to another.
    Gte,
    /// BoundExpr is less than another.
    Lt,
    /// BoundExpr is less or equal to another.
    Lte,
}

impl CompareOperator {
    /// Return the logical inverse of this comparison operator.
    pub fn inverse(self) -> Self {
        match self {
            CompareOperator::Eq => CompareOperator::NotEq,
            CompareOperator::NotEq => CompareOperator::Eq,
            CompareOperator::Gt => CompareOperator::Lte,
            CompareOperator::Gte => CompareOperator::Lt,
            CompareOperator::Lt => CompareOperator::Gte,
            CompareOperator::Lte => CompareOperator::Gt,
        }
    }

    /// Swap the sides of the operator so that swapping lhs and rhs preserves the result.
    pub fn swap(self) -> Self {
        match self {
            CompareOperator::Eq => CompareOperator::Eq,
            CompareOperator::NotEq => CompareOperator::NotEq,
            CompareOperator::Gt => CompareOperator::Lt,
            CompareOperator::Gte => CompareOperator::Lte,
            CompareOperator::Lt => CompareOperator::Gt,
            CompareOperator::Lte => CompareOperator::Gte,
        }
    }
}

impl Display for CompareOperator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match self {
            CompareOperator::Eq => "=",
            CompareOperator::NotEq => "!=",
            CompareOperator::Gt => ">",
            CompareOperator::Gte => ">=",
            CompareOperator::Lt => "<",
            CompareOperator::Lte => "<=",
        };
        Display::fmt(display, f)
    }
}

impl From<CompareOperator> for Operator {
    fn from(value: CompareOperator) -> Self {
        match value {
            CompareOperator::Eq => Operator::Eq,
            CompareOperator::NotEq => Operator::NotEq,
            CompareOperator::Gt => Operator::Gt,
            CompareOperator::Gte => Operator::Gte,
            CompareOperator::Lt => Operator::Lt,
            CompareOperator::Lte => Operator::Lte,
        }
    }
}

impl TryFrom<Operator> for CompareOperator {
    type Error = VortexError;

    fn try_from(value: Operator) -> Result<Self, Self::Error> {
        match value {
            Operator::Eq => Ok(CompareOperator::Eq),
            Operator::NotEq => Ok(CompareOperator::NotEq),
            Operator::Gt => Ok(CompareOperator::Gt),
            Operator::Gte => Ok(CompareOperator::Gte),
            Operator::Lt => Ok(CompareOperator::Lt),
            Operator::Lte => Ok(CompareOperator::Lte),
            other => Err(vortex_error::vortex_err!(
                InvalidArgument: "{other} is not a comparison operator"
            )),
        }
    }
}

#[cfg(feature = "arbitrary")]
impl<'a> arbitrary::Arbitrary<'a> for CompareOperator {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(match u.int_in_range(0..=5)? {
            0 => CompareOperator::Eq,
            1 => CompareOperator::NotEq,
            2 => CompareOperator::Gt,
            3 => CompareOperator::Gte,
            4 => CompareOperator::Lt,
            5 => CompareOperator::Lte,
            _ => unreachable!(),
        })
    }
}
