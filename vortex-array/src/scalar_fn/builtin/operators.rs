// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use crate::dtype::NativePType;
use crate::proto::expr::boolean_opts::BooleanOp;
use crate::proto::expr::compare_opts::CompareOp;
use crate::scalar::NumericOperator;

/// The two boolean operators.
///
/// They obey [Kleene (three-valued) logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics),
/// so they are not strict: `false AND null` is `false`, and `true OR null` is `true`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BooleanOperator {
    /// Boolean AND (∧).
    And,
    /// Boolean OR (∨).
    Or,
}

impl Display for BooleanOperator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match self {
            BooleanOperator::And => "and",
            BooleanOperator::Or => "or",
        };
        Display::fmt(display, f)
    }
}

impl From<BooleanOperator> for BooleanOp {
    fn from(value: BooleanOperator) -> Self {
        match value {
            BooleanOperator::And => BooleanOp::And,
            BooleanOperator::Or => BooleanOp::Or,
        }
    }
}

impl From<BooleanOp> for BooleanOperator {
    fn from(value: BooleanOp) -> Self {
        match value {
            BooleanOp::And => BooleanOperator::And,
            BooleanOp::Or => BooleanOperator::Or,
        }
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
    /// Expression is greater than another.
    Gt,
    /// Expression is greater or equal to another.
    Gte,
    /// Expression is less than another.
    Lt,
    /// Expression is less or equal to another.
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

    /// Returns the total-order comparison that this operator applies to two native values.
    pub fn to_predicate<T: NativePType>(self) -> fn(T, T) -> bool {
        match self {
            CompareOperator::Eq => T::is_eq,
            CompareOperator::NotEq => |lhs, rhs| !lhs.is_eq(rhs),
            CompareOperator::Gt => T::is_gt,
            CompareOperator::Gte => T::is_ge,
            CompareOperator::Lt => T::is_lt,
            CompareOperator::Lte => T::is_le,
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

impl From<CompareOperator> for CompareOp {
    fn from(value: CompareOperator) -> Self {
        match value {
            CompareOperator::Eq => CompareOp::Eq,
            CompareOperator::NotEq => CompareOp::NotEq,
            CompareOperator::Gt => CompareOp::Gt,
            CompareOperator::Gte => CompareOp::Gte,
            CompareOperator::Lt => CompareOp::Lt,
            CompareOperator::Lte => CompareOp::Lte,
        }
    }
}

impl From<CompareOp> for CompareOperator {
    fn from(value: CompareOp) -> Self {
        match value {
            CompareOp::Eq => CompareOperator::Eq,
            CompareOp::NotEq => CompareOperator::NotEq,
            CompareOp::Gt => CompareOperator::Gt,
            CompareOp::Gte => CompareOperator::Gte,
            CompareOp::Lt => CompareOperator::Lt,
            CompareOp::Lte => CompareOperator::Lte,
        }
    }
}

/// An operator of one of the two-operand functions [`Compare`], [`Boolean`], and [`Numeric`].
///
/// Use it where one value can hold any infix operator, such as when a binding or a query engine
/// converts its own operator codes.
///
/// [`Compare`]: crate::scalar_fn::builtin::compare::Compare
/// [`Boolean`]: crate::scalar_fn::builtin::boolean::Boolean
/// [`Numeric`]: crate::scalar_fn::builtin::numeric::Numeric
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum BinaryOperator {
    /// A comparison operator.
    Compare(CompareOperator),
    /// A Kleene boolean operator.
    Boolean(BooleanOperator),
    /// An arithmetic operator.
    Numeric(NumericOperator),
}

impl From<CompareOperator> for BinaryOperator {
    fn from(operator: CompareOperator) -> Self {
        BinaryOperator::Compare(operator)
    }
}

impl From<BooleanOperator> for BinaryOperator {
    fn from(operator: BooleanOperator) -> Self {
        BinaryOperator::Boolean(operator)
    }
}

impl From<NumericOperator> for BinaryOperator {
    fn from(operator: NumericOperator) -> Self {
        BinaryOperator::Numeric(operator)
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
