use core::fmt;
use std::fmt::{Display, Formatter};

use vortex_array::compute;

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Operator {
    // comparison
    Eq,
    NotEq,
    Gt,
    Gte,
    Lt,
    Lte,
    // boolean algebra
    And,
    Or,
}

#[cfg(feature = "proto")]
mod proto {
    use vortex_error::VortexError;
    use vortex_proto::expr::kind::BinaryOp;

    use crate::Operator;

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
            }
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
            Operator::And | Operator::Or => None,
        }
    }

    pub fn logical_inverse(self) -> Option<Self> {
        match self {
            Operator::And => Some(Operator::Or),
            Operator::Or => Some(Operator::And),
            _ => None,
        }
    }

    /// Change the sides of the operator, where changing lhs and rhs won't change the result of the operation
    pub fn swap(self) -> Self {
        match self {
            Operator::Eq => Operator::Eq,
            Operator::NotEq => Operator::NotEq,
            Operator::Gt => Operator::Lt,
            Operator::Gte => Operator::Lte,
            Operator::Lt => Operator::Gt,
            Operator::Lte => Operator::Gte,
            Operator::And => Operator::And,
            Operator::Or => Operator::Or,
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
