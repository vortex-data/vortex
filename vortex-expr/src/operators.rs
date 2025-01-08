use core::fmt;
use std::fmt::{Display, Formatter};

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
}
