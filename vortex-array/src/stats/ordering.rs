use std::cmp::Ordering;

use crate::stats::Precision::{Bound, Exact};
use crate::stats::{Precision, Stat};

pub trait PartialOrder<T: PartialOrd> {
    fn ordered(lhs: &T, other: &T) -> Option<bool>;

    fn lift(value: Precision<T>) -> Self;

    fn into_value(self) -> Precision<T>;
}

pub struct LowerBound<T>(Precision<T>);

impl<T> LowerBound<T> {
    pub fn is_exact(&self) -> bool {
        self.0.is_exact()
    }
}

impl<T: PartialOrd> LowerBound<T> {
    pub fn le(&self, value: &LowerBound<T>) -> Option<bool> {
        Some(match self.0.value().partial_cmp(value.0.value())? {
            Ordering::Less => true,
            Ordering::Equal => {
                // for a fixed value v. exact(v) <= bound(v) is true
                matches!((&self.0, &value.0), (Exact(_), _) | (Bound(_), Bound(_)))
            }
            Ordering::Greater => false,
        })
    }
}

impl<T: PartialOrd> PartialOrder<T> for LowerBound<T> {
    fn ordered(lhs: &T, rhs: &T) -> Option<bool> {
        PartialOrd::partial_cmp(lhs, rhs).map(|o| o != Ordering::Greater)
    }

    fn lift(value: Precision<T>) -> Self {
        Self(value)
    }

    fn into_value(self) -> Precision<T> {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct UpperBound<T>(Precision<T>);

pub trait GtOrd<Rhs: ?Sized = Self> {
    fn ge(&self, other: &Rhs) -> Option<bool>;
}

impl<T: PartialOrd> GtOrd for UpperBound<T> {
    fn ge(&self, other: &Self) -> Option<bool> {
        Some(match self.0.value().partial_cmp(other.0.value())? {
            Ordering::Less => false,
            Ordering::Equal => {
                // for a fixed value v. exact(v) >= bound(v) is true
                matches!((&self.0, &other.0), (Exact(_), _) | (Bound(_), Bound(_)))
            }
            Ordering::Greater => true,
        })
    }
}

impl<T: PartialOrd> GtOrd<T> for UpperBound<T> {
    fn ge(&self, other: &T) -> Option<bool> {
        Some(self.0.value() >= other)
    }
}

impl<T: PartialOrd> PartialOrder<T> for UpperBound<T> {
    fn ordered(lhs: &T, rhs: &T) -> Option<bool> {
        PartialOrd::partial_cmp(lhs, rhs).map(|o| o != Ordering::Less)
    }

    fn lift(value: Precision<T>) -> Self {
        Self(value)
    }

    fn into_value(self) -> Precision<T> {
        self.0
    }
}
