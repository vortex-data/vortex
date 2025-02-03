use std::cmp::Ordering;

use crate::stats::bound::{max, min, JoinResult};
use crate::stats::{LowerBound, Precision, Stat};

/// `StatType` define the bound of a given statistic. (e.g. `Max` is an upper bound),
/// this is used to extract the bound from a `Precision` value, (e.g. `p::bound<Max>()`).
pub trait StatType<T> {
    type Bound: StatBound<T>;

    const STAT: Stat;
}

/// `StatBound` defines the operations that can be performed on a bound.
/// The mains bounds are Upper (e.g. max) and Lower (e.g. min).
pub trait StatBound<T>: Sized {
    /// Creates a new bound from a Precision statistic.
    fn lift(value: Precision<T>) -> Self;

    /// Finds the smallest bound that covers both bounds.
    /// A.k.a. the `meet` of the bound.
    fn union(&self, other: &Self) -> Option<Self>;

    /// Refines the bounds to the most precise estimate we can make for that bound.
    /// If the bounds are disjoint, then the result is `JoinResult::None`.
    /// e.g. `Precision::Inexact(5)` and `Precision::Exact(6)` would result in `Precision::Inexact(5)`.
    /// A.k.a. the `join` of the bound.
    fn intersection(&self, other: &Self) -> Option<JoinResult<Self>>;

    // Returns the exact value from the bound, if that value is exact, otherwise `None`.
    fn as_exact(&self) -> Option<&T>;
}

/// This allows a stat with a `Precision` to be interpreted as a bound.
impl<T> Precision<T> {
    /// Applied the stat associated bound to the precision value
    pub fn bound<S: StatType<T>>(self) -> S::Bound {
        S::Bound::lift(self)
    }
}

impl<T: PartialOrd + Clone> LowerBound<T> {
    pub fn into_value(self) -> Precision<T> {
        self.0
    }
}

impl<T: PartialOrd + Clone> StatBound<T> for Precision<T> {
    fn lift(value: Precision<T>) -> Self {
        value
    }

    fn union(&self, other: &Self) -> Option<Self> {
        self.clone()
            .zip(other.clone())
            .map(|(lhs, rhs)| min(lhs, rhs))
            .transpose()
    }

    fn intersection(&self, other: &Self) -> Option<JoinResult<Self>> {
        Some(match (self, other) {
            (Precision::Exact(lhs), Precision::Exact(rhs)) => {
                if lhs.partial_cmp(rhs)? == Ordering::Equal {
                    JoinResult::Join(Precision::Exact(lhs.clone()))
                } else {
                    JoinResult::None
                }
            }
            (Precision::Exact(exact), Precision::Inexact(_))
            | (Precision::Inexact(_), Precision::Exact(exact)) => {
                JoinResult::Join(Precision::Inexact(exact.clone()))
            }
            (Precision::Inexact(lhs), Precision::Inexact(rhs)) => {
                JoinResult::Join(Precision::Inexact(max(lhs, rhs)?.clone()))
            }
        })
    }

    fn as_exact(&self) -> Option<&T> {
        match self {
            Precision::Exact(val) => Some(val),
            _ => None,
        }
    }
}
