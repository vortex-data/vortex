use std::cmp::Ordering;

use vortex_error::{VortexError, VortexResult};
use Precision::Inexact;

use crate::stats::Precision::Exact;
use crate::stats::{Precision, StatBound};

/// Interpret the value as a lower bound.
/// These format a partial order over successively more precise bounds
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowerBound<T>(pub(crate) Precision<T>);

impl<T> LowerBound<T> {
    pub(crate) fn min_value(self) -> T {
        self.0.into_value()
    }
}

impl<T> LowerBound<T> {
    pub fn is_exact(&self) -> bool {
        self.0.is_exact()
    }
}

/// The result of the fallible intersection of two bound, defined to avoid Option JoinResult mixup.
pub enum JoinResult<T> {
    Join(T),
    None,
}

impl<T> JoinResult<T> {
    pub fn ok_or_else<F>(self, err: F) -> VortexResult<T>
    where
        F: FnOnce() -> VortexError,
    {
        match self {
            JoinResult::Join(v) => Ok(v),
            JoinResult::None => Err(err()),
        }
    }
}

impl<T: PartialOrd + Clone> StatBound<T> for LowerBound<T> {
    fn lift(value: Precision<T>) -> Self {
        Self(value)
    }

    // The meet or tightest covering bound
    fn union(&self, other: &Self) -> Option<LowerBound<T>> {
        Some(LowerBound(match (&self.0, &other.0) {
            (Exact(lhs), Exact(rhs)) => Exact(min(lhs, rhs)?.clone()),
            (Inexact(lhs), Inexact(rhs)) => Inexact(min(lhs, rhs)?.clone()),
            (Inexact(lhs), Exact(rhs)) => {
                if rhs <= lhs {
                    Exact(rhs.clone())
                } else {
                    Inexact(lhs.clone())
                }
            }
            (Exact(lhs), Inexact(rhs)) => {
                if rhs >= lhs {
                    Exact(lhs.clone())
                } else {
                    Inexact(rhs.clone())
                }
            }
        }))
    }

    // The join of the smallest intersection of both bounds, this can fail.
    fn intersection(&self, other: &Self) -> Option<JoinResult<LowerBound<T>>> {
        Some(match (&self.0, &other.0) {
            (Exact(lhs), Exact(rhs)) => {
                if lhs == rhs {
                    JoinResult::Join(LowerBound(Exact(lhs.clone())))
                } else {
                    // The two intervals do not overlap
                    JoinResult::None
                }
            }
            (Inexact(lhs), Inexact(rhs)) => {
                JoinResult::Join(LowerBound(Inexact(max(lhs, rhs)?.clone())))
            }
            (Inexact(lhs), Exact(rhs)) => {
                if rhs >= lhs {
                    JoinResult::Join(LowerBound(Exact(rhs.clone())))
                } else {
                    // The two intervals do not overlap
                    JoinResult::None
                }
            }
            (Exact(lhs), Inexact(rhs)) => {
                if rhs <= lhs {
                    JoinResult::Join(LowerBound(Exact(rhs.clone())))
                } else {
                    // The two intervals do not overlap
                    JoinResult::None
                }
            }
        })
    }
}

impl<T: PartialOrd> PartialEq<T> for LowerBound<T> {
    fn eq(&self, other: &T) -> bool {
        match &self.0 {
            Exact(lhs) => lhs == other,
            _ => false,
        }
    }
}

// We can only compare exact values with values and Precision::inexact values can only be greater than a value
impl<T: PartialOrd> PartialOrd<T> for LowerBound<T> {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        match &self.0 {
            Exact(lhs) => lhs.partial_cmp(other),
            Inexact(lhs) => {
                lhs.partial_cmp(other).and_then(
                    |o| {
                        if o == Ordering::Less {
                            None
                        } else {
                            Some(o)
                        }
                    },
                )
            }
        }
    }
}

/// Interpret the value as an upper bound, see `LowerBound` for more details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpperBound<T>(pub(crate) Precision<T>);

impl<T> UpperBound<T> {
    pub(crate) fn max_value(self) -> T {
        self.0.into_value()
    }
}

impl<T> UpperBound<T> {
    pub fn into_value(self) -> Precision<T> {
        self.0
    }
}

impl<T: PartialOrd + Clone> StatBound<T> for UpperBound<T> {
    fn lift(value: Precision<T>) -> Self {
        Self(value)
    }

    /// The meet or tightest covering bound
    fn union(&self, other: &Self) -> Option<UpperBound<T>> {
        Some(UpperBound(match (&self.0, &other.0) {
            (Exact(lhs), Exact(rhs)) => Exact(max(lhs, rhs)?.clone()),
            (Inexact(lhs), Inexact(rhs)) => Inexact(max(lhs, rhs)?.clone()),
            (Inexact(lhs), Exact(rhs)) => {
                if rhs >= lhs {
                    Exact(rhs.clone())
                } else {
                    Inexact(lhs.clone())
                }
            }
            (Exact(lhs), Inexact(rhs)) => {
                if rhs <= lhs {
                    Exact(lhs.clone())
                } else {
                    Inexact(rhs.clone())
                }
            }
        }))
    }

    fn intersection(&self, other: &Self) -> Option<JoinResult<UpperBound<T>>> {
        Some(match (&self.0, &other.0) {
            (Exact(lhs), Exact(rhs)) => {
                if lhs == rhs {
                    JoinResult::Join(UpperBound(Exact(lhs.clone())))
                } else {
                    // The two intervals do not overlap
                    JoinResult::None
                }
            }
            (Inexact(lhs), Inexact(rhs)) => {
                JoinResult::Join(UpperBound(Inexact(min(lhs, rhs)?.clone())))
            }
            (Inexact(lhs), Exact(rhs)) => {
                if rhs <= lhs {
                    JoinResult::Join(UpperBound(Exact(rhs.clone())))
                } else {
                    // The two intervals do not overlap
                    JoinResult::None
                }
            }
            (Exact(lhs), Inexact(rhs)) => {
                if rhs <= lhs {
                    JoinResult::Join(UpperBound(Exact(lhs.clone())))
                } else {
                    // The two intervals do not overlap
                    JoinResult::None
                }
            }
        })
    }
}

impl<T: PartialOrd> PartialEq<T> for UpperBound<T> {
    fn eq(&self, other: &T) -> bool {
        match &self.0 {
            Exact(lhs) => lhs == other,
            _ => false,
        }
    }
}

// We can only compare exact values with values and Precision::inexact values can only be greater than a value
impl<T: PartialOrd> PartialOrd<T> for UpperBound<T> {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        match &self.0 {
            Exact(lhs) => lhs.partial_cmp(other),
            Inexact(lhs) => lhs.partial_cmp(other).and_then(|o| {
                if o == Ordering::Greater {
                    None
                } else {
                    Some(o)
                }
            }),
        }
    }
}

#[inline]
pub fn min<T: PartialOrd>(a: T, b: T) -> Option<T> {
    if a.partial_cmp(&b)? == Ordering::Less {
        Some(a)
    } else {
        Some(b)
    }
}

#[inline]
pub fn max<T: PartialOrd>(a: T, b: T) -> Option<T> {
    if a.partial_cmp(&b)? == Ordering::Greater {
        Some(a)
    } else {
        Some(b)
    }
}

#[cfg(test)]
mod tests {

    use crate::stats::{Precision, StatBound, UpperBound};

    #[test]
    fn test_upper_bound_cmp() {
        let ub = UpperBound(Precision::exact(10i32));

        assert_eq!(ub, 10);
        assert!(ub > 9);
        assert!(ub <= 10);
        assert!(ub <= 10);

        let ub = UpperBound(Precision::inexact(10i32));

        assert_ne!(ub, 10);
        assert!(ub < 11);
        // We cannot say anything about a value in the bound.
        assert!(!(ub >= 9));
    }

    #[test]
    fn test_upper_bound_meet() {
        let ub1: UpperBound<i32> = UpperBound(Precision::exact(10i32));
        let ub2 = UpperBound(Precision::exact(12i32));

        assert_eq!(Some(ub2.clone()), ub1.union(&ub2));

        let ub1: UpperBound<i32> = UpperBound(Precision::inexact(10i32));
        let ub2 = UpperBound(Precision::exact(12i32));

        assert_eq!(Some(ub2.clone()), ub1.union(&ub2));

        let ub1: UpperBound<i32> = UpperBound(Precision::exact(10i32));
        let ub2 = UpperBound(Precision::inexact(12i32));

        assert_eq!(Some(ub2.clone()), ub1.union(&ub2));

        let ub1: UpperBound<i32> = UpperBound(Precision::inexact(10i32));
        let ub2 = UpperBound(Precision::inexact(12i32));

        assert_eq!(Some(ub2.clone()), ub1.union(&ub2))
    }
}
