use std::cmp::Ordering;

use vortex_error::{VortexError, VortexResult};

use crate::partial_ord::{partial_max, partial_min};
use crate::stats::Precision::{Exact, Inexact};
use crate::stats::{Precision, StatBound};

/// Interpret the value as a lower bound.
/// These form a partial order over successively more precise bounds
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

/// The result of the fallible intersection of two bound, defined to avoid `Option`
/// `IntersectionResult` mixup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntersectionResult<T> {
    /// An intersection result was found
    Value(T),
    /// Values has no intersection.
    None,
}

impl<T> IntersectionResult<T> {
    pub fn ok_or_else<F>(self, err: F) -> VortexResult<T>
    where
        F: FnOnce() -> VortexError,
    {
        match self {
            IntersectionResult::Value(v) => Ok(v),
            IntersectionResult::None => Err(err()),
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
            (Exact(lhs), Exact(rhs)) => Exact(partial_min(lhs, rhs)?.clone()),
            (Inexact(lhs), Inexact(rhs)) => Inexact(partial_min(lhs, rhs)?.clone()),
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
    fn intersection(&self, other: &Self) -> Option<IntersectionResult<LowerBound<T>>> {
        Some(match (&self.0, &other.0) {
            (Exact(lhs), Exact(rhs)) => {
                if lhs == rhs {
                    IntersectionResult::Value(LowerBound(Exact(lhs.clone())))
                } else {
                    // The two intervals do not overlap
                    IntersectionResult::None
                }
            }
            (Inexact(lhs), Inexact(rhs)) => {
                IntersectionResult::Value(LowerBound(Inexact(partial_max(lhs, rhs)?.clone())))
            }
            (Inexact(lhs), Exact(rhs)) => {
                if rhs >= lhs {
                    IntersectionResult::Value(LowerBound(Exact(rhs.clone())))
                } else {
                    // The two intervals do not overlap
                    IntersectionResult::None
                }
            }
            (Exact(lhs), Inexact(rhs)) => {
                if rhs <= lhs {
                    IntersectionResult::Value(LowerBound(Exact(rhs.clone())))
                } else {
                    // The two intervals do not overlap
                    IntersectionResult::None
                }
            }
        })
    }

    fn as_exact(&self) -> Option<&T> {
        self.0.as_exact()
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
            (Exact(lhs), Exact(rhs)) => Exact(partial_max(lhs, rhs)?.clone()),
            (Inexact(lhs), Inexact(rhs)) => Inexact(partial_max(lhs, rhs)?.clone()),
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

    fn intersection(&self, other: &Self) -> Option<IntersectionResult<UpperBound<T>>> {
        Some(match (&self.0, &other.0) {
            (Exact(lhs), Exact(rhs)) => {
                if lhs == rhs {
                    IntersectionResult::Value(UpperBound(Exact(lhs.clone())))
                } else {
                    // The two intervals do not overlap
                    IntersectionResult::None
                }
            }
            (Inexact(lhs), Inexact(rhs)) => {
                IntersectionResult::Value(UpperBound(Inexact(partial_min(lhs, rhs)?.clone())))
            }
            (Inexact(lhs), Exact(rhs)) => {
                if rhs <= lhs {
                    IntersectionResult::Value(UpperBound(Exact(rhs.clone())))
                } else {
                    // The two intervals do not overlap
                    IntersectionResult::None
                }
            }
            (Exact(lhs), Inexact(rhs)) => {
                if rhs >= lhs {
                    IntersectionResult::Value(UpperBound(Exact(lhs.clone())))
                } else {
                    // The two intervals do not overlap
                    IntersectionResult::None
                }
            }
        })
    }

    fn as_exact(&self) -> Option<&T> {
        self.0.as_exact()
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

#[cfg(test)]
mod tests {
    use crate::stats::bound::IntersectionResult;
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
    fn test_upper_bound_union() {
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

    #[test]
    fn test_upper_bound_intersection() {
        let ub1: UpperBound<i32> = UpperBound(Precision::exact(10i32));
        let ub2 = UpperBound(Precision::inexact(12i32));

        assert_eq!(
            Some(IntersectionResult::Value(ub1.clone())),
            ub1.intersection(&ub2)
        );

        let ub1: UpperBound<i32> = UpperBound(Precision::exact(13i32));
        let ub2 = UpperBound(Precision::inexact(12i32));

        assert_eq!(Some(IntersectionResult::None), ub1.intersection(&ub2));
    }
}
