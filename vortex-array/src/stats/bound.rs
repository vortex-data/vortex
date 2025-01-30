use std::cmp::{max, Ordering};

use vortex_scalar::Scalar;

use crate::stats::Precision::{Exact, Inexact};
use crate::stats::{Precision, Stat};

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

impl<T: PartialOrd + Clone> LowerBound<T> {
    // The meet or tightest covering bound
    pub fn meet(&self, other: &Self) -> Option<LowerBound<T>> {
        Some(LowerBound(match (&self.0, &other.0) {
            (Exact(lhs), Exact(rhs)) => Exact(try_min(lhs, rhs)?.clone()),
            (Inexact(lhs), Inexact(rhs)) => Inexact(try_min(lhs, rhs)?.clone()),
            (Inexact(lhs), Exact(rhs)) => {
                if rhs <= lhs {
                    Exact(rhs.clone())
                } else {
                    Inexact(lhs.clone())
                }
            }
            (Exact(lhs), Inexact(rhs)) => {
                if lhs <= rhs {
                    Exact(lhs.clone())
                } else {
                    Inexact(rhs.clone())
                }
            }
        }))
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

// We can only compare exact values with values and inexact values can only be greater than a value
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

/// Interpret the value as an upper bound
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

impl<T: PartialOrd + Clone> UpperBound<T> {
    /// The meet or tightest covering bound
    pub fn meet(&self, other: &Self) -> Option<UpperBound<T>> {
        Some(UpperBound(match (&self.0, &other.0) {
            (Exact(lhs), Exact(rhs)) => Exact(try_max(lhs, rhs)?.clone()),
            (Inexact(lhs), Inexact(rhs)) => Inexact(try_max(lhs, rhs)?.clone()),
            (Inexact(lhs), Exact(rhs)) => {
                if rhs >= lhs {
                    Exact(rhs.clone())
                } else {
                    Inexact(lhs.clone())
                }
            }
            (Exact(lhs), Inexact(rhs)) => {
                if lhs >= rhs {
                    Exact(lhs.clone())
                } else {
                    Inexact(rhs.clone())
                }
            }
        }))
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

// We can only compare exact values with values and inexact values can only be greater than a value
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

fn try_max<'a, T: PartialOrd + Clone>(lhs: &'a T, rhs: &'a T) -> Option<&'a T> {
    if lhs.partial_cmp(rhs)? == Ordering::Greater {
        Some(lhs)
    } else {
        Some(rhs)
    }
}

fn try_min<'a, T: PartialOrd + Clone>(lhs: &'a T, rhs: &'a T) -> Option<&'a T> {
    if lhs.partial_cmp(rhs)? == Ordering::Less {
        Some(lhs)
    } else {
        Some(rhs)
    }
}

#[cfg(test)]
mod tests {
    use std::io::empty;

    use crate::stats::{exact, inexact, UpperBound};

    #[test]
    fn test_upper_bound_cmp() {
        let ub = UpperBound(exact(10i32));

        assert_eq!(ub, 10);
        assert!(ub > 9);
        assert!(ub <= 10);
        assert!(ub <= 10);

        let ub = UpperBound(inexact(10i32));

        assert_ne!(ub, 10);
        assert!(ub < 11);
        // We cannot say anything about a value in the bound.
        assert!(!(ub >= 9));
    }

    #[test]
    fn test_upper_bound_meet() {
        let ub1: UpperBound<i32> = UpperBound(exact(10i32));
        let ub2 = UpperBound(exact(12i32));

        assert_eq!(Some(ub2.clone()), ub1.meet(&ub2));

        let ub1: UpperBound<i32> = UpperBound(inexact(10i32));
        let ub2 = UpperBound(exact(12i32));

        assert_eq!(Some(ub2.clone()), ub1.meet(&ub2));

        let ub1: UpperBound<i32> = UpperBound(exact(10i32));
        let ub2 = UpperBound(inexact(12i32));

        assert_eq!(Some(ub2.clone()), ub1.meet(&ub2));

        let ub1: UpperBound<i32> = UpperBound(inexact(10i32));
        let ub2 = UpperBound(inexact(12i32));

        assert_eq!(Some(ub2.clone()), ub1.meet(&ub2))
    }
}
