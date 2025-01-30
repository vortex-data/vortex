use std::cmp::{max, Ordering};

use crate::stats::Precision::{Bound, Exact};
use crate::stats::{BoundDirection, DirectionalBound, Precision, Stat};

#[derive(Debug, Clone, PartialEq)]
pub struct LowerBound<T>(pub(crate) Precision<T>);

impl<T> LowerBound<T> {
    pub fn is_exact(&self) -> bool {
        self.0.is_exact()
    }
}

impl<T> LowerBound<T> {
    pub fn into_value(self) -> Precision<T> {
        self.0
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

impl<T: PartialOrd + Clone> LowerBound<T> {
    pub fn meet(&self, other: &Self) -> Option<LowerBound<T>> {
        Some(LowerBound(match (&self.0, &other.0) {
            (Exact(lhs), Exact(rhs)) => Exact(try_min(lhs, rhs)?.clone()),
            (Bound(lhs), Bound(rhs)) => Bound(try_min(lhs, rhs)?.clone()),
            (Bound(lhs), Exact(rhs)) => {
                if rhs <= lhs {
                    Exact(rhs.clone())
                } else {
                    Bound(lhs.clone())
                }
            }
            (Exact(lhs), Bound(rhs)) => {
                if lhs <= rhs {
                    Exact(lhs.clone())
                } else {
                    Bound(rhs.clone())
                }
            }
        }))
    }
}

impl<T: PartialOrd + PartialEq> PartialEq<T> for DirectionalBound<T> {
    fn eq(&self, other: &T) -> bool {
        match &self.value {
            Exact(val) => val == other,
            _ => false,
        }
    }
}

impl<T: PartialOrd + PartialEq> PartialOrd<T> for DirectionalBound<T> {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        match self.direction {
            BoundDirection::Lower => match &self.value {
                Exact(lhs) => lhs.partial_cmp(other),
                Bound(lhs) => lhs.partial_cmp(other).and_then(|o| {
                    if o == Ordering::Less {
                        None
                    } else {
                        Some(o)
                    }
                }),
            },
            BoundDirection::Upper => match &self.value {
                Exact(lhs) => lhs.partial_cmp(other),
                Bound(lhs) => lhs.partial_cmp(other).and_then(|o| {
                    if o == Ordering::Greater {
                        None
                    } else {
                        Some(o)
                    }
                }),
            },
            BoundDirection::Neither => match &self.value {
                Exact(lhs) => lhs.partial_cmp(other),
                Bound(lhs) => lhs.partial_cmp(other),
            },
        }
    }
}

// impl<T: PartialOrd> PartialOrder<T> for LowerBound<T> {
//     fn ordered(lhs: &T, rhs: &T) -> Option<bool> {
//         PartialOrd::partial_cmp(lhs, rhs).map(|o| o != Ordering::Greater)
//     }
//
//     fn lift(value: Precision<T>) -> Self {
//         Self(value)
//     }
//

// }

#[derive(Debug, Clone)]
pub struct UpperBound<T>(pub(crate) Precision<T>);

impl<T> UpperBound<T> {
    pub fn into_value(self) -> Precision<T> {
        self.0
    }
}

pub fn try_max<'a, T: PartialOrd + Clone>(lhs: &'a T, rhs: &'a T) -> Option<&'a T> {
    if lhs.partial_cmp(rhs)? == Ordering::Greater {
        Some(lhs)
    } else {
        Some(rhs)
    }
}

pub fn try_min<'a, T: PartialOrd + Clone>(lhs: &'a T, rhs: &'a T) -> Option<&'a T> {
    if lhs.partial_cmp(rhs)? == Ordering::Less {
        Some(lhs)
    } else {
        Some(rhs)
    }
}

impl<T: PartialEq> PartialEq for UpperBound<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.structural_eq(&other.0)
    }
}

impl<T: PartialOrd + Clone> UpperBound<T> {
    pub fn meet(&self, other: &Self) -> Option<UpperBound<T>> {
        Some(UpperBound(match (&self.0, &other.0) {
            (Exact(lhs), Exact(rhs)) => Exact(try_max(lhs, rhs)?.clone()),
            (Bound(lhs), Bound(rhs)) => Bound(try_max(lhs, rhs)?.clone()),
            (Bound(lhs), Exact(rhs)) => {
                if rhs >= lhs {
                    Exact(rhs.clone())
                } else {
                    Bound(lhs.clone())
                }
            }
            (Exact(lhs), Bound(rhs)) => {
                if lhs >= rhs {
                    Exact(lhs.clone())
                } else {
                    Bound(rhs.clone())
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

// We can only compare exact bound with values and bounded bounds can only be greater than a value
impl<T: PartialOrd> PartialOrd<T> for UpperBound<T> {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        match self {
            UpperBound(Exact(lhs)) => lhs.partial_cmp(other),
            UpperBound(Bound(lhs)) => lhs.partial_cmp(other).and_then(|o| {
                if o == Ordering::Greater {
                    None
                } else {
                    Some(o)
                }
            }),
        }
    }
}

// impl<T: PartialOrd> PartialOrder<T> for UpperBound<T> {
//     fn ordered(lhs: &T, rhs: &T) -> Option<bool> {
//         PartialOrd::partial_cmp(lhs, rhs).map(|o| o != Ordering::Less)
//     }
//
//     fn lift(value: Precision<T>) -> Self {
//         Self(value)
//     }
//
//     fn into_value(self) -> Precision<T> {
//         self.0
//     }
// }

#[cfg(test)]
mod tests {
    use std::io::empty;

    use crate::stats::{bound, exact, UpperBound};

    #[test]
    fn test_upper_bound_cmp() {
        let ub = UpperBound(exact(10i32));

        assert_eq!(ub, 10);
        assert!(ub > 9);
        assert!(ub <= 10);
        assert!(ub <= 10);

        let ub = UpperBound(bound(10i32));

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

        let ub1: UpperBound<i32> = UpperBound(bound(10i32));
        let ub2 = UpperBound(exact(12i32));

        assert_eq!(Some(ub2.clone()), ub1.meet(&ub2));

        let ub1: UpperBound<i32> = UpperBound(exact(10i32));
        let ub2 = UpperBound(bound(12i32));

        assert_eq!(Some(ub2.clone()), ub1.meet(&ub2));

        let ub1: UpperBound<i32> = UpperBound(bound(10i32));
        let ub2 = UpperBound(bound(12i32));

        assert_eq!(Some(ub2.clone()), ub1.meet(&ub2))
    }
}
