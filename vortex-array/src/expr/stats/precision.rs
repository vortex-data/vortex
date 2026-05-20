// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect;

use crate::dtype::DType;
use crate::expr::stats::IntersectionResult;
use crate::expr::stats::StatBound;
use crate::expr::stats::StatType;
use crate::partial_ord::partial_min;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A statistic has a precision `Exact` or `Inexact`. This represents uncertainty in that value.
/// Exact values are computed, where can inexact values are likely inferred from compute functions.
///
/// Inexact statistics form a range of possible values that the statistic could be.
/// This is statistic specific, for max this will be an upper bound. Meaning that the actual max
/// in an array is guaranteed to be less than or equal to the inexact value, but equal to the exact
/// value.
#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub enum Precision<T> {
    Exact(T),
    Inexact(T),
    #[default]
    Absent,
}

impl<T> Precision<Option<T>> {
    /// Transpose the `Precision<Option<T>>` into `Option<Precision<T>>`.
    pub fn transpose(self) -> Option<Precision<T>> {
        use Precision::*;

        match self {
            Exact(Some(x)) => Some(Exact(x)),
            Inexact(Some(x)) => Some(Inexact(x)),
            Absent => Some(Absent),
            Exact(None) | Inexact(None) => None,
        }
    }
}

impl<T, E> Precision<Result<T, E>> {
    /// Transpose a `Precision<Result<T, E>>` into a `Result<Precision<T>, E>`.
    pub fn transpose(self) -> Result<Precision<T>, E> {
        match self {
            Self::Exact(value) => value.map(Precision::Exact),
            Self::Inexact(value) => value.map(Precision::Inexact),
            Self::Absent => Ok(Precision::Absent),
        }
    }
}

impl<T> Precision<T>
where
    T: Copy,
{
    pub fn to_inexact(&self) -> Self {
        use Precision::*;

        match self {
            Exact(v) | Inexact(v) => Inexact(*v),
            Absent => Absent,
        }
    }
}

impl<T> Precision<T> {
    /// Creates an exact value
    pub fn exact<S: Into<T>>(s: S) -> Precision<T> {
        Self::Exact(s.into())
    }

    /// Creates an inexact value
    pub fn inexact<S: Into<T>>(s: S) -> Precision<T> {
        Self::Inexact(s.into())
    }

    /// Pushed the ref into the Precision enum
    pub fn as_ref(&self) -> Precision<&T> {
        use Precision::*;

        match self {
            Exact(val) => Exact(val),
            Inexact(val) => Inexact(val),
            Absent => Absent,
        }
    }

    /// Converts `self` into an inexact bound
    pub fn into_inexact(self) -> Self {
        use Precision::*;

        match self {
            Exact(v) | Inexact(v) => Inexact(v),
            Absent => Absent,
        }
    }

    /// Returns the exact value from the bound, if that value is inexact, otherwise `None`.
    pub fn as_exact(self) -> Option<T> {
        match self {
            Self::Exact(val) => Some(val),
            _ => None,
        }
    }

    /// Returns the exact value from the bound, if that value is inexact, otherwise `None`.
    pub fn as_inexact(self) -> Option<T> {
        match self {
            Self::Inexact(val) => Some(val),
            _ => None,
        }
    }

    /// Returns true when representing an exact value.
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact(_))
    }

    /// Returns true when representing an absent value
    pub fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }

    /// Map the value of either precision value
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Precision<U> {
        use Precision::*;

        match self {
            Exact(value) => Exact(f(value)),
            Inexact(value) => Inexact(f(value)),
            Absent => Absent,
        }
    }

    /// Zip two `Precision` values into a tuple, keeping the inexactness if any.
    pub fn zip<U>(self, other: Precision<U>) -> Precision<(T, U)> {
        use Precision::*;

        match (self, other) {
            (Exact(lhs), Exact(rhs)) => Exact((lhs, rhs)),
            (Inexact(lhs), Exact(rhs))
            | (Exact(lhs), Inexact(rhs))
            | (Inexact(lhs), Inexact(rhs)) => Inexact((lhs, rhs)),
            (Absent, _) | (_, Absent) => Absent,
        }
    }

    /// Unwrap the underlying value
    pub fn into_inner(self) -> Option<T> {
        use Precision::*;

        match self {
            Exact(val) | Inexact(val) => Some(val),
            Absent => None,
        }
    }
}

impl<T: Display> Display for Precision<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use Precision::*;

        match self {
            Exact(v) => {
                write!(f, "{v}")
            }
            Inexact(v) => {
                write!(f, "~{v}")
            }
            Absent => {
                write!(f, "{{empty}}")
            }
        }
    }
}

impl<T: PartialEq> PartialEq<T> for Precision<T> {
    fn eq(&self, other: &T) -> bool {
        match self {
            Self::Exact(v) => v == other,
            _ => false,
        }
    }
}

impl Precision<ScalarValue> {
    /// Convert this [`Precision<ScalarValue>`] into a [`Precision<Scalar>`] with the given
    /// [`DType`].
    pub fn into_scalar(self, dtype: DType) -> Precision<Scalar> {
        self.map(|v| {
            Scalar::try_new(dtype, Some(v)).vortex_expect("`Precision<ScalarValue>` was invalid")
        })
    }
}

impl Precision<&ScalarValue> {
    /// Convert this [`Precision<&ScalarValue>`] into a [`Precision<Scalar>`] with the given
    /// [`DType`].
    pub fn into_scalar(self, dtype: DType) -> Precision<Scalar> {
        self.map(|v| {
            Scalar::try_new(dtype, Some(v.clone()))
                .vortex_expect("`Precision<ScalarValue>` was invalid")
        })
    }
}

/// This allows a stat with a `Precision` to be interpreted as a bound.
impl<T> Precision<T> {
    /// Applied the stat associated bound to the precision value
    pub fn bound<S: StatType<T>>(self) -> Option<S::Bound> {
        if self.is_absent() {
            None
        } else {
            Some(S::Bound::lift(self))
        }
    }
}

impl<T: PartialOrd + Clone> StatBound<T> for Precision<T> {
    fn lift(value: Precision<T>) -> Self {
        value
    }

    fn into_value(self) -> Precision<T> {
        self
    }

    fn union(&self, other: &Self) -> Option<Self> {
        self.clone()
            .zip(other.clone())
            .map(|(lhs, rhs)| partial_min(&lhs, &rhs).cloned())
            .transpose()
    }

    fn intersection(&self, other: &Self) -> Option<IntersectionResult<Self>> {
        Some(match (self, other) {
            (Precision::Exact(lhs), Precision::Exact(rhs)) => {
                if lhs.partial_cmp(rhs)?.is_eq() {
                    IntersectionResult::Value(Precision::Exact(lhs.clone()))
                } else {
                    IntersectionResult::Empty
                }
            }
            (Precision::Exact(exact), Precision::Inexact(inexact))
            | (Precision::Inexact(inexact), Precision::Exact(exact)) => {
                if exact.partial_cmp(inexact)?.is_lt() {
                    IntersectionResult::Value(Precision::Inexact(exact.clone()))
                } else {
                    IntersectionResult::Value(Precision::Exact(exact.clone()))
                }
            }
            (Precision::Inexact(lhs), Precision::Inexact(rhs)) => {
                IntersectionResult::Value(Precision::Inexact(partial_min(lhs, rhs)?.clone()))
            }
            (_, Precision::Absent) | (Precision::Absent, _) => IntersectionResult::Empty,
        })
    }

    fn to_exact(&self) -> Option<&T> {
        match self {
            Precision::Exact(val) => Some(val),
            _ => None,
        }
    }
}
