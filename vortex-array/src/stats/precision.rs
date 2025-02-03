use std::fmt::{Debug, Display, Formatter};

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_scalar::{Scalar, ScalarValue};

use crate::stats::precision::Precision::{Exact, Inexact};

/// A statistic has a precision `Exact` or `Inexact`. This represents uncertainty in that value.
/// Exact values are computed, where can inexact values are likely inferred from compute functions.
///
/// Inexact statistics form a range of possible values that the statistic could be.
/// This is statistic specific, for max this will be an upper bound. Meaning that the actual max
/// in an array is guaranteed to be less than or equal to the inexact value, but equal to the exact
/// value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Precision<T> {
    Exact(T),
    Inexact(T),
}

impl<T> Precision<Option<T>> {
    /// Transpose the `Option<Precision<T>>` into `Option<Precision<T>>`.
    pub fn transpose(self) -> Option<Precision<T>> {
        match self {
            Exact(Some(x)) => Some(Exact(x)),
            Inexact(Some(x)) => Some(Inexact(x)),
            _ => None,
        }
    }
}

impl<T: Clone> Precision<T> {
    // Coverts an exact to an inexact bound.
    pub fn into_inexact(self) -> Self {
        match self {
            Exact(val) => Inexact(val),
            Inexact(_) => self,
        }
    }
}

impl<T> Precision<T> {
    /// Creates an exact value
    pub fn exact<S: Into<T>>(s: S) -> Precision<T> {
        Exact(s.into())
    }

    /// Creates an inexact value
    pub fn inexact<S: Into<T>>(s: S) -> Precision<T> {
        Inexact(s.into())
    }

    /// Pushed the ref into the Precision enum
    pub fn as_ref(&self) -> Precision<&T> {
        match self {
            Exact(val) => Exact(val),
            Inexact(val) => Inexact(val),
        }
    }

    /// Returns the exact value from the bound, if that value is inexact, otherwise `None`.
    pub fn some_exact(self) -> Option<T> {
        match self {
            Exact(val) => Some(val),
            _ => None,
        }
    }

    /// Returns the exact value from the bound, if that value is inexact, otherwise `None`.
    pub fn some_inexact(self) -> Option<T> {
        match self {
            Inexact(val) => Some(val),
            _ => None,
        }
    }

    /// True iff self == Exact(_)
    pub fn is_exact(&self) -> bool {
        matches!(self, Exact(_))
    }

    /// Map the value of either precision value
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Precision<U> {
        match self {
            Exact(value) => Exact(f(value)),
            Inexact(value) => Inexact(f(value)),
        }
    }

    /// Zip two `Precision` values into a tuple, keeping the inexactness if any.
    pub fn zip<U>(self, other: Precision<U>) -> Precision<(T, U)> {
        match (self, other) {
            (Exact(lhs), Exact(rhs)) => Exact((lhs, rhs)),
            (Inexact(lhs), Exact(rhs))
            | (Exact(lhs), Inexact(rhs))
            | (Inexact(lhs), Inexact(rhs)) => Inexact((lhs, rhs)),
        }
    }

    /// Similar to `map` but handles fucntions that can fail.
    pub fn try_map<U, F: FnOnce(T) -> VortexResult<U>>(self, f: F) -> VortexResult<Precision<U>> {
        let precision = match self {
            Exact(value) => Exact(f(value)?),
            Inexact(value) => Inexact(f(value)?),
        };
        Ok(precision)
    }

    pub(crate) fn into_value(self) -> T {
        match self {
            Exact(val) | Inexact(val) => val,
        }
    }
}

impl<T: Display> Display for Precision<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Exact(v) => {
                write!(f, "{}", v)
            }
            Inexact(v) => {
                write!(f, "~{}", v)
            }
        }
    }
}

impl<T: PartialEq> PartialEq<T> for Precision<T> {
    fn eq(&self, other: &T) -> bool {
        match self {
            Exact(v) => v == other,
            _ => false,
        }
    }
}

impl Precision<ScalarValue> {
    pub fn into_scalar(self, dtype: DType) -> Precision<Scalar> {
        self.map(|v| Scalar::new(dtype, v))
    }
}

impl Precision<&ScalarValue> {
    pub fn into_scalar(self, dtype: DType) -> Precision<Scalar> {
        self.map(|v| Scalar::new(dtype, v.clone()))
    }
}
