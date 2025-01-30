use std::fmt::{Debug, Display, Formatter};

use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::stats::precision::Precision::{Exact, Inexact};
use crate::stats::{LowerBound, Stat, UpperBound};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Precision<T> {
    Exact(T),
    Inexact(T),
}

pub fn exact<S: Into<T>, T>(s: S) -> Precision<T> {
    Exact(s.into())
}

pub fn inexact<S: Into<T>, T>(s: S) -> Precision<T> {
    Inexact(s.into())
}

impl<T> Precision<Option<T>> {
    pub fn transpose(self) -> Option<Precision<T>> {
        match self {
            Exact(Some(x)) => Some(Exact(x)),
            Inexact(Some(x)) => Some(Inexact(x)),
            _ => None,
        }
    }
}

impl<T> Precision<T> {
    pub fn ok_exact(self) -> Option<T> {
        match self {
            Exact(val) => Some(val),
            _ => None,
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(self, Exact(_))
    }

    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Precision<U> {
        match self {
            Exact(value) => Exact(f(value)),
            Inexact(value) => Inexact(f(value)),
        }
    }

    // Similar to option and then, but if either value is inexact, then the whole value is inexact.
    pub fn and_then_prefer_inexact<U, F: FnOnce(T) -> Precision<U>>(self, f: F) -> Precision<U> {
        match self {
            Exact(value) => f(value),
            Inexact(value) => match f(value) {
                Exact(value) | Inexact(value) => Inexact(value),
            },
        }
    }

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
                write!(f, "exact({})", v)
            }
            Inexact(v) => {
                write!(f, "inexact({})", v)
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
