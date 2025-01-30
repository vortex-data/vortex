use std::fmt::{Debug, Display, Formatter};

use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::stats::precision::Precision::{Bound, Exact};
use crate::stats::{LowerBound, Stat, UpperBound};

#[derive(Debug, Clone)]
pub enum Precision<T> {
    Exact(T),
    Bound(T),
}

pub fn exact<S: Into<T>, T>(s: S) -> Precision<T> {
    Exact(s.into())
}

pub fn bound<S: Into<T>, T>(s: S) -> Precision<T> {
    Bound(s.into())
}

impl<T: Debug> Precision<T> {
    // This panics if the precision value is not exact
    pub fn unwrap_exact(self) -> T {
        match self {
            Exact(val) => val,
            _ => vortex_panic!("Expected exact value, got value {:?}", self),
        }
    }
}

impl<T: PartialEq> PartialEq for Precision<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Exact(lhs), Exact(rhs)) => lhs == rhs,
            _ => false,
        }
    }
}

impl<T: PartialEq> Precision<T> {
    pub fn structural_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Exact(lhs), Exact(rhs)) | (Bound(lhs), Bound(rhs)) => lhs == rhs,
            _ => false,
        }
    }
}

impl<T> Precision<Option<T>> {
    pub fn transpose(self) -> Option<Precision<T>> {
        match self {
            Exact(Some(x)) => Some(Exact(x)),
            Bound(Some(x)) => Some(Bound(x)),
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
            Bound(value) => Bound(f(value)),
        }
    }

    // Similar to option and then, but if either value is bound, then the whole value is bound.
    pub fn and_then_prefer_bound<U, F: FnOnce(T) -> Precision<U>>(self, f: F) -> Precision<U> {
        match self {
            Exact(value) => f(value),
            Bound(value) => match f(value) {
                Exact(value) | Bound(value) => Bound(value),
            },
        }
    }

    pub fn try_map<U, F: FnOnce(T) -> VortexResult<U>>(self, f: F) -> VortexResult<Precision<U>> {
        let prec = match self {
            Exact(value) => Exact(f(value)?),
            Bound(value) => Bound(f(value)?),
        };
        Ok(prec)
    }

    pub fn as_ref(&self) -> Precision<&T> {
        match self {
            Exact(val) => Exact(val),
            Bound(val) => Bound(val),
        }
    }

    pub fn value(&self) -> &T {
        match self {
            Exact(val) | Bound(val) => val,
        }
    }

    pub fn into_value(self) -> T {
        match self {
            Exact(val) | Bound(val) => val,
        }
    }

    pub fn with_direction(self, direction: BoundDirection) -> DirectionalBound<T> {
        DirectionalBound::new(direction, self)
    }

    pub fn with_stat(self, stat: Stat) -> DirectionalBound<T> {
        DirectionalBound::new(stat.direction(), self)
    }
}

impl<T: Display> Display for Precision<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Exact(v) => {
                write!(f, "exact({})", v)
            }
            Bound(v) => {
                write!(f, "bound({})", v)
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

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum BoundDirection {
    Upper,
    Lower,
    Neither,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DirectionalBound<T> {
    pub(crate) direction: BoundDirection,
    pub(crate) value: Precision<T>,
}

impl<T> DirectionalBound<T> {
    pub(crate) fn lower_ok(self) -> Option<LowerBound<T>> {
        match self.direction {
            BoundDirection::Lower => Some(LowerBound(self.value)),
            _ => None,
        }
    }

    pub(crate) fn upper_ok(self) -> Option<UpperBound<T>> {
        match self.direction {
            BoundDirection::Upper => Some(UpperBound(self.value)),
            _ => None,
        }
    }
}

impl<T> DirectionalBound<T> {
    pub fn new(direction: BoundDirection, value: Precision<T>) -> Self {
        Self { direction, value }
    }

    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> DirectionalBound<U> {
        DirectionalBound {
            direction: self.direction,
            value: self.value.map(f),
        }
    }

    pub fn try_map<U, F: FnOnce(T) -> VortexResult<U>>(
        self,
        f: F,
    ) -> VortexResult<DirectionalBound<U>> {
        Ok(DirectionalBound {
            direction: self.direction,
            value: self.value.try_map(f)?,
        })
    }

    pub fn value(&self) -> &Precision<T> {
        &self.value
    }

    pub fn into_value(self) -> Precision<T> {
        self.value
    }
}

impl DirectionalBound<ScalarValue> {
    pub fn into_scalar(self, dtype: DType) -> DirectionalBound<Scalar> {
        self.map(|v| Scalar::new(dtype, v))
    }
}

impl DirectionalBound<&ScalarValue> {
    pub fn into_scalar(self, dtype: DType) -> DirectionalBound<Scalar> {
        self.map(|v| Scalar::new(dtype, v.clone()))
    }
}
