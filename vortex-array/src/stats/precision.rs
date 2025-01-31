use std::fmt::{Debug, Display, Formatter};

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_scalar::{Scalar, ScalarValue};

use crate::stats::precision::Precision::{Exact, Inexact};

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

// #[allow(dead_code)]
// pub fn take<T, F>(mut_ref: &mut T, closure: F)
// where
//     F: FnOnce(T) -> T,
// {
//     use std::ptr;
//
//     unsafe {
//         let old_t = ptr::read(mut_ref);
//         let new_t = panic::catch_unwind(panic::AssertUnwindSafe(|| closure(old_t)))
//             .unwrap_or_else(|_| ::std::process::abort());
//         ptr::write(mut_ref, new_t);
//     }
// }

// impl<T> Precision<T> {
// pub fn mut_inexact(&mut self) {
// take(self, |v| match v {
//     Exact(val) => Inexact(val),
//     Inexact(val) => Inexact(val),
// })
// }
// }

impl<T: Clone> Precision<T> {
    pub fn mut_inexact(&mut self) {
        match self {
            Exact(val) => *self = Inexact(val.clone()),
            Inexact(_) => (),
        };
    }
}

impl<T> Precision<T> {
    pub fn ok_exact(self) -> Option<T> {
        match self {
            Exact(val) => Some(val),
            _ => None,
        }
    }

    pub fn ok_inexact(self) -> Option<T> {
        match self {
            Inexact(val) => Some(val),
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
