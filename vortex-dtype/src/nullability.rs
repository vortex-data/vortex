// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};
use std::ops::{BitOr, BitOrAssign};

/// Whether an instance of a DType can be `null or not
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Nullability {
    /// Instances of this DType are guaranteed to be non-nullable
    #[default]
    NonNullable,
    /// Instances of this DType may contain a null value
    Nullable,
}

impl Nullability {
    /// Returns `Some(f())` if the the nullability is [`Nullable`](Self::Nullable), otherwise
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_dtype::Nullability::*;
    ///
    /// assert_eq!(NonNullable.is_nullable_then(|| 0), None);
    /// assert_eq!(Nullable.is_nullable_then(|| 0), Some(0));
    /// ```
    ///
    /// ```
    /// # use vortex_dtype::Nullability::*;
    /// #
    /// let mut a = 0;
    ///
    /// Nullable.is_nullable_then(|| { a += 1; });
    /// NonNullable.is_nullable_then(|| { a += 1; });
    ///
    /// // `a` is incremented once because the closure is evaluated lazily by `is_nullable_then`.
    /// assert_eq!(a, 1);
    /// ```c
    ///
    /// Inspired by the [`bool::then`] function.
    pub fn is_nullable_then<T, F: FnOnce() -> T>(self, f: F) -> Option<T> {
        match self {
            Nullability::NonNullable => None,
            Nullability::Nullable => Some(f()),
        }
    }

    /// Returns `Some(t)` if the the nullability is [`Nullable`](Self::Nullable), otherwise returns
    /// `None`.
    ///
    /// Arguments passed to `is_nullable_then_some` are eagerly evaluated; if you are passing the
    /// result of a function call, it is recommended to use [`then`](bool::then), which is lazily
    /// evaluated.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_dtype::Nullability::*;
    ///
    /// assert_eq!(NonNullable.is_nullable_then_some(0), None);
    /// assert_eq!(Nullable.is_nullable_then_some(0), Some(0));
    /// ```
    ///
    /// ```
    /// # use vortex_dtype::Nullability::*;
    /// #
    /// let mut a = 0;
    /// let mut function_with_side_effects = || { a += 1; };
    ///
    /// Nullable.is_nullable_then_some(function_with_side_effects());
    /// NonNullable.is_nullable_then_some(function_with_side_effects());
    ///
    /// // `a` is incremented twice because the value passed to `then_some` is evaluated eagerly.
    /// assert_eq!(a, 2);
    /// ```
    ///
    /// Inspired by the [`bool::then_some`] function.
    pub fn is_nullable_then_some<T>(self, t: T) -> Option<T> {
        match self {
            Nullability::NonNullable => None,
            Nullability::Nullable => Some(t),
        }
    }
}

impl BitOr for Nullability {
    type Output = Nullability;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::NonNullable, Self::NonNullable) => Self::NonNullable,
            _ => Self::Nullable,
        }
    }
}

impl BitOrAssign for Nullability {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs
    }
}

impl From<bool> for Nullability {
    #[inline]
    fn from(value: bool) -> Self {
        if value {
            Self::Nullable
        } else {
            Self::NonNullable
        }
    }
}

impl From<Nullability> for bool {
    #[inline]
    fn from(value: Nullability) -> Self {
        match value {
            Nullability::NonNullable => false,
            Nullability::Nullable => true,
        }
    }
}

impl Display for Nullability {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonNullable => write!(f, ""),
            Self::Nullable => write!(f, "?"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nullability_default() {
        let default = Nullability::default();
        assert_eq!(default, Nullability::NonNullable);
    }

    #[test]
    fn test_nullability_bitor() {
        use Nullability::*;

        // NonNullable | NonNullable = NonNullable
        assert_eq!(NonNullable | NonNullable, NonNullable);

        // NonNullable | Nullable = Nullable
        assert_eq!(NonNullable | Nullable, Nullable);

        // Nullable | NonNullable = Nullable
        assert_eq!(Nullable | NonNullable, Nullable);

        // Nullable | Nullable = Nullable
        assert_eq!(Nullable | Nullable, Nullable);
    }

    #[test]
    fn test_nullability_from_bool() {
        assert_eq!(Nullability::from(false), Nullability::NonNullable);
        assert_eq!(Nullability::from(true), Nullability::Nullable);
    }

    #[test]
    fn test_bool_from_nullability() {
        assert!(!bool::from(Nullability::NonNullable));
        assert!(bool::from(Nullability::Nullable));
    }

    #[test]
    fn test_nullability_roundtrip() {
        // Test roundtrip conversion bool -> Nullability -> bool
        assert!(!bool::from(Nullability::from(false)));
        assert!(bool::from(Nullability::from(true)));

        // Test roundtrip conversion Nullability -> bool -> Nullability
        assert_eq!(
            Nullability::from(bool::from(Nullability::NonNullable)),
            Nullability::NonNullable
        );
        assert_eq!(
            Nullability::from(bool::from(Nullability::Nullable)),
            Nullability::Nullable
        );
    }

    #[test]
    fn test_nullability_chained_bitor() {
        // Test chaining multiple BitOr operations
        let result = Nullability::NonNullable | Nullability::NonNullable | Nullability::NonNullable;
        assert_eq!(result, Nullability::NonNullable);

        let result = Nullability::NonNullable | Nullability::Nullable | Nullability::NonNullable;
        assert_eq!(result, Nullability::Nullable);
    }
}
