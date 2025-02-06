use std::cmp::Ordering;
use std::fmt::{Display, Formatter};

/// Whether an instance of a DType can be `null or not
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Nullability {
    /// Instances of this DType are guaranteed to be non-nullable
    #[default]
    NonNullable,
    /// Instances of this DType may contain a null value
    Nullable,
}

impl PartialOrd for Nullability {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::NonNullable, Self::NonNullable) => Some(Ordering::Equal),
            (Self::NonNullable, Self::Nullable) => Some(Ordering::Greater),
            (Self::Nullable, Self::NonNullable) => Some(Ordering::Less),
            (Self::Nullable, Self::Nullable) => Some(Ordering::Equal),
        }
    }
}

impl Ord for Nullability {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl From<bool> for Nullability {
    fn from(value: bool) -> Self {
        if value {
            Self::Nullable
        } else {
            Self::NonNullable
        }
    }
}

impl From<Nullability> for bool {
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
    use std::cmp::{min, Ordering};

    use crate::Nullability::{NonNullable, Nullable};

    #[test]
    fn test_max_dtype() {
        assert_eq!(Nullable.partial_cmp(&NonNullable), Some(Ordering::Less));
        assert!(Nullable <= NonNullable);
        // assert!(Nullable  NonNullable);
        assert!(Nullable != NonNullable);
        assert_eq!(Nullable, min(Nullable, NonNullable));
        assert_eq!(NonNullable, min(NonNullable, NonNullable));

        assert_eq!(Nullable, min(Nullable, Nullable));
    }
}
