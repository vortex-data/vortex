use std::fmt::{Display, Formatter};
use std::ops::BitOr;

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
    /// A self-describing displayed form.
    ///
    /// The usual Display renders [Nullability::NonNullable] as the empty string.
    pub fn verbose_display(&self) -> impl Display {
        match self {
            Nullability::NonNullable => "NonNullable",
            Nullability::Nullable => "Nullable",
        }
    }
}

impl BitOr for Nullability {
    type Output = Nullability;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::NonNullable, Self::NonNullable) => Self::NonNullable,
            _ => Self::Nullable,
        }
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
